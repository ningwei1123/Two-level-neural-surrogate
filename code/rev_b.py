# -*- coding: utf-8 -*-
"""
rev_b.py — P1-B：8×8 / 16×16 网格缩放（回应评审 #3 / mandatory ② + Q1/Q4）
做法：
 · 级Ⅰ PINN 网格无关：同一训练网络在任意 n×n 节点求 DOF（label-free）——直接测 8/16 精度
 · 级Ⅱ 算子按工作网格重训（Le 相关），训练载荷按网格规模缩减并如实记录
 · 计时（同机、单载荷、perf_counter、中位数 3 次）：
     FE   = fem_solve(n,q)（dense 组装+求解，每工况全新）
     代理 = 级Ⅰ节点DOF求值 + 全单元算子前向(npts=11) + Poisson 解码
 · 误差：q=1 与 6 个未见载荷下（exact/PINN DOF 输入）κ/σ/w* 全局 L2
落盘 supp_review.scaling
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
from rev_common import (RES, exact_dof_arr, analytic_metrics, l2rel,
                        load_json, save_json)
from eval_poisson import _load_pinn

torch.set_num_threads(8)


def unique_idx(n_el, npts):
    keep = []
    for e in range(n_el):
        keep += list(range(e * npts, e * npts + npts - 1))
    keep.append(n_el * npts - 1)
    return np.asarray(keep, dtype=np.int64)


def assemble_op_grid(d_all, op, n_el, npts=15):
    """参数化 n_el 的算子装配（eval_symavg._assemble 的推广）"""
    n1 = npts
    W = np.zeros((n_el * n1, n_el * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W); WX = np.zeros_like(W); WY = np.zeros_like(W)
    Lex = S.A / n_el
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    for ei in range(n_el):
        for ej in range(n_el):
            ids = [(ei + di) * (n_el + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d_all.reshape(-1, 4)[ids].ravel()
            d_t = torch.tensor(de, dtype=torch.float64).reshape(1, 16)
            with torch.no_grad():
                wk = op.wk_at(d_t, torch.tensor(pts, dtype=torch.float64)).numpy()
                sp = op.s_at(d_t, torch.stack(
                    [torch.tensor(pts[:, 0], dtype=torch.float64),
                     torch.tensor(pts[:, 1], dtype=torch.float64),
                     torch.ones(pts.shape[0])], dim=1)).numpy()
            sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = wk[:, 0].reshape(n1, n1)
            KX[sl_i, sl_j] = wk[:, 1].reshape(n1, n1)
            KY[sl_i, sl_j] = wk[:, 2].reshape(n1, n1)
            KXY[sl_i, sl_j] = wk[:, 3].reshape(n1, n1)
            SX[sl_i, sl_j] = sp[:, 0].reshape(n1, n1)
            SY[sl_i, sl_j] = sp[:, 1].reshape(n1, n1)
            SXY[sl_i, sl_j] = sp[:, 2].reshape(n1, n1)
            xg = xs_pts(ei, n_el, n1, pts[:, 0]); yg = xs_pts(ej, n_el, n1, pts[:, 1])
            WX[sl_i, sl_j] = xg.reshape(n1, n1); WY[sl_i, sl_j] = yg.reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)


def assemble_op_batch(d_all, op, n_el, npts=15):
    """批量装配：trunk 特征只依赖单元局部网格（全场共享），branch 可整批前向。
    输出与逐单元循环版逐点一致（无梯度）。"""
    n1 = npts
    Ne = n_el * n_el
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)     # (m,2), m=n1²
    de_all = np.zeros((Ne, 16))
    for e in range(Ne):
        ei, ej = divmod(e, n_el)
        ids = [(ei + di) * (n_el + 1) + (ej + dj) for (di, dj) in
               [(0, 0), (1, 0), (1, 1), (0, 1)]]
        de_all[e] = d_all.reshape(-1, 4)[ids].ravel()
    with torch.no_grad():
        b = op.branch(torch.tensor(de_all, dtype=torch.float64))          # (Ne,p)
        T_wk = op.trunk_wk(torch.tensor(pts, dtype=torch.float64))        # (m,4p)
        T_wk = T_wk.reshape(n1 * n1, 4, op.p)
        wk = torch.einsum("ep,mkp->emk", b, T_wk) + op.b0[0:4].reshape(1, 1, 4)
        T_s = op.trunk_s(torch.tensor(np.stack(
            [pts[:, 0], pts[:, 1], np.ones(pts.shape[0])], axis=1),
            dtype=torch.float64)).reshape(n1 * n1, 3, op.p)
        sp = torch.einsum("ep,mkp->emk", b, T_s) + op.b0[4:7].reshape(1, 1, 3)
        wk = wk.numpy(); sp = sp.numpy()                  # (Ne,m,4),(Ne,m,3)

    def place(A, field_dim):
        A = A.reshape(n_el, n_el, n1, n1, field_dim)
        return {k: np.ascontiguousarray(
            A[:, :, :, :, j].transpose(0, 2, 1, 3).reshape(n_el * n1, n_el * n1))
            for j, k in enumerate(["W", "KX", "KY", "KXY"])} if field_dim == 4 else None
    res = place(wk, 4)
    sps = sp.reshape(n_el, n_el, n1, n1, 3)
    for j, k in enumerate(["SX", "SY", "SXY"]):
        res[k] = np.ascontiguousarray(
            sps[:, :, :, :, j].transpose(0, 2, 1, 3).reshape(n_el * n1, n_el * n1))
    rowx = np.concatenate([(ei + np.arange(n1) / (n1 - 1)) / n_el
                           for ei in range(n_el)])
    WX = np.broadcast_to(rowx[:, None], (n_el * n1, n_el * n1))
    WY = np.broadcast_to(rowx[None, :], (n_el * n1, n_el * n1))
    res["X"], res["Y"] = WX.copy(), WY.copy()
    return res


def xs_pts(ei, n_el, n1, xi):
    Lex = S.A / n_el
    return ei * Lex + 0.5 * Lex * (xi + 1.0)


def grid_metrics(F, q, n_el, npts=15):
    keep = unique_idx(n_el, npts)
    kx = F["KX"][np.ix_(keep, keep)]; ky = F["KY"][np.ix_(keep, keep)]
    M = keep.size
    x = np.linspace(0.0, S.A, M)
    h = x[1] - x[0]
    w_star = S.poisson_reconstruct_w(kx, ky, h)
    Xr, Yr = F["X"].ravel(), F["Y"].ravel()
    kx_t, ky_t, kxy_t = S.curv_analytic(Xr, Yr, q)
    sx_t, sy_t, sxy_t = S.stress_analytic(Xr, Yr, 1.0, q)
    XU, YU = np.meshgrid(x, x, indexing="ij")
    return dict(
        k=l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                np.concatenate([kx_t, ky_t, kxy_t])),
        s=l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                np.concatenate([sx_t, sy_t, sxy_t])),
        w_star=l2rel(w_star.ravel(), S.w_analytic(XU.ravel(), YU.ravel(), q)))


def pinn_dof_grid(pinn, q, n_el):
    xs = np.linspace(0, S.A, n_el + 1); ys = np.linspace(0, S.B, n_el + 1)
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    arr = np.array(S.pinn_field(pinn, np.stack([Xn.ravel(), Yn.ravel()], axis=1), q))
    return arr.T


def time_med(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def level1_grid_errors(pinn, n_el, q=1.0):
    xs = np.linspace(0, S.A, n_el + 1); ys = np.linspace(0, S.B, n_el + 1)
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    Xr, Yr = Xn.ravel(), Yn.ravel()
    d = pinn_dof_grid(pinn, q, n_el)
    wt = S.w_analytic(Xr, Yr, q)
    wxt, wyt = S._w_grad(Xr, Yr, q)
    kxyt = S.curv_analytic(Xr, Yr, q)[2]
    return dict(
        w_max=float(np.abs(d[:, 0] - wt).max()),
        wx_max=float(np.abs(d[:, 1] - wxt).max()),
        wy_max=float(np.abs(d[:, 2] - wyt).max()),
        wxy_max=float(np.abs(d[:, 3] - (-kxyt)).max()))


def train_grid_op(n_el, qs, epochs, n_pts=10):
    samples = S.make_operator_dataset(qs, n=n_el, n_pts=n_pts, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=epochs, batch=24, Le=S.A / n_el,
                     scale=None, lam_w=1.0, log_every=1000)
    dt = time.time() - t0
    return op, dt, len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="8,16")
    ap.add_argument("--epochs8", type=int, default=2500)
    ap.add_argument("--epochs16", type=int, default=2000)
    ap.add_argument("--loads8", type=int, default=80)
    ap.add_argument("--loads16", type=int, default=60)
    args = ap.parse_args()

    pinn = _load_pinn()
    op4 = S.PlateOperator(p=64, hidden=64)
    op4.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"),
                                   weights_only=False))
    op4.eval()
    rng = np.random.default_rng(3)
    qs_all = rng.uniform(S.Q_LO, S.Q_HI, 800)
    qs_tr = qs_all[(qs_all < 0.9) | (qs_all > 1.1)]
    rng2 = np.random.default_rng(7)
    q_hold = np.concatenate([rng2.uniform(0.5, 0.9, 3), rng2.uniform(1.1, 2.0, 3)])

    out = {}
    for ng in [4] + [int(x) for x in args.grids.split(",")]:
        if ng == 4:
            op = op4
            epochs = n_train = None
        else:
            epochs = args.epochs8 if ng == 8 else args.epochs16
            n_loads = args.loads8 if ng == 8 else args.loads16
            qs = qs_tr[:n_loads]
            print(f"\n[P1-B] grid {ng}×{ng}: 训练 {len(qs)} loads × {epochs} ep ...")
            op, dt, ns = train_grid_op(ng, qs, epochs)
            torch.save(op.state_dict(), os.path.join(RES, f"op_state_g{ng}.pt"))
            print(f"   train done {dt:.0f}s, samples {ns}")
            n_train = len(qs)
        op.eval()

        # 计时
        fe_ms = time_med(lambda: S.fem_solve(ng, 1.0)) * 1e3
        def surrogate():
            d = pinn_dof_grid(pinn, 1.0, ng)
            assemble_op_batch(d, op, ng, npts=11)
        surr_ms = time_med(surrogate) * 1e3
        naive_ms = None
        if ng in (4, 8):
            naive_ms = time_med(lambda: (lambda d: assemble_op_grid(d, op, ng, npts=11))(
                pinn_dof_grid(pinn, 1.0, ng))) * 1e3

        # 误差：q=1（exact/PINN DOF）与 6 未见载荷（exact）——批量装配
        e1 = {}
        de = exact_dof_arr(1.0, n=ng)
        Fe = assemble_op_batch(de, op, ng, npts=11)
        e1["exact"] = grid_metrics(Fe, 1.0, ng, npts=11)
        dp = pinn_dof_grid(pinn, 1.0, ng)
        Fp = assemble_op_batch(dp, op, ng, npts=11)
        e1["pinn"] = grid_metrics(Fp, 1.0, ng, npts=11)
        gen = {"w_star": [], "k": [], "s": []}
        for qv in q_hold:
            dq = exact_dof_arr(qv, n=ng)
            Fq = assemble_op_batch(dq, op, ng, npts=11)
            gm = grid_metrics(Fq, qv, ng, npts=11)
            gen["w_star"].append(gm["w_star"]); gen["k"].append(gm["k"]); gen["s"].append(gm["s"])
        gen = {k: float(np.mean(v)) for k, v in gen.items()}
        l1 = level1_grid_errors(pinn, ng) if ng >= 8 else None
        out[str(ng)] = {
            "fe_ms": fe_ms, "surrogate_ms": surr_ms, "surrogate_naive_loop_ms": naive_ms,
            "speedup_x": float(np.round(fe_ms / max(surr_ms, 1e-9), 2)),
            "epochs": epochs, "n_train_loads": n_train,
            "q1": e1, "gen6_mean_exact": gen,
            "level1_node_err": l1,
        }
        print(f"[P1-B {ng}×{ng}] FE={fe_ms:.2f}ms surrogate={surr_ms:.2f}ms "
              f"(x{out[str(ng)]['speedup_x']}) | q1 exact κ={e1['exact']['k']:.2e} "
              f"σ={e1['exact']['s']:.2e} w*={e1['exact']['w_star']:.2e}")

    R = load_json()
    R.setdefault("supp_review", {})["scaling"] = {
        "note": ("Same-machine median wall-clock per load: FE = dense BFS assembly+solve "
                 "(fem_solve, fresh per load); surrogate = Level-I nodal DOF evaluation + "
                 "element-local operator forward (npts=11) + Poisson decode. Level-II is "
                 "retrained per grid (element size enters the physical consistency terms)."),
        "grids": out,
    }
    save_json(R)
    print("\n[P1-B] saved supp_review.scaling")


if __name__ == "__main__":
    main()
