# -*- coding: utf-8 -*-
"""
rev_a_fe.py — P1-A：FE 标签重训级Ⅱ + 独立 FE 验证（回应评审 #1 / mandatory ①）
关键逻辑：
 1) 训练数据：输入 = 粗网格(4×4) BFS-FE 节点 DOF（fem_solve），
    标签 = 细网格(nf=16) FE 参考场（在粗单元采样点求值）→ "FE-generated labels"
 2) 训练 twin operator（同一架构/训练器/种子，仅标签源不同）
 3) 独立验证（细 FE 参考 + 未参与训练载荷）：
    (a) exact DOF / PINN DOF / FE-DOF 三类输入下 op_fe 与基线 op 对比
    (b) 同 FE-DOF 输入下 OP 与经典 BFS 后处理对比（证明逼近的是 FE 场而非纯解析场）
落盘 supp_review.fe_labels
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
import eval_symavg
from rev_common import (RES, coarse_dof, exact_dof_arr, fe_field_grid, fe_ref_field,
                        assemble_bfs, l2rel, analytic_metrics, mxy_center_max,
                        load_json, save_json, xs_g)

NF = 16          # 细网格 FE 参考
NC = S.N_MESH    # 粗网格 = 工作网格 4×4


def make_fe_dataset(qs, nf=NF, nc=NC, n_pts=10, seed=1):
    """样本结构与 make_operator_dataset 一致，但 de=粗 FE DOF、标签=细 FE 场"""
    n = nc
    Lex = A_n = S.A / n
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, n_pts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    XI = XI.ravel(); ETA = ETA.ravel()
    rng = np.random.default_rng(seed)
    npts = XI.size
    PX = np.concatenate([xs[ei] + 0.5 * Lex * (XI + 1.0) for ei in range(n) for _ in range(n)])
    PY = np.concatenate([ys[ej] + 0.5 * Lex * (ETA + 1.0) for _ in range(n) for ej in range(n)])
    samples = []
    for q in qs:
        d_fem, _, _ = S.fem_solve(n, q)                       # 粗 FE DOF
        dnode = d_fem.reshape(n + 1, n + 1, 4)                # (n+1,n+1,4)
        d_fine, _, _ = S.fem_solve(nf, q)                     # 细 FE 参考
        zeta = rng.choice([-1.0, 1.0], size=PX.size)
        (w_all, kx_a, ky_a, kxy_a, sx_a, sy_a, sxy_a) = fe_field_grid(nf, d_fine, PX, PY, zeta)
        for c, (ei, ej) in enumerate([(i, j) for i in range(n) for j in range(n)]):
            de = dnode[ei:ei + 2, ej:ej + 2, :].reshape(16)   # 与 ids 顺序一致? (见下)
            # 逐点顺序：corner [(0,0),(1,0),(1,1),(0,1)] 各自 4 DOF
            corners = np.array([dnode[ei, ej], dnode[ei + 1, ej],
                                dnode[ei + 1, ej + 1], dnode[ei, ej + 1]])
            de = corners.ravel()                               # (16,)
            sl = slice(c * npts, (c + 1) * npts)
            samples.append((de, np.stack([XI, ETA], axis=1), zeta[sl],
                            w_all[sl], np.stack([kx_a[sl], ky_a[sl], kxy_a[sl]], axis=1),
                            np.stack([sx_a[sl], sy_a[sl], sxy_a[sl]], axis=1), float(q)))
    return samples


def assemble_op(d_all, op, npts=15):
    return eval_symavg._assemble(d_all, op, scale=None, npts=npts)


def fe_truth(q, nf=NF):
    """返回 truth 调用器（X,Y 展平）"""
    d, _, _ = S.fem_solve(nf, q)

    def wrap(key):
        def fn(X, Y):
            out = fe_field_grid(nf, d, X, Y, None if key == "w" else 1.0)
            idx = ["w", "kx", "ky", "kxy", "sx", "sy", "sxy"].index(key)
            return out[idx]
        return fn
    return {k: wrap(k) for k in ["w", "kx", "ky", "kxy", "sx", "sy", "sxy"]}


def eval_case(op, dof, q, npts=15):
    F = assemble_op(dof, op, npts=npts)
    am = analytic_metrics(F, q, npts=npts)
    return F, am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--loads", type=int, default=120)
    ap.add_argument("--out", default="op_state_fe.pt")
    args = ap.parse_args()
    t_start = time.time()

    # ---- 训练载荷（与主实验同分布：q∈[0.5,0.9]∪[1.1,2.0] 120 个，避开 [0.9,1.1]）
    qs_all = np.random.default_rng(3).uniform(S.Q_LO, S.Q_HI, 400)
    qs_train = qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:args.loads]
    print(f"[P1-A] FE-label 数据集：{len(qs_train)} 工况 × 16 单元；细网格 FE nf={NF}")

    samples = make_fe_dataset(qs_train, nf=NF)
    print(f"[P1-A] 样本数 {len(samples)}，开始训练（{args.epochs} epoch）")
    op_fe = S.PlateOperator(p=64, hidden=64)
    Lex = S.A / NC
    hist = S.train_operator(op_fe, samples, n_epochs=args.epochs, batch=24,
                            Le=Lex, scale=None, lam_w=1.0)
    torch.save(op_fe.state_dict(), os.path.join(RES, args.out))
    print(f"[P1-A] op_fe 训练完成 {time.time()-t_start:.0f}s")

    # ---- 基线算子
    op0 = S.PlateOperator(p=64, hidden=64)
    op0.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"),
                                   weights_only=False))
    op0.eval(); op_fe.eval()

    # ---- 独立验证载荷（含留出带 [0.9,1.1] 与训练谱边缘）
    q_hold = [0.60, 0.80, 0.95, 1.00, 1.30, 1.60, 1.90]

    # q=1 细项：三类输入 × 两算子 vs analytic & fine-FE
    row = {}
    for dof_name, dof_fn in [("exact", lambda: exact_dof_arr(1.0)),
                             ("pinn", None)]:
        pass
    # PINN DOF
    pinn = eval_poisson_pinn()
    dpin = pinn_dof_at(1.0, pinn)
    d_ex = exact_dof_arr(1.0)
    d_fe = coarse_dof(1.0)

    print("\n===== q=1.0 闭环：三输入源 × 双算子（vs 解析 & vs 细 FE 参考）=====")
    res1 = {"q": 1.0}
    for iname, dd in [("exactDOF", d_ex), ("pinnDOF", dpin), ("feDOF", d_fe)]:
        for oname, op in [("op_fe", op_fe), ("op_nav", op0)]:
            F = assemble_op(dd, op, npts=15)
            am = analytic_metrics(F, 1.0, npts=15)
            tr = fe_truth(1.0)
            # κ/σ vs 细 FE（同一坐标）
            Xr, Yr = F["X"].ravel(), F["Y"].ravel()
            kx_t, ky_t, kxy_t = [tr[k](Xr, Yr) for k in ("kx", "ky", "kxy")]
            sx_t, sy_t, sxy_t = [tr[k](Xr, Yr) for k in ("sx", "sy", "sxy")]
            m_fe = {
                "k": l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                           np.concatenate([kx_t, ky_t, kxy_t])),
                "s": l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                           np.concatenate([sx_t, sy_t, sxy_t]))}
            w_star, x_u = S.wstar_from_field(F, npts=15)
            XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
            w_fe_ref = tr["w"](XU.ravel(), YU.ravel()).reshape(XU.shape)
            m_fe["w_star"] = l2rel(w_star.ravel(), w_fe_ref.ravel())
            key = f"{iname}|{oname}"
            res1[key] = {"vs_analytic": am, "vs_fineFE": m_fe,
                         "mxy_center": mxy_center_max(F)}
            print(f"  {iname:8s} {oname:8s}  "
                  f"vsAnalytic w*={am['w_star']:.2e} κ={am['k']:.2e} σ={am['s']:.2e} | "
                  f"vsFineFE w*={m_fe['w_star']:.2e} κ={m_fe['k']:.2e} σ={m_fe['s']:.2e}")

    # FE-DOF 输入：BFS 后处理对比（经典 FE 环节）
    print("\n===== FE-DOF 输入：OP(fe) vs OP(nav) vs BFS 后处理（细 FE 参考为真值）=====")
    val_fe = []
    for qv in q_hold:
        dq = coarse_dof(qv)
        tr = fe_truth(qv)
        Fbf = assemble_bfs(dq, n=NC, npts=15)
        Fof = assemble_op(dq, op_fe, npts=15)
        Fon = assemble_op(dq, op0, npts=15)
        Xr, Yr = Fof["X"].ravel(), Fof["Y"].ravel()
        kx_t, ky_t, kxy_t = [tr[k](Xr, Yr) for k in ("kx", "ky", "kxy")]
        sx_t, sy_t, sxy_t = [tr[k](Xr, Yr) for k in ("sx", "sy", "sxy")]
        rowq = {"q": qv}
        for nm, F in [("bfs", Fbf), ("op_fe", Fof), ("op_nav", Fon)]:
            rowq[nm] = {
                "k": l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                           np.concatenate([kx_t, ky_t, kxy_t])),
                "s": l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                           np.concatenate([sx_t, sy_t, sxy_t]))}
        val_fe.append(rowq)
        print(f"  q={qv:.2f}  κ: bfs={rowq['bfs']['k']:.2e} "
              f"op_fe={rowq['op_fe']['k']:.2e} op_nav={rowq['op_nav']['k']:.2e} | "
              f"σ: bfs={rowq['bfs']['s']:.2e} op_fe={rowq['op_fe']['s']:.2e} "
              f"op_nav={rowq['op_nav']['s']:.2e}")

    R = load_json()
    sr = R.setdefault("supp_review", {})
    sr["fe_labels"] = {
        "note": ("Level-II retrained on FE-generated labels: inputs = coarse 4x4 BFS-FE nodal DOFs "
                 "(fem_solve), labels = fine 16x16 BFS-FE reference fields evaluated at coarse-element "
                 "sample points. Same architecture/seeds/loads as the Navier-label operator."),
        "nf_fine": NF, "n_train_loads": len(qs_train), "epochs": args.epochs,
        "q1_closed_loop": res1,
        "fe_dof_input_vs_fineFE": val_fe,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    save_json(R)
    print("\n[P1-A] saved supp_review.fe_labels")


def pinn_dof_at(q, pinn):
    xs = np.linspace(0, S.A, S.N_MESH + 1); ys = np.linspace(0, S.B, S.N_MESH + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij")
    Xnr, Ynr = Xnr.ravel(), Ynr.ravel()
    arr = np.array(S.pinn_field(pinn, np.stack([Xnr, Ynr], axis=1), q))
    return arr.T


def eval_poisson_pinn():
    import eval_poisson as EP
    return EP._load_pinn()


if __name__ == "__main__":
    main()
