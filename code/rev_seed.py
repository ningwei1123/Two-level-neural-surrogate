# -*- coding: utf-8 -*-
"""
rev_seed.py — R3 #3 统计严谨性：多种子闭环评测
==================================================================
对每个随机种子重跑「级I 能量 PINN + 级II DeepONet + Poisson 解码」闭环，
评测口径与 shell_system.main 一致（单位方板 4×4、q=1 闭环 npts=15、
30 个未见载荷 exact-DOF 泛化 npts=11），产出每个种子的：
  pinn : 全场 L2 / max |Δw| / 节点误差
  closed.pinn_op / closed.pinn_bfs : 同输入(q=1, PINN 节点自由度)场误差
  gen30 : 30 工况 w_star / k / s 的均值与 CV（exact DOF）
逐种子写 results/seeds/seed_<s>.json（可断点续跑：已存在则跳过）。

用法:  python rev_seed.py --seeds 2026,42,7 --ep-pinn 8000 --ep-op 4000
      --lbfgs 1200 --no-holdout --nproc-label 'run1' --threads 8
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
SEED_DIR = os.path.join(RES, "seeds")
OUTDIR = SEED_DIR          # 可被 --outdir 覆盖
os.makedirs(SEED_DIR, exist_ok=True)

import numpy as np
import torch

import shell_system as S          # 设置默认 dtype/线程；导入后我们会重置种子
import rev_b
from rev_common import exact_dof_arr, l2rel


def seed_pinn(seed, ep_pinn, lbfgs):
    # 播种由 run_seed 统一执行一次；此处不重置，保证级Ⅰ→级Ⅱ 延续同一 RNG 状态
    # （与 shell_system.main 原流程一致：import 播种后级Ⅰ 训练推进状态，级Ⅱ 继续）
    torch.manual_seed(seed); np.random.seed(seed)
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    t0 = time.time()
    hist = S.train_pinn(pinn, n_epochs=ep_pinn, n_int=4000, lbfgs_iter=lbfgs)
    return pinn, time.time() - t0, hist


def seed_operator(seed, ep_op, qs_train, n=4):
    # 注意：不在级Ⅱ 前重置 RNG —— 复现主稿 op（级Ⅰ 后的延续状态）
    samples = S.make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=ep_op, batch=24, Le=S.A / n,
                     scale=None, lam_w=1.0, log_every=1000)
    return op, time.time() - t0, len(samples)


def bfs_assemble(d_all, n=4, npts=15):
    """BFS 同输入装配（与 shell_system.main 的 assemble 一致，仅 BFS 分支）"""
    n1 = npts
    Lex, Ley = S.A / n, S.B / n
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    W = np.zeros((n * n1, n * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W)
    WX = np.zeros_like(W); WY = np.zeros_like(W)
    for ei in range(n):
        for ej in range(n):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d_all.reshape(-1, 4)[ids].ravel()
            w_bf, kxx, kyy, kxy = S.bfs_interp(de, pts[:, 0], pts[:, 1], Lex, Ley)
            mx = S.D_FLEX * (kxx + S.NU * kyy); my = S.D_FLEX * (kyy + S.NU * kxx)
            mxy = S.D_FLEX * (1 - S.NU) * kxy
            s_bf = (6.0 / S.H_TH ** 2) * np.stack([mx, my, mxy], axis=1)
            sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = w_bf.reshape(n1, n1)
            KX[sl_i, sl_j] = kxx.reshape(n1, n1); KY[sl_i, sl_j] = kyy.reshape(n1, n1)
            KXY[sl_i, sl_j] = kxy.reshape(n1, n1)
            SX[sl_i, sl_j] = s_bf[:, 0].reshape(n1, n1)
            SY[sl_i, sl_j] = s_bf[:, 1].reshape(n1, n1)
            SXY[sl_i, sl_j] = s_bf[:, 2].reshape(n1, n1)
            # 真坐标：单元内等参映射（相邻单元端点重合），不能按 linspace 均匀铺
            WX[sl_i, sl_j] = ((ei + 0.5 * (pts[:, 0] + 1.0)) * Lex).reshape(n1, n1)
            WY[sl_i, sl_j] = ((ej + 0.5 * (pts[:, 1] + 1.0)) * Ley).reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)


def field_metrics(F, q, npts):
    """对装配场（方板均匀网格）计算 w/kappa/sigma 相对 L2 与解码 w*"""
    if "X" in F and "Y" in F:
        # 装配场自带真坐标（单元内等参、跨单元端点重合）
        XR, YR = F["X"].ravel(), F["Y"].ravel()
    else:
        Xr = np.linspace(0, S.A, F["W"].shape[0])
        Yr = np.linspace(0, S.B, F["W"].shape[1])
        XR, YR = np.meshgrid(Xr, Yr, indexing="ij")
        XR = XR.ravel(); YR = YR.ravel()
    kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, q)
    sx_t, sy_t, sxy_t = S.stress_analytic(XR, YR, 1.0, q)
    rec = dict(
        w=l2rel(F["W"].ravel(), S.w_analytic(XR, YR, q)),
        kxx=l2rel(F["KX"].ravel(), kx_t), kyy=l2rel(F["KY"].ravel(), ky_t),
        kxy=l2rel(F["KXY"].ravel(), kxy_t),
        sxx=l2rel(F["SX"].ravel(), sx_t), syy=l2rel(F["SY"].ravel(), sy_t),
        sxy=l2rel(F["SXY"].ravel(), sxy_t),
        k=l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(),
                                F["KXY"].ravel()]),
                np.concatenate([kx_t, ky_t, kxy_t])),
        s=l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(),
                                F["SXY"].ravel()]),
                np.concatenate([sx_t, sy_t, sxy_t])))
    if "KX" in F:
        keep = rev_b.unique_idx(4, npts)
        kx = F["KX"][np.ix_(keep, keep)]; ky = F["KY"][np.ix_(keep, keep)]
        M = keep.size
        x = np.linspace(0.0, S.A, M); h = x[1] - x[0]
        w_star = S.poisson_reconstruct_w(kx, ky, h)
        XU, YU = np.meshgrid(x, x, indexing="ij")
        rec["w_star"] = l2rel(w_star.ravel(), S.w_analytic(XU.ravel(), YU.ravel(), q))
    return rec


def run_seed(seed, ep_pinn, ep_op, lbfgs, no_holdout, n=4, quiet=False):
    out_path = os.path.join(OUTDIR, f"seed_{seed}.json")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            return json.load(f), 0.0

    # ---- 训练载荷集合（与 shell_system.main 一致） ----
    rng = np.random.default_rng(3)
    qs_all = rng.uniform(S.Q_LO, S.Q_HI, 200)
    if no_holdout:
        qs_train = np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])
    else:
        qs_train = qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:120]

    t_all = time.time()
    pinn, tp, _ = seed_pinn(seed, ep_pinn, lbfgs)
    if not quiet:
        print(f"[{seed}] Level-I done in {tp/60:.1f} min", flush=True)
    op, to_, ns = seed_operator(seed, ep_op, qs_train, n=n)
    if not quiet:
        print(f"[{seed}] Level-II done in {to_/60:.1f} min (samples {ns})", flush=True)

    # ---- 级I 场与节点误差 ----
    ng = 61
    xg = np.linspace(0, S.A, ng)
    XG, YG = np.meshgrid(xg, xg, indexing="ij")
    w_pin, _, _, _ = S.pinn_field(pinn, np.stack([XG.ravel(), YG.ravel()], axis=1), 1.0)
    w_ex = S.w_analytic(XG.ravel(), YG.ravel(), 1.0)
    e_pinn_l2 = float(l2rel(w_pin, w_ex))
    e_pinn_max = float(np.abs(w_pin - w_ex).max())
    l1 = rev_b.level1_grid_errors(pinn, n, q=1.0)      # 节点误差（4×4 网格）
    # PINN 节点 DOF（q=1，n×n 网格）
    d_pin = np.array(S.pinn_field(
        pinn, np.stack([x for x in np.meshgrid(np.linspace(0, S.A, n + 1),
                                               np.linspace(0, S.B, n + 1),
                                               indexing="ij")], axis=-1).reshape(-1, 2),
        1.0)).T.ravel()

    # ---- 闭环 q=1：算子与 BFS 同输入（PINN DOF） ----
    Fop = rev_b.assemble_op_grid(d_pin, op, n, npts=15)
    Fbf = bfs_assemble(d_pin, n=n, npts=15)
    # 存权重与装配场：以后复算指标不必再训练
    torch.save(pinn.state_dict(), os.path.join(OUTDIR, f"seed_{seed}_pinn.pt"))
    torch.save(op.state_dict(), os.path.join(OUTDIR, f"seed_{seed}_op.pt"))
    np.savez_compressed(
        os.path.join(OUTDIR, f"seed_{seed}_fields.npz"),
        d_pin=d_pin,
        **{"op_" + k: v for k, v in Fop.items()},
        **{"bfs_" + k: v for k, v in Fbf.items()})
    cl_op = field_metrics(Fop, 1.0, 15)
    cl_bf = field_metrics(Fbf, 1.0, 15)

    # ---- 30 未见载荷泛化（exact DOF） ----
    rng11 = np.random.default_rng(11)
    qs_test = np.concatenate([rng11.uniform(0.5, 0.9, 15), rng11.uniform(1.1, 2.0, 15)])
    wl, kl, sl = [], [], []
    for qv in qs_test:
        dq = exact_dof_arr(qv, n=n)
        Fq = rev_b.assemble_op_grid(dq, op, n, npts=11)
        gm = field_metrics(Fq, qv, 11)
        wl.append(gm["w_star"]); kl.append(gm["k"]); sl.append(gm["s"])
    wl, kl, sl = map(np.asarray, (wl, kl, sl))
    gen = dict(w_star_mean=float(wl.mean()), w_star_std=float(wl.std()),
               w_star_cv=float(wl.std() / wl.mean() * 100),
               k_mean=float(kl.mean()), k_std=float(kl.std()),
               k_cv=float(kl.std() / kl.mean() * 100),
               s_mean=float(sl.mean()), s_std=float(sl.std()),
               s_cv=float(sl.std() / sl.mean() * 100))

    rec = dict(
        seed=int(seed), elapsed_min=round((time.time() - t_all) / 60, 2),
        pinn=dict(l2=e_pinn_l2, max_abs=e_pinn_max, node=l1),
        closed=dict(pinn_op=cl_op, pinn_bfs=cl_bf),
        gen30=gen)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    if not quiet:
        print(f"[{seed}] done {rec['elapsed_min']} min | "
              f"cl op k={np.mean([cl_op['kxx'],cl_op['kyy'],cl_op['kxy']]):.2e} "
              f"w*={cl_op['w_star']:.2e} | gen k={gen['k_mean']:.2e} "
              f"(CV {gen['k_cv']:.0f}%)", flush=True)
    return rec, time.time() - t_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2026")
    ap.add_argument("--ep-pinn", type=int, default=8000)
    ap.add_argument("--ep-op", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=1200)
    ap.add_argument("--no-holdout", action="store_true", default=False,
                    help="默认 False：与主稿一致的留出配置（op 训练排除 q∈[0.9,1.1]）")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--outdir", default=SEED_DIR,
                    help="输出目录（默认 results/seeds）")
    args = ap.parse_args()

    global OUTDIR
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    torch.set_num_threads(args.threads)
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        rec, _ = run_seed(s, args.ep_pinn, args.ep_op, args.lbfgs,
                          args.no_holdout, n=4)
        print(json.dumps({k: rec[k] for k in ("seed", "elapsed_min")}))


if __name__ == "__main__":
    main()
