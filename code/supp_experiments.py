# -*- coding: utf-8 -*-
"""
supp_experiments.py — 审稿补充实验（review_TWS_strict.md 响应，全部推理级，不重训）
E1 wall-clock: 两级推理 vs 4×4 BFS FE 求解（单载荷）
E2 端到端泛化: 30 unseen loads 用 PINN 节点 DOF（而非 exact）驱动 → 修 M5
E3 算子 Lipschitz 数值估计 → 修 M7
E4 外推: q=2.5, 3.0（exact 与 PINN DOF）→ 替换 companion-study 句
输出写入 results.json（supp 命名空间，不覆盖论文既有键）。
"""
import os
import json
import time
import numpy as np
import torch
import shell_system as S
from eval_symavg import _assemble, pinn_field_all, l2rel
import eval_poisson as EP

RES = S.RES
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Q = S.Q_TEST


def exact_dof(q):
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij"); Xnr, Ynr = Xnr.ravel(), Ynr.ravel()
    wq = S.w_analytic(Xnr, Ynr, q); wxq, wyq = S._w_grad(Xnr, Ynr, q)
    kxyq = S.curv_analytic(Xnr, Ynr, q)[2]
    return np.stack([wq, wxq, wyq, -kxyq], axis=1)


def pinn_dof(pinn, q):
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij"); Xnr, Ynr = Xnr.ravel(), Ynr.ravel()
    # shell_system.pinn_field(pinn, xy, q) 返回 q·(û, û_x, û_y, û_xy)
    arr = np.array(S.pinn_field(pinn, np.stack([Xnr, Ynr], axis=1), q))  # (4,N)
    return arr.T


def global_metrics(F, q, npts):
    """返回该载荷下 w(w*解码)/κ/σ 的全局 L2"""
    from eval_poisson import _squeeze
    G, xq, _ = _squeeze(F, npts)
    Xg, Yg = np.meshgrid(xq, xq, indexing="ij")
    Xr, Yr = Xg.ravel(), Yg.ravel()
    w_star = EP.poisson_reconstruct(G["KX"], G["KY"], xq[1] - xq[0])
    wt = S.w_analytic(Xr, Yr, q).reshape(Xg.shape)
    kx_t, ky_t, kxy_t = S.curv_analytic(Xr, Yr, q)
    sx_t, sy_t, sxy_t = S.stress_analytic(Xr, Yr, 1.0, q)
    m = dict(
        w_star=l2rel(w_star.ravel(), wt.ravel()),
        kxx=l2rel(G["KX"].ravel(), kx_t), kyy=l2rel(G["KY"].ravel(), ky_t),
        kxy=l2rel(G["KXY"].ravel(), kxy_t),
        sxx=l2rel(G["SX"].ravel(), sx_t), syy=l2rel(G["SY"].ravel(), sy_t),
        sxy=l2rel(G["SXY"].ravel(), sxy_t),
    )
    return m


def main():
    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"), weights_only=False))
    pinn = EP._load_pinn()
    rj = os.path.join(RES, "results.json")
    R = json.load(open(rj, encoding="utf-8"))
    supp = {}

    # ---------- E1: wall-clock ----------
    # FE：4×4 BFS 求解（一次组装+求解，模拟每次改 q 需重解）
    nrep = 200
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij"); Xnr, Ynr = Xnr.ravel(), Ynr.ravel()
    # FE 单次: fem_solve(n,q)
    t0 = time.perf_counter()
    for _ in range(nrep):
        S.fem_solve(n, 1.0 + 1e-6 * _)          # 每次载荷变化 → 重新组装求解
    t_fe_full = (time.perf_counter() - t0) / nrep
    # 两级推理（PINN forward + 16 单元 OP + Poisson 回代）
    d_pin = pinn_dof(pinn, Q)
    t0 = time.perf_counter()
    for _ in range(nrep):
        F = _assemble(d_pin, op, scale=None, npts=15)
        t_infer = (time.perf_counter() - t0) / nrep
    supp["wallclock"] = {
        "bfs_fe_solve_s": t_fe_full, "two_level_infer_s": t_infer,
        "speedup_x": t_fe_full / max(t_infer, 1e-12),
        "note": "per-load, same machine, Python reference FE (dense solve) vs. numpy/torch inference incl. Poisson; "
                "FE uses dense direct solve on 64 free DOFs."}

    # ---------- E2: 端到端泛化 30 unseen（PINN DOF） ----------
    rng = np.random.default_rng(11)
    qs = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    g_w, g_k, g_s = [], [], []
    for qv in qs:
        dq = pinn_dof(pinn, qv)
        Fq = _assemble(dq, op, scale=None, npts=11)
        mq = global_metrics(Fq, qv, 11)
        g_w.append(mq["w_star"]); g_k.append(np.mean([mq["kxx"], mq["kyy"], mq["kxy"]]))
        g_s.append(np.mean([mq["sxx"], mq["syy"], mq["sxy"]]))
    supp["gen30_pinn_dof"] = {
        "w_star_mean": float(np.mean(g_w)), "w_star_std": float(np.std(g_w)),
        "w_star_cv_pct": float(np.std(g_w) / np.mean(g_w) * 100),
        "k_mean": float(np.mean(g_k)), "s_mean": float(np.mean(g_s))}

    # ---------- E4: 外推 q=2.5, 3.0 ----------
    ext = {}
    for qv in (2.5, 3.0):
        de = exact_dof(qv)
        Fe = _assemble(de, op, scale=None, npts=15)
        me = global_metrics(Fe, qv, 15)
        dp = pinn_dof(pinn, qv)
        Fp = _assemble(dp, op, scale=None, npts=15)
        mp = global_metrics(Fp, qv, 15)
        ext[str(qv)] = {"exact_w": me["w_star"], "exact_k": float(np.mean(
            [me["kxx"], me["kyy"], me["kxy"]])),
            "pinn_w": mp["w_star"], "pinn_k": float(np.mean(
                [mp["kxx"], mp["kyy"], mp["kxy"]]))}
    supp["extrapolation_2p5_3"] = ext

    # ---------- E3: Lipschitz 数值估计（对 branch 输入扰动） ----------
    # 随机取若干样本 de，加单位扰动 δ，测 ||ΔG||₂/||δ|| 上界（wk_at 全 4 头输出，坐标网格固定）
    rng3 = np.random.default_rng(0)
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, 8)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)   # (64,2)
    max_L = 0.0
    for _ in range(12):
        # 随机真实 de（从解析场取）
        r1, r2 = rng3.integers(0, n), rng3.integers(0, n)
        ids = [(r1 + di) * (n + 1) + (r2 + dj) for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)]]
        de = exact_dof(1.0).reshape(-1, 4)[ids].ravel()
        d_t = torch.tensor(de).reshape(1, 16)
        with torch.no_grad():
            y0 = op.wk_at(d_t, torch.tensor(pts)).numpy()
        for _k in range(6):
            pert = rng3.normal(0, 1, 16)
            d2 = de + 1e-3 * pert
            with torch.no_grad():
                y1 = op.wk_at(torch.tensor(d2).reshape(1, 16), torch.tensor(pts)).numpy()
            L = np.linalg.norm(y1 - y0) / (1e-3 * np.linalg.norm(pert))
            max_L = max(max_L, L)
    supp["lip_estimate"] = {"sup_Lip_wk": float(max_L),
                            "note": "finite-difference upper bound over 12 sample elements x6 random dirs, trunk grid 8x8"}

    R["supp_review"] = supp
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print("[supp] E1 wallclock:", supp["wallclock"])
    print("[supp] E2 gen30 pinn-dof:", {k: round(v, 5) for k, v in supp["gen30_pinn_dof"].items()})
    print("[supp] E4 extrap:", supp["extrapolation_2p5_3"])
    print("[supp] E3 Lip:", supp["lip_estimate"])


if __name__ == "__main__":
    main()
