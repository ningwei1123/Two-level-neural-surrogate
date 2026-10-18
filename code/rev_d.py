# -*- coding: utf-8 -*-
"""
rev_d.py — P1-B 基线对比（回应评审 #4 / mandatory ④）
两个"载荷→全场"基准（输入仅 q，均匀载荷线性族）：
  A) POD-ROM：训练载荷快照 SVD，1/2/4 模态重构测试载荷
  B) vanilla DeepONet：branch(q) · trunk(x,y) → w 全场
预期（与评审 #6 自洽）：均匀载荷线性族是 1 参数线性流形 → 载荷→场基准"平凡精确"；
这正说明需要 FE 标签/非均匀载荷才能构成有区分度的基准——写进回复信。
落盘 supp_review.baselines
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import shell_system as S
from rev_common import RES, l2rel, load_json, save_json

torch.set_num_threads(8)
M = 41                       # 均匀评估网格（与 w* 解码唯一网格一致）
Xg, Yg = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M), indexing="ij")
XR, YR = Xg.ravel(), Yg.ravel()


def state_vec(q):
    """q → 全场状态向量：w; κx,κy,κxy; σx,σy,σxy(顶面) 全 41² 点"""
    w = S.w_analytic(XR, YR, q)
    kx, ky, kxy = S.curv_analytic(XR, YR, q)
    sx, sy, sxy = S.stress_analytic(XR, YR, 1.0, q)
    return np.concatenate([w, kx, ky, kxy, sx, sy, sxy])


def rel_err(a, b):
    return l2rel(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=1500)
    args = ap.parse_args()
    qs_all = np.random.default_rng(3).uniform(S.Q_LO, S.Q_HI, 400)
    qs_tr = qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:120]
    rng = np.random.default_rng(5)
    qs_te = np.concatenate([rng.uniform(0.5, 0.9, 5), rng.uniform(1.1, 2.0, 5)])

    # ---------- A) POD-ROM ----------
    Snap = np.stack([state_vec(q) for q in qs_tr], axis=1)   # (7M², ntr)
    Snapm = Snap.mean(axis=1, keepdims=True)
    Ac = Snap - Snapm
    U, Sv, _ = np.linalg.svd(Ac, full_matrices=False)        # (7M²,ntr)
    pod = {"singular_values": [float(x) for x in Sv[:8]],
           "test": {}}
    for km in (1, 2, 4):
        errs = []
        for qv in qs_te:
            v = state_vec(qv)
            alpha = U[:, :km].T @ (v - Snapm[:, 0])
            rec = Snapm[:, 0] + U[:, :km] @ alpha
            errs.append(rel_err(rec, v))
        pod["test"][f"k{km}"] = {"mean": float(np.mean(errs))}
        print(f"[POD k={km}] test mean rel L2 = {np.mean(errs):.2e}")
    # 真实载荷需 2 个快照→仅 1 维：直接回归系数测试
    A1 = np.stack([state_vec(q) for q in qs_tr], axis=1)     # 不中心化，线性族 rank1
    B = qs_tr.reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(B, A1.T, rcond=None)          # (1, 7M²)
    lr_errs = [rel_err(state_vec(qv), coef[0] * qv) for qv in qs_te]
    pod["linear_regression"] = {"mean": float(np.mean(lr_errs))}
    print(f"[POD 线性回归] test mean rel L2 = {np.mean(lr_errs):.2e}")

    # ---------- B) 消融：监督-only 算子（去掉物理一致性 ③⑤⑥；保留 ①②④） ----------
    # 与主实验同数据集/架构/权重/种子，仅损失结构不同 → 量化 six-term loss 的价值
    torch.manual_seed(2026)
    np.random.seed(2026)

    def train_supervised(op, samples, n_epochs=args.ep, batch=24, lr=1e-3):
        opt = torch.optim.Adam(op.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        for it in range(n_epochs):
            opt.zero_grad()
            idx = np.random.choice(len(samples), batch, replace=False)
            loss = torch.tensor(0.0, dtype=torch.float64)
            for i in idx:
                de, pts, zeta, w_ref, k_ref, s_ref, q = samples[i]
                d_t = torch.tensor(de, dtype=torch.float64).reshape(1, 16)
                xy = torch.tensor(pts, dtype=torch.float64)
                wk = op.wk_at(d_t, xy)
                sz = torch.tensor(zeta, dtype=torch.float64)
                s_hat = op.s_at(d_t, torch.stack([xy[:, 0], xy[:, 1], sz], dim=1))
                loss = loss + ((wk[:, 0] - torch.tensor(w_ref, dtype=torch.float64)) ** 2).mean()
                loss = loss + ((wk[:, 1:4] - torch.tensor(k_ref, dtype=torch.float64)) ** 2).mean()
                loss = loss + ((s_hat - torch.tensor(s_ref, dtype=torch.float64)) ** 2).mean()
            loss = loss / batch
            loss.backward(); opt.step(); sched.step()
            if (it + 1) % 400 == 0:
                print(f"  [SupOnly {it+1}] loss {loss.item():.3e}")
        return loss

    import shell_system as S2
    qs_tr_full = qs_tr
    sam = S2.make_operator_dataset(qs_tr_full, n=S2.N_MESH, n_pts=10, seed=1)
    op_sup = S2.PlateOperator(p=64, hidden=64)
    torch.manual_seed(2026); np.random.seed(2026)
    train_supervised(op_sup, sam, n_epochs=args.ep)
    torch.save(op_sup.state_dict(), os.path.join(RES, "op_state_suponly.pt"))
    # 评估（q=1 exact/PINN DOF + gen30 exact）
    from eval_symavg import _assemble
    from rev_common import exact_dof_arr, analytic_metrics
    import eval_poisson as EP
    pinn = EP._load_pinn()
    xs = np.linspace(0, 1, S2.N_MESH + 1); ys = xs
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    dp = np.array(S2.pinn_field(pinn, np.stack([Xn.ravel(), Yn.ravel()], axis=1), 1.0)).T
    de = exact_dof_arr(1.0)
    res = {}
    for iname, dd in [("exact", de), ("pinn", dp)]:
        F = _assemble(dd, op_sup, scale=None, npts=15)
        res[iname] = analytic_metrics(F, 1.0, npts=15)
    rng2 = np.random.default_rng(11)
    qs30 = np.concatenate([rng2.uniform(0.5, 0.9, 15), rng2.uniform(1.1, 2.0, 15)])
    agg = {"w_star": [], "k": [], "s": []}
    for qv in qs30:
        dq = exact_dof_arr(qv)
        Fq = _assemble(dq, op_sup, scale=None, npts=11)
        m = analytic_metrics(Fq, qv, npts=11)
        agg["w_star"].append(m["w_star"]); agg["k"].append(m["k"]); agg["s"].append(m["s"])
    gen = {k: float(np.mean(v)) for k, v in agg.items()}
    ab = {"q1": res, "gen30_exact_mean": gen, "epochs": args.ep}
    print(f"[SupOnly] q1 exact w*={res['exact']['w_star']:.2e} κ={res['exact']['k']:.2e} "
          f"σ={res['exact']['s']:.2e} | gen30 κ={gen['k']:.2e}")

    R = load_json()
    R.setdefault("supp_review", {})["baselines"] = {
        "note": ("(A) Load-to-field POD on the uniform-load linear family: because the family is a "
                 "one-parameter linear manifold (w ∝ q), the load-parametrised surrogate is exactly rank "
                 "one and trivially accurate — the meaningful benchmark is the DOF-to-field recovery "
                 "task used in the paper (vs BFS post-processing on identical inputs). "
                 "(B) Ablation of the six-term loss: supervised-only operator (terms 1,2,4, no "
                 "physical-consistency terms 3,5,6) trained on the same data/architecture/seeds."),
        "grid_M": M, "pod": pod,
        "suponly": ab,
        "loss_sixterm_q1": None,   # 由主结果填（见下）
    }
    save_json(R)
    print("\n[P1-D] saved supp_review.baselines")


if __name__ == "__main__":
    main()
