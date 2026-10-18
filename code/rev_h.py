# -*- coding: utf-8 -*-
"""
rev_h.py — 实验 C：强化对称/界面约束重训（回应评审 #2：Mxy 中心线伪影）
rev_c 'sym' 已证：λ_sym 惩罚不劣化（同预算 gen30 κ 0.068 vs plain 0.074）
但 λ=1e-2 强度不足（mxy_center 仅 -8.5%）。本脚本：
  sym2 : λ_sym=1e-1 约束带加宽到中心线两侧 ±0.03 物理坐标（界面带 2 行），
        同时压 σ̂xy 与 κ̂xy（直接压 mxy 曲率，经 L5 间接压应力）
  plain4 : 六项损失 4000ep 对照（与主结果同预算的公平基线）
训练后：q=1 闭环 exact/PINN DOF（k/s/w*）+ mxy_center + interface jump
       + gen30（exact DOF）。落盘 supp_review.rev_h[mode]。
"""
import os
import json
import numpy as np
import torch
import shell_system as S
import rev_c
from rev_common import RES, load_json, save_json

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Lex = S.A / n


def make_sym2_dataset(qs, n_pts=10, seed=1, band=0.035):
    """make_operator_dataset 同构 + 界面带掩码（中心线 x=0.5,y=0.5 两侧 ±band）"""
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, n_pts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    XI = XI.ravel(); ETA = ETA.ravel()
    rng = np.random.default_rng(seed)
    npts = XI.size
    samples = []
    for q in qs:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Xr, Yr = X.ravel(), Y.ravel()
        w = S.w_analytic(Xr, Yr, q)
        kx, ky, kxy = S.curv_analytic(Xr, Yr, q)
        wx, wy = S._w_grad(Xr, Yr, q)
        dnode = np.stack([w, wx, wy, -kxy], axis=1)
        for (ei, ej) in [(i, j) for i in range(n) for j in range(n)]:
            gx = xs[ei] + 0.5 * Lex * (XI + 1.0)
            gy = ys[ej] + 0.5 * Lex * (ETA + 1.0)
            mask = (np.abs(gx - 0.5) <= band) | (np.abs(gy - 0.5) <= band)
            zeta = rng.choice([-1.0, 1.0], size=XI.size)
            w_all = S.w_analytic(gx, gy, q)
            kx_a, ky_a, kxy_a = S.curv_analytic(gx, gy, q)
            sx_a, sy_a, sxy_a = S.stress_analytic(gx, gy, zeta, q)
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dnode[ids].ravel()
            samples.append((de, np.stack([XI, ETA], axis=1), zeta,
                            w_all, np.stack([kx_a, ky_a, kxy_a], axis=1),
                            np.stack([sx_a, sy_a, sxy_a], axis=1), float(q), mask))
    return samples


def train_op_sym2(op, samples, n_epochs, lam_sym=1e-1, batch=24, lr=1e-3,
                  log_every=500):
    CORNER_XI = S.CORNER_XI
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    keys = ["L1", "L2", "L3", "L4", "L5", "L6", "Lsym"]
    history = {k: [] for k in keys}
    for it in range(n_epochs):
        opt.zero_grad()
        idx = np.random.choice(len(samples), batch, replace=False)
        acc = {k: torch.tensor(0.0) for k in keys}
        for i in idx:
            s_i = samples[i]
            de, pts, zeta, w_ref, k_ref, s_ref, q, mask = s_i
            d_t = torch.tensor(de, dtype=torch.float64).reshape(1, 16)
            xy = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
            wk = op.wk_at(d_t, xy)
            w_hat = wk[:, 0]; k_hat = wk[:, 1:4]
            sz = torch.tensor(zeta, dtype=torch.float64)
            s_hat = op.s_at(d_t, torch.stack([xy[:, 0], xy[:, 1], sz], dim=1))
            acc["L1"] = acc["L1"] + ((w_hat - torch.tensor(w_ref)) ** 2).mean()
            acc["L2"] = acc["L2"] + ((k_hat - torch.tensor(k_ref)) ** 2).mean()
            xc = CORNER_XI.clone().requires_grad_(True)
            wc = op.wk_at(d_t, xc)[:, 0]
            gc = torch.autograd.grad(wc.sum(), xc, create_graph=True)[0]
            hx = torch.autograd.grad(gc[:, 0].sum(), xc, create_graph=True)[0]
            dnode = torch.tensor(de, dtype=torch.float64).reshape(4, 4)
            acc["L3"] = acc["L3"] + ((wc - dnode[:, 0]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Lex) * gc[:, 0] - dnode[:, 1]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Lex) * gc[:, 1] - dnode[:, 2]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((4.0 / Lex ** 2) * hx[:, 1] - dnode[:, 3]) ** 2).mean()
            acc["L4"] = acc["L4"] + ((s_hat - torch.tensor(s_ref)) ** 2).mean()
            zz = sz
            mx = S.D_FLEX * (k_hat[:, 0] + S.NU * k_hat[:, 1])
            my = S.D_FLEX * (k_hat[:, 1] + S.NU * k_hat[:, 0])
            mxy = S.D_FLEX * (1.0 - S.NU) * k_hat[:, 2]
            f = 6.0 * zz / S.H_TH ** 2
            acc["L5"] = acc["L5"] + ((s_hat - torch.stack([f * mx, f * my, f * mxy], dim=1)) ** 2).mean()
            g1 = torch.autograd.grad(w_hat.sum(), xy, create_graph=True)[0]
            h1 = torch.autograd.grad(g1[:, 0].sum(), xy, create_graph=True)[0]
            h2 = torch.autograd.grad(g1[:, 1].sum(), xy, create_graph=True)[0]
            c = 4.0 / Lex ** 2
            k_from_w = torch.stack([-c * h1[:, 0], -c * h2[:, 1],
                                    -c * 0.5 * (h1[:, 1] + h2[:, 0])], dim=1)
            acc["L6"] = acc["L6"] + ((k_hat - k_from_w) ** 2).mean()
            m = torch.tensor(mask, dtype=torch.bool)
            if m.any():
                # 同时压中心带内 σ̂xy 与 κ̂xy（界面连续性 + 反对称零）
                acc["Lsym"] = acc["Lsym"] + (s_hat[:, 2][m] ** 2).mean()
                acc["Lsym"] = acc["Lsym"] + (k_hat[:, 2][m] ** 2).mean()
        loss = (acc["L1"] + acc["L2"] + acc["L3"] + acc["L4"] + acc["L5"]
                + 1e-2 * acc["L6"] + lam_sym * acc["Lsym"]) / batch
        loss.backward()
        opt.step(); sched.step()
        for k in keys:
            history[k].append((acc[k] / batch).item())
        if (it + 1) % log_every == 0:
            print(f"  [OP {it+1}] " + "  ".join(
                f"{k}={history[k][-1]:.2e}" for k in keys))
    return history


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sym2", "plain4"], required=True)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=1e-1)
    args = ap.parse_args()

    qs = rev_c.qs_train()
    if args.mode == "sym2":
        samples = make_sym2_dataset(qs)
        op = S.PlateOperator(p=64, hidden=64)
        train_op_sym2(op, samples, args.epochs, lam_sym=args.lam)
        op_path = os.path.join(RES, "op_state_sym2.pt")
    else:
        samples = rev_c.make_sym_dataset(qs)   # 同 rev_c，无掩码加权也可训练
        op = S.PlateOperator(p=64, hidden=64)
        # plain4 用普通六项（复用 rev_c 训练器但关 Lsym 权重=0）
        rev_c.train_op_sym(op, samples, args.epochs, lam_sym=0.0)
        op_path = os.path.join(RES, "op_state_plain4.pt")
    torch.save(op.state_dict(), op_path)
    ev = rev_c.eval_full(op, args.mode)
    print("q=1 exact:", {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in ev["exact"].items()})
    print("gen30 exact mean k:", ev["gen30_exact"]["k"].get("mean"))
    R = load_json()
    R.setdefault("supp_review", {}).setdefault("rev_h", {})[args.mode] = {
        **ev, "epochs": args.epochs, "lam_sym": args.lam}
    save_json(R)
    print("saved supp_review.rev_h.", args.mode)


if __name__ == "__main__":
    main()
