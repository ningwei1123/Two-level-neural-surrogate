# -*- coding: utf-8 -*-
"""
rev_c.py — P1-C：λ_sym 对称惩罚重训 + λ 敏感性（回应评审 #2 / mandatory ③ 与 #9）
mode:
  sym          : 在级Ⅱ损失加 λ_sym·mean(σ_xy² | 中心线附近)，抑制 Mxy 伪影
  lw-lo/lw-hi  : 位移监督权重 λ_1 ∈ {0.5, 1.5}（默认 1.0）
  l6-lo/l6-hi  : 曲率-位移一致性权重 λ_6 ∈ {5e-3, 1.5e-2}（默认 1e-2）
每个 mode 训练后：q=1 闭环（exact/PINN DOF）指标 + Mxy 中心线伪影 + 界面最大跳变，
另跑 30 未见载荷（exact DOF）泛化均值，全部写入 results.json supp_review.rev_c[mode]。
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
import eval_symavg
from rev_common import (RES, exact_dof_arr, assemble_bfs, analytic_metrics,
                        mxy_center_max, interface_jump, load_json, save_json)
from eval_poisson import _load_pinn

n = S.N_MESH
Lex = S.A / n


def make_sym_dataset(qs, n_pts=10, seed=1):
    """make_operator_dataset 同构 + 每样本附带中心线掩码 (n_pts²,)"""
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
            mask = (np.abs(gx - 0.5) < 1e-9) | (np.abs(gy - 0.5) < 1e-9)
            PX = gx.ravel(); PY = gy.ravel()
            zeta = rng.choice([-1.0, 1.0], size=PX.size)
            w_all = S.w_analytic(PX, PY, q)
            kx_a, ky_a, kxy_a = S.curv_analytic(PX, PY, q)
            sx_a, sy_a, sxy_a = S.stress_analytic(PX, PY, zeta, q)
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dnode[ids].ravel()
            samples.append((de, np.stack([XI, ETA], axis=1), zeta,
                            w_all, np.stack([kx_a, ky_a, kxy_a], axis=1),
                            np.stack([sx_a, sy_a, sxy_a], axis=1), float(q), mask))
    return samples


def train_op_sym(op, samples, n_epochs, lam_sym=1e-2, lam6=1e-2, lam_w=1.0,
                 batch=24, lr=1e-3, log_every=500):
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
            if len(s_i) == 8:
                de, pts, zeta, w_ref, k_ref, s_ref, q, mask = s_i
            else:
                de, pts, zeta, w_ref, k_ref, s_ref, q = s_i
                mask = None
            d_t = torch.tensor(de, dtype=torch.float64).reshape(1, 16)
            xy = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
            wk = op.wk_at(d_t, xy)
            w_hat = wk[:, 0]; k_hat = wk[:, 1:4]
            sz = torch.tensor(zeta, dtype=torch.float64)
            s_hat = op.s_at(d_t, torch.stack([xy[:, 0], xy[:, 1], sz], dim=1))
            w_r = torch.tensor(w_ref); k_r = torch.tensor(k_ref); s_r = torch.tensor(s_ref)
            acc["L1"] = acc["L1"] + ((w_hat - w_r) ** 2).mean()
            acc["L2"] = acc["L2"] + ((k_hat - k_r) ** 2).mean()
            xc = CORNER_XI.clone().requires_grad_(True)
            wc = op.wk_at(d_t, xc)[:, 0]
            gc = torch.autograd.grad(wc.sum(), xc, create_graph=True)[0]
            hx = torch.autograd.grad(gc[:, 0].sum(), xc, create_graph=True)[0]
            dnode = torch.tensor(de, dtype=torch.float64).reshape(4, 4)
            acc["L3"] = acc["L3"] + ((wc - dnode[:, 0]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Lex) * gc[:, 0] - dnode[:, 1]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Lex) * gc[:, 1] - dnode[:, 2]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((4.0 / Lex ** 2) * hx[:, 1] - dnode[:, 3]) ** 2).mean()
            acc["L4"] = acc["L4"] + ((s_hat - s_r) ** 2).mean()
            zz = sz
            mx = S.D_FLEX * (k_hat[:, 0] + S.NU * k_hat[:, 1])
            my = S.D_FLEX * (k_hat[:, 1] + S.NU * k_hat[:, 0])
            mxy = S.D_FLEX * (1.0 - S.NU) * k_hat[:, 2]
            f = 6.0 * zz / S.H_TH ** 2
            s_cons = torch.stack([f * mx, f * my, f * mxy], dim=1)
            acc["L5"] = acc["L5"] + ((s_hat - s_cons) ** 2).mean()
            g1 = torch.autograd.grad(w_hat.sum(), xy, create_graph=True)[0]
            h1 = torch.autograd.grad(g1[:, 0].sum(), xy, create_graph=True)[0]
            h2 = torch.autograd.grad(g1[:, 1].sum(), xy, create_graph=True)[0]
            c = 4.0 / Lex ** 2
            k_from_w = torch.stack([-c * h1[:, 0], -c * h2[:, 1],
                                    -c * 0.5 * (h1[:, 1] + h2[:, 0])], dim=1)
            acc["L6"] = acc["L6"] + ((k_hat - k_from_w) ** 2).mean()
            m = torch.tensor(mask, dtype=torch.bool) if mask is not None else None
            if m is not None and m.any():
                acc["Lsym"] = acc["Lsym"] + (s_hat[:, 2][m] ** 2).mean()
        loss = (lam_w * acc["L1"] + acc["L2"] + acc["L3"] + acc["L4"] + acc["L5"]
                + lam6 * acc["L6"] + lam_sym * acc["Lsym"]) / batch
        loss.backward()
        opt.step(); sched.step()
        for k in keys:
            history[k].append((acc[k] / batch).item())
        if (it + 1) % log_every == 0:
            print(f"  [OP {it+1}] " + "  ".join(
                f"{k}={history[k][-1]:.2e}" for k in keys))
    return history


def qs_train():
    qs_all = np.random.default_rng(3).uniform(S.Q_LO, S.Q_HI, 400)
    return qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:120]


def gen30(op, dof="exact", npts=11):
    rng = np.random.default_rng(11)
    qs = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    agg = {"w_star": [], "k": [], "s": []}
    for qv in qs:
        if dof == "exact":
            d = exact_dof_arr(qv)
        else:
            raise ValueError(dof)
        F = eval_symavg._assemble(d, op, scale=None, npts=npts)
        am = analytic_metrics(F, qv, npts=npts)
        agg["w_star"].append(am["w_star"]); agg["k"].append(am["k"]); agg["s"].append(am["s"])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in agg.items()}


def eval_full(op, tag):
    pinn = _load_pinn()
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    dp = np.array(S.pinn_field(pinn, np.stack([Xn.ravel(), Yn.ravel()], axis=1), 1.0)).T
    d_ex = exact_dof_arr(1.0)
    out = {}
    for iname, dd in [("exact", d_ex), ("pinn", dp)]:
        F = eval_symavg._assemble(dd, op, scale=None, npts=15)
        am = analytic_metrics(F, 1.0, npts=15)
        out[iname] = {**am, "mxy_center": mxy_center_max(F),
                      "jump": interface_jump(F, npts=15)}
    out["gen30_exact"] = gen30(op, dof="exact")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["sym", "plain", "lw-lo", "lw-hi", "l6-lo", "l6-hi"])
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lam-sym", type=float, default=1e-2)
    args = ap.parse_args()

    lam_w, lam6, lam_sym, use_sym = 1.0, 1e-2, 0.0, False
    if args.mode == "sym":
        lam_sym, use_sym = args.lam_sym, True
    elif args.mode == "lw-lo":
        lam_w = 0.5
    elif args.mode == "lw-hi":
        lam_w = 1.5
    elif args.mode == "l6-lo":
        lam6 = 5e-3
    elif args.mode == "l6-hi":
        lam6 = 1.5e-2

    t0 = time.time()
    qs = qs_train()
    if use_sym:
        samples = make_sym_dataset(qs)
    else:
        samples = S.make_operator_dataset(qs, n=n, n_pts=10, seed=1)
    print(f"[P1-C {args.mode}] 样本 {len(samples)}，训练 {args.epochs} ep"
          f" (lam_w={lam_w}, lam6={lam6}, lam_sym={lam_sym})")
    op = S.PlateOperator(p=64, hidden=64)
    hist = train_op_sym(op, samples, n_epochs=args.epochs,
                        lam_sym=lam_sym, lam6=lam6, lam_w=lam_w)
    tag = args.mode.replace("-", "")
    torch.save(op.state_dict(), os.path.join(RES, f"op_state_{tag}.pt"))
    ev = eval_full(op, args.mode)
    ev["elapsed_s"] = round(time.time() - t0, 1)
    ev["params"] = {"lam_w": lam_w, "lam6": lam6, "lam_sym": lam_sym,
                    "epochs": args.epochs}
    R = load_json()
    R.setdefault("supp_review", {}).setdefault("rev_c", {})[args.mode] = ev
    save_json(R)
    print(f"\n[P1-C {args.mode}] saved.")
    print(f"  q=1 exact : w*={ev['exact']['w_star']:.2e} κ={ev['exact']['k']:.2e} "
          f"σ={ev['exact']['s']:.2e} | Mxy_center={ev['exact']['mxy_center']:.1e}")
    print(f"  q=1 pinn  : w*={ev['pinn']['w_star']:.2e} κ={ev['pinn']['k']:.2e} "
          f"σ={ev['pinn']['s']:.2e} | Mxy_center={ev['pinn']['mxy_center']:.1e}")
    print(f"  gen30      : w*={ev['gen30_exact']['w_star']['mean']:.2e} "
          f"κ={ev['gen30_exact']['k']['mean']:.2e} σ={ev['gen30_exact']['s']['mean']:.2e}")


if __name__ == "__main__":
    main()
