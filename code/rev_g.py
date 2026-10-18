# -*- coding: utf-8 -*-
"""
rev_g.py — 位置感知 branch 消除中心线 kxy 伪影（#3 最终结构路径）
原理：单元局部坐标无法表达"本单元位于板中心线"；把单元全局中心 (cx,cy)
作为 branch 附加输入（16→18），网络即可学得：处于中心线界面的单元在共享
边界上输出 kxy→0（同时压低界面 jump）。训练数据 = 主 Navier 族（同 qs_train/
make_sym_dataset），pos=(2cx-1,2cy-1)*0.25。4000ep 与主结果同预算。
评估（exact DOF, q=1）：k/s/w_star + x=0.5 中心行 max|kxy| + 界面 jump KXY。
落盘 supp_review.rev_g。
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
import rev_c
from rev_common import (RES, exact_dof_arr, mxy_center_max, interface_jump,
                        load_json, save_json, analytic_metrics)
from eval_poisson import _load_pinn

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Lex = S.A / n
POS_SC = 0.25


def make_pos_dataset(qs, n_pts=10, seed=1):
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
            zeta = rng.choice([-1.0, 1.0], size=XI.size)
            w_all = S.w_analytic(gx, gy, q)
            kx_a, ky_a, kxy_a = S.curv_analytic(gx, gy, q)
            sx_a, sy_a, sxy_a = S.stress_analytic(gx, gy, zeta, q)
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dnode[ids].ravel()
            cx = (ei + 0.5) * Lex; cy = (ej + 0.5) * Lex
            pos = np.array([2 * cx - 1.0, 2 * cy - 1.0]) * POS_SC
            samples.append((de, pos, np.stack([XI, ETA], axis=1), zeta,
                            w_all, np.stack([kx_a, ky_a, kxy_a], axis=1),
                            np.stack([sx_a, sy_a, sxy_a], axis=1), float(q)))
    return samples


def train_pos(op, samples, n_epochs, batch=24, lr=1e-3, log_every=500):
    CORNER_XI = S.CORNER_XI
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    keys = ["L1", "L2", "L3", "L4", "L5", "L6"]
    hist = {k: [] for k in keys}
    for it in range(n_epochs):
        opt.zero_grad()
        idx = np.random.choice(len(samples), batch, replace=False)
        acc = {k: torch.tensor(0.0) for k in keys}
        for i in idx:
            de, pos, pts, zeta, w_ref, k_ref, s_ref, q = samples[i]
            d_t = torch.tensor(np.concatenate([de, pos]),
                               dtype=torch.float64).reshape(1, 18)
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
            mx = S.D_FLEX * (k_hat[:, 0] + S.NU * k_hat[:, 1])
            my = S.D_FLEX * (k_hat[:, 1] + S.NU * k_hat[:, 0])
            mxy = S.D_FLEX * (1.0 - S.NU) * k_hat[:, 2]
            f = 6.0 * sz / S.H_TH ** 2
            acc["L5"] = acc["L5"] + ((s_hat - torch.stack([f * mx, f * my, f * mxy], dim=1)) ** 2).mean()
            g1 = torch.autograd.grad(w_hat.sum(), xy, create_graph=True)[0]
            h1 = torch.autograd.grad(g1[:, 0].sum(), xy, create_graph=True)[0]
            h2 = torch.autograd.grad(g1[:, 1].sum(), xy, create_graph=True)[0]
            c = 4.0 / Lex ** 2
            acc["L6"] = acc["L6"] + ((k_hat - torch.stack(
                [-c * h1[:, 0], -c * h2[:, 1], -c * 0.5 * (h1[:, 1] + h2[:, 0])], dim=1)) ** 2).mean()
        loss = (acc["L1"] + acc["L2"] + acc["L3"] + acc["L4"] + acc["L5"]
                + 1e-2 * acc["L6"]) / batch
        loss.backward()
        opt.step(); sched.step()
        for k in keys:
            hist[k].append((acc[k] / batch).item())
        if (it + 1) % log_every == 0:
            print(f"  [POS {it+1}] " + "  ".join(
                f"{k}={hist[k][-1]:.2e}" for k in keys))
    return hist


def assemble_pos(d_all, op, npts=15):
    """位置感知装配：输入 (n+1)²,4 或 ndof；逐单元 pos 拼接"""
    n1 = npts
    W = np.zeros((n * n1, n * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W); WX = np.zeros_like(W); WY = np.zeros_like(W)
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    d4 = np.asarray(d_all).reshape(-1, 4)
    for ei in range(n):
        for ej in range(n):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d4[ids].ravel()
            cx = (ei + 0.5) * Lex; cy = (ej + 0.5) * Lex
            pos = np.array([2 * cx - 1.0, 2 * cy - 1.0]) * POS_SC
            d_t = torch.tensor(np.concatenate([de, pos]),
                               dtype=torch.float64).reshape(1, 18)
            with torch.no_grad():
                wk = op.wk_at(d_t, torch.tensor(pts)).numpy()
                sp = op.s_at(d_t, torch.stack(
                    [torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1]),
                     torch.ones(pts.shape[0])], dim=1)).numpy()
            sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = wk[:, 0].reshape(n1, n1); KX[sl_i, sl_j] = wk[:, 1].reshape(n1, n1)
            KY[sl_i, sl_j] = wk[:, 2].reshape(n1, n1); KXY[sl_i, sl_j] = wk[:, 3].reshape(n1, n1)
            SX[sl_i, sl_j] = sp[:, 0].reshape(n1, n1); SY[sl_i, sl_j] = sp[:, 1].reshape(n1, n1)
            SXY[sl_i, sl_j] = sp[:, 2].reshape(n1, n1)
            xg = xs[ei] + 0.5 * Lex * (pts[:, 0] + 1)
            yg = ys[ej] + 0.5 * Lex * (pts[:, 1] + 1)
            WX[sl_i, sl_j] = xg.reshape(n1, n1); WY[sl_i, sl_j] = yg.reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    args = ap.parse_args()
    t0 = time.time()
    qs = rev_c.qs_train()
    samples = make_pos_dataset(qs)
    op = S.PlateOperator(p=64, hidden=64, d_in=18)
    train_pos(op, samples, args.epochs)
    op_path = os.path.join(RES, "op_state_pos.pt")
    torch.save(op.state_dict(), op_path)
    op.eval()

    d_ex = exact_dof_arr(1.0)
    F = assemble_pos(d_ex, op, npts=15)
    am = analytic_metrics(F, 1.0, npts=15)
    n1 = 15
    cl = max(float(np.abs(F["KXY"][2 * n1 - 1, :]).max()),
             float(np.abs(F["KXY"][2 * n1, :]).max()))
    jk = float(np.abs(F["KXY"][2 * n1 - 1, :] - F["KXY"][2 * n1, :]).max())
    out = {"epochs": args.epochs, "n_train": len(qs),
           "q1_exact": {"k": am["k"], "s": am["s"], "w_star": am["w_star"]},
           "centreline_max_kxy": float(cl),
           "interface_jump_KXY": float(jk),
           "band5_mxy": mxy_center_max(F)}
    print("q=1 exact:", {k: round(v, 4) for k, v in out["q1_exact"].items()},
          "| centreline max|kxy| =", f"{cl:.2e}", "| jump KXY =", f"{jk:.2e}")
    R = load_json()
    R.setdefault("supp_review", {})["rev_g"] = out
    save_json(R)
    print("saved supp_review.rev_g in", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
