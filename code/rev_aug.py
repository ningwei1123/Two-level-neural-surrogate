# -*- coding: utf-8 -*-
"""
rev_aug.py — 审稿 R2 #5：D4 对称数据增强（第 4 条 Mxy 修复路径）
对均匀载荷简支方板族，把每个 4×4 单元样本按 x/y 反射扩充 4 份：
DOF 取自镜像后角点的解析场（含导数自动变号），单元内参考场直接在
镜像位置解析求值。4000 次迭代与 plain 相同训练预算（样本池 4 倍大）。
评估与 rev_c.plain 同构：q=1 exact/pinn + gen30 + centreline。
落盘 supp_review.rev_c.aug（与 plain 对照）。
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
import rev_c
from rev_common import (RES, load_json, save_json, analytic_metrics,
                        exact_dof_arr)
from eval_poisson import _load_pinn

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Lex = S.A / n


def make_aug_dataset(qs, n_pts=10, seed=1):
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, n_pts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    XI = XI.ravel(); ETA = ETA.ravel()
    rng = np.random.default_rng(seed)
    samples = []
    flips = [(0, 0), (1, 0), (0, 1), (1, 1)]     # (reflect_x, reflect_y)
    for q in qs:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Xr, Yr = X.ravel(), Y.ravel()
        w = S.w_analytic(Xr, Yr, q)
        kx, ky, kxy = S.curv_analytic(Xr, Yr, q)
        wx, wy = S._w_grad(Xr, Yr, q)
        dnode = np.stack([w, wx, wy, -kxy], axis=1)
        for (ei, ej) in [(i, j) for i in range(n) for j in range(n)]:
            for fx, fy in flips:
                i0 = (n - 1 - ei) if fx else ei
                j0 = (n - 1 - ej) if fy else ej
                gx = xs[i0] + 0.5 * Lex * (XI + 1.0)
                gy = ys[j0] + 0.5 * Lex * (ETA + 1.0)
                PX = gx.ravel(); PY = gy.ravel()
                zeta = rng.choice([-1.0, 1.0], size=PX.size)
                ids = [(i0 + di) * (n + 1) + (j0 + dj) for (di, dj) in
                       [(0, 0), (1, 0), (1, 1), (0, 1)]]
                de = dnode[ids].ravel()
                w_all = S.w_analytic(PX, PY, q)
                kx_a, ky_a, kxy_a = S.curv_analytic(PX, PY, q)
                sx_a, sy_a, sxy_a = S.stress_analytic(PX, PY, zeta, q)
                samples.append((de, np.stack([XI, ETA], axis=1), zeta,
                                w_all,
                                np.stack([kx_a, ky_a, kxy_a], axis=1),
                                np.stack([sx_a, sy_a, sxy_a], axis=1),
                                float(q)))
    return samples


def centreline_max_kxy(F, npts=15):
    KXY = F["KXY"]
    r1, r2 = 2 * npts - 1, 2 * npts
    return max(float(np.abs(KXY[r1, :]).max()),
               float(np.abs(KXY[r2, :]).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    args = ap.parse_args()
    t0 = time.time()
    qs = rev_c.qs_train()
    samples = make_aug_dataset(qs)
    print(f"[R2-aug] 增强样本 {len(samples)} (≈plain 的 4 倍池)，{args.epochs} iter")
    op = S.PlateOperator(p=64, hidden=64)
    hist = rev_c.train_op_sym(op, samples, n_epochs=args.epochs,
                              lam_sym=0.0, lam6=1e-2, lam_w=1.0)
    torch.save(op.state_dict(), os.path.join(RES, "op_state_aug.pt"))
    op.eval()

    # 评估与 plain 同构
    ev = {"epochs": args.epochs, "n_train": len(qs), "n_samples": len(samples)}
    d_ex = exact_dof_arr(1.0)
    import eval_symavg
    from rev_common import mxy_center_max
    pinn = _load_pinn()
    xs = np.linspace(0, 1, n + 1); ys = xs
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    dp = np.array(S.pinn_field(pinn, np.stack([Xn.ravel(), Yn.ravel()],
                                              axis=1), 1.0)).T
    for iname, dd in [("exact", d_ex), ("pinn", dp)]:
        F = eval_symavg._assemble(dd, op, scale=None, npts=15)
        am = analytic_metrics(F, 1.0, npts=15)
        ev[iname] = {"w_star": am["w_star"], "k": am["k"], "s": am["s"],
                     "mxy_center": centreline_max_kxy(F, 15),
                     "mxy_band": mxy_center_max(F)}
    ev["gen30_exact"] = rev_c.gen30(op, dof="exact", npts=11)
    ev["elapsed_s"] = round(time.time() - t0, 1)
    ev["note"] = ("D4 reflection data augmentation (x/y mirrors, 4x sample pool), "
                  "same architecture/seeds/iterations as plain control.")
    R = load_json()
    R.setdefault("supp_review", {}).setdefault("rev_c", {})["aug"] = ev
    save_json(R)
    print(f"\n[R2-aug saved] q1 exact w*={ev['exact']['w_star']:.2e} "
          f"k={ev['exact']['k']:.2e} s={ev['exact']['s']:.2e} "
          f"mxy_cl={ev['exact']['mxy_center']:.1e}")
    print(f"  gen30 k={ev['gen30_exact']['k']['mean']:.2e}  "
          f"elapsed {ev['elapsed_s']}s")


if __name__ == "__main__":
    main()
