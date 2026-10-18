# -*- coding: utf-8 -*-
"""
rev_e2.py — 跨工况族迁移：在 clamped-uniform / ss-patch 载荷族上用 FE 标签重训级Ⅱ
（回应评审 Q3 / #1：级Ⅱ 为数据驱动恢复器，迁移=在该族重训，不改架构；
 同时证明 FE 标签管线对"无解析解载荷"是必需且充分的。）
对照链（每族）：
  OOD（rev_e/rev_f 已存：简支均匀训练的算子直接喂 OOD DOF → 退化）
  BFS 同 DOF（经典插值，分布无关、稳健但 κ 精度受粗网格/表示限制）
  族内重训 op（FE 标签：4×4 DOF in → 16×16 FE 场 label out）→ 应回到 ~5% 量级
验证载荷：
  clamp: q∈留出谱（含 [0.9,1.1]）
  patch: q∈留出谱 + 位置留出（patch 中心偏移 0.125）检查位置泛化边界
落盘 supp_review.transfer_study[mode]
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import shell_system as S
import eval_symavg
import fe_extra as FX
from rev_common import (RES, l2rel, fe_field_grid, mxy_center_max,
                        assemble_bfs, load_json, save_json)

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
NF, NC = 16, S.N_MESH


def make_fe_dataset2(qs, load, bc, nf=NF, nc=NC, n_pts=10, seed=1):
    n = nc
    Lex = S.A / n
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
        d_fem, _, _ = FX.fem_solve2(n, load, bc, q=q)          # 粗 FE DOF
        dnode = d_fem.reshape(n + 1, n + 1, 4)
        d_fine, _, _ = FX.fem_solve2(nf, load, bc, q=q)         # 细 FE 参考
        zeta = rng.choice([-1.0, 1.0], size=PX.size)
        w_all, kx_a, ky_a, kxy_a, sx_a, sy_a, sxy_a = fe_field_grid(
            nf, d_fine, PX, PY, zeta)
        for c, (ei, ej) in enumerate([(i, j) for i in range(n) for j in range(n)]):
            corners = np.array([dnode[ei, ej], dnode[ei + 1, ej],
                                dnode[ei + 1, ej + 1], dnode[ei, ej + 1]])
            de = corners.ravel()
            sl = slice(c * npts, (c + 1) * npts)
            samples.append((de, np.stack([XI, ETA], axis=1), zeta[sl],
                            w_all[sl], np.stack([kx_a[sl], ky_a[sl], kxy_a[sl]], axis=1),
                            np.stack([sx_a[sl], sy_a[sl], sxy_a[sl]], axis=1), float(q)))
    return samples


def eval_transfer(op, load, bc, q_list, tag_note=""):
    """逐工况：粗 FE DOF → op/BFS vs 细 FE 参考（w_star 解码 SS/Clamped Dirichlet 均合法）"""
    rows = []
    for qv in q_list:
        d_c, _, _ = FX.fem_solve2(NC, load, bc, q=qv)
        d_f, _, _ = FX.fem_solve2(NF, load, bc, q=qv)
        F_op = eval_symavg._assemble(d_c, op, scale=None, npts=15)
        F_bf = assemble_bfs(d_c, n=NC, npts=15)
        Xr, Yr = F_op["X"].ravel(), F_op["Y"].ravel()
        wf, kxf, kyf, kxyf, sxf, syf, sxyf = fe_field_grid(NF, d_f, Xr, Yr)
        def met(F):
            k = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                      np.concatenate([kxf, kyf, kxyf]))
            s = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                      np.concatenate([sxf, syf, sxyf]))
            w = l2rel(F["W"].ravel(), wf)
            ws = None
            try:
                w_star, x_u = S.wstar_from_field(F, npts=15)
                XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
                wr, *_ = fe_field_grid(NF, d_f, XU.ravel(), YU.ravel())
                ws = float(l2rel(w_star.ravel(), wr.ravel()))
            except Exception:
                pass
            return {"w": float(w), "k": float(k), "s": float(s), "w_star": ws}
        row = {"q": qv, "op": met(F_op), "bfs": met(F_bf),
               "op_mxy_center": mxy_center_max(F_op)}
        rows.append(row)
        print(f"  {tag_note}q={qv:.2f}  κ: bfs={row['bfs']['k']:.2e} op={row['op']['k']:.2e}"
              f" | σ: bfs={row['bfs']['s']:.2e} op={row['op']['s']:.2e}"
              f" | w*: op={row['op'].get('w_star')}")
    return rows


def train(mode, epochs, loads):
    t0 = time.time()
    rng = np.random.default_rng(7)
    qs = rng.uniform(S.Q_LO, S.Q_HI, 300)
    qs = qs[(qs < 0.9) | (qs > 1.1)][:loads]
    if mode == "clamp":
        load, bc = "uniform", "clamped"
    else:
        load, bc = "patch", "ss"
    samples = make_fe_dataset2(qs, load, bc)
    op = S.PlateOperator(p=64, hidden=64)
    Lex = S.A / NC
    S.train_operator(op, samples, n_epochs=epochs, batch=24, Le=Lex, scale=None, lam_w=1.0)
    op_path = os.path.join(RES, f"op_state_{mode}.pt")
    torch.save(op.state_dict(), op_path)
    print(f"[transfer.{mode}] trained {epochs}ep/{len(qs)} loads in {time.time()-t0:.0f}s")

    out = {"mode": mode, "load": load, "bc": bc, "epochs": epochs,
           "n_train": len(qs), "note": (
               "Level II retrained on FE labels of the target load/BC family "
               "(input: 4x4 FE DOFs, labels: 16x16 FE fields); architecture unchanged.")}
    if mode == "clamp":
        out["test"] = {
            "in_family_holdout": eval_transfer(op, "uniform", "clamped",
                                               [0.60, 0.80, 0.95, 1.00, 1.40, 1.80], "clamp "),
            "uniform_ss_cross": eval_transfer(op, "uniform", "ss",
                                              [1.0], "ss-cross ")}
    else:
        out["test"] = {
            "patch_holdout": eval_transfer(op, "patch", "ss",
                                           [0.6, 0.8, 0.95, 1.0, 1.4, 1.8], "patch "),
            "patch_offset125": eval_transfer(op, "patch_off", "ss", [1.0], "off "),
            "uniform_ss_cross": eval_transfer(op, "uniform", "ss", [1.0], "uni ")}
    R = load_json()
    R.setdefault("supp_review", {}).setdefault("transfer_study", {})[mode] = out
    save_json(R)
    print(f"saved transfer_study.{mode}")


def load_offset_patch(x, y, q=1.0, half=0.25, cx=0.375):
    m = (np.abs(x - cx) <= half) & (np.abs(y - 0.5) <= half)
    return np.where(m, q, 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clamp", "patch"], required=True)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--loads", type=int, default=48)
    args = ap.parse_args()
    # patch_off 简写注册（在 fe_extra.fem_solve2 不支持时动态处理：改写 qfun）
    if args.mode == "patch":
        import fe_extra as FX
        orig = FX.fem_solve2

        def fs2_off(n, load, bc="ss", q=1.0, **kw):
            if load == "patch_off":
                return orig(n, lambda x, y: load_offset_patch(x, y, q), bc)
            return orig(n, load, bc, q=q)
        FX.fem_solve2 = fs2_off
        import eval_symavg  # noqa: F401  保持引用一致
    train(args.mode, args.epochs, args.loads)
