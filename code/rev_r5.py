# -*- coding: utf-8 -*-
"""
rev_r5.py — R5 量化实验
=====================
(1) r5_abl  含噪（PINN 导出 DOF）消融：full vs no_phys（六项 vs 仅监督 ②④）
    同一 rev_ablate 协议（seed 2026、140 loads、4000 ep、batch 24、n=4、scale=None、
    lam_w=1.0），在 exact-DOF 与 PINN-DOF(q=1) 两种驱动下评估 k/s/w*（npts=11）。
    对照 BFS 插值（同输入）。
(2) r5_cost 训练成本标定：n=4/8/16 各自跑 4 个 epoch 测每 epoch 秒数，
    外推 4000-epoch（匹配预算）训练分钟数；并测每网格一次性的
    dense-LU / SuperLU 因子化时间。
落盘 supp_review.r5_abl / r5_cost（results.json）。
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch

import shell_system as S
import rev_b
from rev_common import (RES as RRES, exact_dof_arr, assemble_bfs,
                        analytic_metrics, load_json, save_json)

torch.set_default_dtype(torch.float64)


def make_qs():
    rng = np.random.default_rng(3)
    qs_all = rng.uniform(S.Q_LO, S.Q_HI, 200)
    return np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])


def train_model(name, weights, qs_train, n_epochs=4000, threads=8, n=4):
    torch.set_num_threads(threads)
    torch.manual_seed(2026); np.random.seed(2026)
    samples = S.make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=n_epochs, batch=24, Le=S.A / n,
                     scale=None, lam_w=1.0, weights=weights, log_every=500)
    dt = time.time() - t0
    print(f"[r5][{name}] trained {dt/60:.1f} min", flush=True)
    return op, dt


def evals(op, pinn, n=4, npts=11):
    out = {}
    de = exact_dof_arr(1.0, n=n)
    F = rev_b.assemble_op_grid(de, op, n, npts=npts)
    out["exact"] = rev_b.grid_metrics(F, 1.0, n, npts=npts)
    dp = rev_b.pinn_dof_grid(pinn, 1.0, n)
    Fp = rev_b.assemble_op_grid(dp, op, n, npts=npts)
    out["pinn"] = rev_b.grid_metrics(Fp, 1.0, n, npts=npts)
    return out


def bfs_evals(pinn, n=4, npts=11):
    out = {}
    de = exact_dof_arr(1.0, n=n)
    out["exact"] = analytic_metrics(assemble_bfs(de, n=n, npts=npts), 1.0,
                                    npts=npts, decode=False)
    dp = rev_b.pinn_dof_grid(pinn, 1.0, n)
    out["pinn"] = analytic_metrics(assemble_bfs(dp, n=n, npts=npts), 1.0,
                                   npts=npts, decode=False)
    return out


def part1():
    import os as _os
    n_epochs = int(_os.environ.get("REV_R5_EPOCHS", "4000"))
    R = load_json()
    out = R.setdefault("supp_review", {}).setdefault("r5_abl", {})
    if out.get("models", {}).get("full") and out.get("models", {}).get("no_phys"):
        print("[r5] abl exists, skip"); return
    qs_train = make_qs()
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    pinn.eval()
    out["meta"] = {"protocol": "rev_ablate same: seed 2026, 140 loads "
                             "(qs[:137]+[1,.98,1.02]), batch 24, "
                             "n=4, scale=None, lam_w=1.0, npts=11, q=1",
                   "epochs": n_epochs,
                   "note": ("epoch budget set by REV_R5_EPOCHS (4000 = full "
                            "protocol of Section 4.9; 800 = paired ranking probe)"),
                   "drivers": ["exact DOF", "PINN DOF (q=1)"],
                   "bfs_reference": True}
    mods = {}
    for name, weights in (("no_phys", {"L3": 0.0, "L5": 0.0, "L6": 0.0}),
                          ("full", {})):
        op, dt = train_model(name, weights, qs_train, n_epochs=n_epochs)
        ev = evals(op, pinn)
        mods[name] = {"weights": weights, "train_s": round(dt, 1),
                      "metrics": {drv: {k: float(v) for k, v in ev[drv].items()}
                                  for drv in ("exact", "pinn")}}
        print(f"[r5][{name}] exact k={ev['exact']['k']:.3e} "
              f"w*={ev['exact']['w_star']:.3e} | pinn k={ev['pinn']['k']:.3e} "
              f"w*={ev['pinn']['w_star']:.3e}", flush=True)
        out["models"] = mods
        save_json(R)
    out["bfs"] = bfs_evals(pinn)
    save_json(R)
    print("[r5] saved r5_abl")


def part2():
    R = load_json()
    out = R.setdefault("supp_review", {}).setdefault("r5_cost", {})
    qs_train = make_qs()
    per_ep = {}
    for n_s in ("4", "8"):
        n = int(n_s)
        torch.set_num_threads(8)
        torch.manual_seed(2026); np.random.seed(2026)
        t0 = time.time()
        samples = S.make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
        t_build = time.time() - t0
        op = S.PlateOperator(p=64, hidden=64)
        t0 = time.time()
        S.train_operator(op, samples, n_epochs=4, batch=24, Le=S.A / n,
                         scale=None, lam_w=1.0, log_every=10000)
        dt = time.time() - t0
        per_ep[n_s] = {"build_s": round(t_build, 1), "n_func": len(samples),
                       "epoch4_s": round(dt, 3),
                       "est_train_4000_min": round(dt / 4 * 4000 / 60, 1)}
        print(f"[r5][cost n={n_s}] per-epoch {dt/4:.3f}s -> "
              f"est 4000ep {dt/4*4000/60:.1f} min (samples {len(samples)})",
              flush=True)
    # 16×16：数据集构建 ~4× of 8×8（element-count），逐 epoch 成本随
    # 单元样本数线性增长 → est ≈ 8×8 值 ×4（标注为估算）
    if "8" in per_ep:
        per_ep["16"] = {
            "est_train_4000_min": round(per_ep["8"]["est_train_4000_min"] * 4.0, 1),
            "note": "estimated by element-count scaling (dataset build and "
                    "per-epoch steps scale ~linearly with n_elem)"}
    # 因子化一次性成本
    import scipy.linalg as sla
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import rev_scale2 as RS
    fac = {}
    for n_s in ("4", "8", "16"):
        n = int(n_s)
        Kf, Fr, free, ndof = RS.build_system(n, 1.0)
        Kd = Kf.toarray()
        t0 = time.time(); sla.lu_factor(Kd); t_d = time.time() - t0
        Kc = Kf.tocsc()
        t0 = time.time(); spla.splu(Kc); t_s = time.time() - t0
        fac[n_s] = {"dense_lu_ms": round(t_d * 1e3, 3),
                    "sparse_splu_ms": round(t_s * 1e3, 3)}
        print(f"[r5][fact n={n_s}] dense-LU {t_d*1e3:.3f} ms, "
              f"SuperLU {t_s*1e3:.3f} ms", flush=True)
    out["per_epoch_est"] = per_ep
    out["factor_ms"] = fac
    out["note"] = ("per-epoch calibration: 4 epochs, threads=8, same protocol "
                   "as Table-4 matched-budget operators (4000 epochs/140 loads); "
                   "est_train_4000_min is the linear extrapolation. "
                   "factor_ms: one-off factorisation per grid.")
    save_json(R)
    print("[r5] saved r5_cost")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="1,2")
    args = ap.parse_args()
    for p in args.parts.split(","):
        if p == "1":
            part1()
        elif p == "2":
            part2()
