# -*- coding: utf-8 -*-
"""
rev_ablate.py — R3 #5：六项损失逐项消融（主文级）+ λ_kin 一个量级敏感性
==================================================================
每个变体在完全相同的数据/配置（no-holdout 140 工况、4000 ep、batch 24）
下训练一个 4×4 单元算子，然后评测（exact DOF, q=1, npts=11）：
  闭环节点场（PINN DOF 同输入，需要 seed=2026 级I —— 以 seed_2026 闭环记录
  为准）→ 这里统一用 exact-DOF 报告 k/s/w*，另以现有 PINN seed2026 节点
  自由度做同输入对比（可选 --pinn-dof）。

变体:
  full           : 六项全开（对照组）
  no_L1          : 去掉位移监督头 ①（w 头仅作一致性锚的作用检验）
  no_L3          : 去掉角点插值自洽 ③
  no_L5          : 去掉应力-曲率一致性 ⑤
  no_L6          : 去掉弱正则 ⑥
  lam6_hi        : λ_kin = 1e-1（⑥ 放大 10×）
  lam6_lo        : λ_kin = 1e-3（⑥ 缩小 10×）
落盘 supp_review.ablation（每变体 κ/σ/w* 与相对 full 的变化）。
用法: python rev_ablate.py --epochs 4000 --threads 8 [--variants full,no_L3,no_L5]
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch

import shell_system as S
import rev_b
from rev_common import (RES as RRES, exact_dof_arr, load_json, save_json,
                        l2rel)

torch.set_default_dtype(torch.float64)

VARIANTS = {
    "full":    {},
    "no_L1":   {"L1": 0.0},
    "no_L3":   {"L3": 0.0},
    "no_L5":   {"L5": 0.0},
    "no_L6":   {"L6": 0.0},
    "no_phys": {"L3": 0.0, "L5": 0.0, "L6": 0.0},   # 仅监督项（检验历史"≥10%"说法）
    "lam6_hi": {"L6": 1e-1},
    "lam6_lo": {"L6": 1e-3},
}


def train_one(name, weights, epochs, qs_train, threads=8, n=4):
    torch.set_num_threads(threads)
    torch.manual_seed(2026); np.random.seed(2026)
    samples = S.make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=epochs, batch=24, Le=S.A / n,
                     scale=None, lam_w=1.0, weights=weights, log_every=1000)
    dt = time.time() - t0
    print(f"[{name}] trained in {dt/60:.1f} min", flush=True)
    return op


def eval_op(op, n=4):
    de = exact_dof_arr(1.0, n=n)
    F = rev_b.assemble_op_grid(de, op, n, npts=11)
    return rev_b.grid_metrics(F, 1.0, n, npts=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variants", default="full,no_L1,no_L3,no_L5,no_L6,lam6_hi,lam6_lo")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    qs_all = rng.uniform(S.Q_LO, S.Q_HI, 200)
    qs_train = np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])

    R = load_json()
    ab = R.setdefault("supp_review", {}).setdefault("ablation", {})
    ab["meta"] = {"epochs": args.epochs, "n_train_loads": int(len(qs_train)),
                  "seed": 2026, "config": "no-holdout 140 loads, batch 24",
                  "metric_note": "exact-DOF q=1, npts=11, L2 relative"}
    for name in args.variants.split(","):
        name = name.strip()
        if name in ab and args.skip_existing and "metrics" in ab[name]:
            print(f"[{name}] skip (exists)"); continue
        weights = VARIANTS[name]
        op = train_one(name, weights, args.epochs, qs_train, threads=args.threads)
        gm = eval_op(op)
        ab[name] = {"weights": weights,
                    "metrics": {k: float(v) for k, v in gm.items()}}
        print(f"[{name}] k={gm['k']:.3e} s={gm['s']:.3e} w*={gm['w_star']:.3e}")
        save_json(R)
    # 相对 full 的退化
    if "full" in ab and "metrics" in ab.get("full", {}):
        full = ab["full"]["metrics"]
        ab["delta_vs_full"] = {
            name: {k: (ab[name]["metrics"][k] / full[k] - 1.0) if full[k] else None
                   for k in ("k", "s", "w_star")}
            for name in ab if name not in ("meta", "delta_vs_full")
            and isinstance(ab[name], dict) and "metrics" in ab[name]}
    save_json(R)
    print("saved supp_review.ablation")


if __name__ == "__main__":
    main()
