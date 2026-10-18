# -*- coding: utf-8 -*-
"""rev_n8.py — 训练方板 8×8（Le=1/8）单元算子 op_state_n8.pt
配置与主算子完全一致：no-holdout 140 工况、batch 24、4000 ep、seed 2026。
供矩形板演示（rect_plate.py）与方板 8×8 对照列共用。"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch

import shell_system as S

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026); np.random.seed(2026)

    rng = np.random.default_rng(3)
    qs_all = rng.uniform(S.Q_LO, S.Q_HI, 200)
    qs_train = np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])
    n = 8
    samples = S.make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=args.epochs, batch=24, Le=S.A / n,
                     scale=None, lam_w=1.0, log_every=1000)
    dt = time.time() - t0
    torch.save(op.state_dict(), os.path.join(RES, "op_state_n8.pt"))
    print(f"[n8] trained {dt/60:.1f} min, samples {len(samples)} -> op_state_n8.pt",
          flush=True)


if __name__ == "__main__":
    main()
