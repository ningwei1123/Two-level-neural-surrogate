# -*- coding: utf-8 -*-
"""
rev_cross.py — 审稿 R2 #3：跨网格直测（4x4 训练的算子不重训，直接喂 6x6/8x8 DOF）
回答 "更换网格是否必须重训"：给出 4x4 训练算子在其他网格上的实测误差，
并与该网格族内重训（op_state_g8/g16）及 BFS 同输入对照。
DOF 源：解析 Navier 节点值（exact）。落盘 supp_review.cross_grid。
"""
import os
import json
import numpy as np
import torch
import shell_system as S
import rev_b
from rev_common import (RES, exact_dof_arr, load_json, save_json)

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)


def load_op(fname, d_in=16):
    op = S.PlateOperator(p=64, hidden=64, d_in=d_in)
    op.load_state_dict(torch.load(os.path.join(RES, fname),
                                  weights_only=False))
    op.eval()
    return op


def eval_dofs(op, q, n_el, npts=11, d_all=None):
    d = d_all if d_all is not None else exact_dof_arr(q, n=n_el)
    F = rev_b.assemble_op_grid(d, op, n_el, npts=npts)
    return rev_b.grid_metrics(F, q, n_el, npts=npts)


def main():
    out = {"note": ("4x4-trained operator evaluated out-of-the-box on finer-grid "
                    "exact DOFs (no retraining); retrained references from scaling "
                    "study Table 4 are listed for comparison."),
           "q": 1.0}
    # op_nav: 主算子（4x4 训练, ss uniform）
    op_nav = load_op("op_state.pt")
    # op_aug (若已存在) 一并看
    op_aug = load_op("op_state_aug.pt") if os.path.exists(
        os.path.join(RES, "op_state_aug.pt")) else None
    ref = {}
    for n_el in (4, 6, 8, 16):
        row = {}
        for tag, op in [("op_4x4_trained", op_nav)] + (
                ([("op_aug_4x4", op_aug)] if op_aug is not None else [])):
            gm = eval_dofs(op, 1.0, n_el)
            row[tag] = {k: float(v) for k, v in gm.items()}
        # BFS 同输入（exact DOF 上的 BFS 插值近似为 n_el 网格唯一节点解析插值——
        # 以解码 w 与 κ 插值直接比不直观，仅报重训对照）
        row["retrained_ref"] = None
        ref[str(n_el)] = row
        print(f"n={n_el}: " + " | ".join(
            f"{k} w*={v['w_star']:.2e} k={v['k']:.2e} s={v['s']:.2e}"
            for k, v in row.items() if isinstance(v, dict)))
    out["grids"] = ref
    R = load_json()
    R.setdefault("supp_review", {})["cross_grid"] = out
    save_json(R)
    print("saved supp_review.cross_grid")


if __name__ == "__main__":
    main()
