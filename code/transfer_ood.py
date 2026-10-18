# -*- coding: utf-8 -*-
"""
transfer_ood.py — 统一 16×16 FE 参考口径下的 OOD 对照（供表 5 与 4.7 文本）
对 clamped-uniform 与 ss-patch 各 q=1.0 工况：粗(4×4) FE DOF 输入下
  算 BFS 与 op_state.pt / op_state_fe.pt（均为简支均匀族训练）的 κ/σ 误差，
  参考 = 同工况 16×16 FE 场。写 supp_review.transfer_study.ood16。
"""
import os
import numpy as np
import torch
import shell_system as S
import eval_symavg
import fe_extra as FX
from rev_common import (RES, l2rel, fe_field_grid, assemble_bfs,
                        load_json, save_json)

NF, NC = 16, S.N_MESH


def one(load, bc):
    q = 1.0
    d_c, _, _ = FX.fem_solve2(NC, load, bc, q=q)
    d_f, _, _ = FX.fem_solve2(NF, load, bc, q=q)
    out = {}
    F_bf = assemble_bfs(d_c, n=NC, npts=15)
    Xr, Yr = F_bf["X"].ravel(), F_bf["Y"].ravel()
    wf, kxf, kyf, kxyf, sxf, syf, sxyf = fe_field_grid(NF, d_f, Xr, Yr)

    def met(F):
        k = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                  np.concatenate([kxf, kyf, kxyf]))
        s = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                  np.concatenate([sxf, syf, sxyf]))
        return float(k), float(s)

    out["bfs_k"], out["bfs_s"] = met(F_bf)
    for tag, fn in [("op_nav", "op_state.pt"), ("op_fe", "op_state_fe.pt")]:
        op = S.PlateOperator(p=64, hidden=64)
        op.load_state_dict(torch.load(os.path.join(RES, fn), weights_only=False))
        op.eval()
        F_op = eval_symavg._assemble(d_c, op, scale=None, npts=15)
        out[f"{tag}_k"], out[f"{tag}_s"] = met(F_op)
    print(load, bc, {k: round(v, 4) for k, v in out.items()})
    return out


def main():
    out = {
        "clamp": one("uniform", "clamped"),
        "patch": one("patch", "ss"),
        "ref": f"{NF}x{NF} FE, q=1.0, coarse {NC}x{NC} FE DOF input",
    }
    R = load_json()
    R.setdefault("supp_review", {}).setdefault("transfer_study", {})["ood16"] = out
    save_json(R)
    print("saved transfer_study.ood16")


if __name__ == "__main__":
    main()
