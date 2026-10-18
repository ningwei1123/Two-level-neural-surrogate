# -*- coding: utf-8 -*-
"""
rev_f.py — 实验 B：固支方板（clamped）均匀载迁移
级Ⅱ（仅在简支均布族上训练）面对"固支 DOF"——训练分布之外的 BC 输入——
仍保持 DOF→场恢复能力（与 BFS 同级/更好），证明单元级算子与边界条件无关；
物理解码器依赖 Dirichlet w=0（固支也满足），可用。落盘 supp_review.clamped_study。
"""
import os
import json
import numpy as np
import torch
import shell_system as S
import fe_extra as FX
import eval_symavg
from rev_common import (RES, assemble_bfs, l2rel, mxy_center_max,
                        load_json, save_json, fe_field_grid)

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
NC, NF = 4, 24


def err_vs(ref, F, npts=15):
    k = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
              np.concatenate([ref["kx"], ref["ky"], ref["kxy"]]))
    s = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
              np.concatenate([ref["sx"], ref["sy"], ref["sxy"]]))
    w = l2rel(F["W"].ravel(), ref["w"])
    return {"w": float(w), "k": float(k), "s": float(s)}


def wstar_err(F, d_f, npts=15):
    """Poisson 解码 w* vs 细 FE 参考（固支 w|∂Ω=0 → Dirichlet 解码合法）"""
    try:
        w_star, x_u = S.wstar_from_field(F, npts=npts)
        XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
        wr, *_ = fe_field_grid(NF, d_f, XU.ravel(), YU.ravel())
        return float(l2rel(w_star.ravel(), wr.ravel()))
    except Exception as e:
        return str(e)


def main():
    for op_name in ["op_nav", "op_fe"]:
        op = S.PlateOperator(p=64, hidden=64)
        op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"
                               if op_name == "op_nav" else "op_state_fe.pt"),
                           weights_only=False))
        op.eval()
        out = {"config": "clamped square plate, uniform load q=1", "nc": NC, "nf": NF}
        d_f, _, _ = FX.fem_solve2(NF, "uniform", "clamped", q=1.0)
        d_c, _, _ = FX.fem_solve2(NC, "uniform", "clamped", q=1.0)
        cn = FX.coarse_nodes_of_fine(NC, NF)
        d_cf = d_f.reshape(-1, 4)[cn].ravel()

        def evaldofs(d_all, tag):
            F_op = eval_symavg._assemble(d_all, op, scale=None, npts=15)
            F_bfs = assemble_bfs(d_all, n=NC, npts=15)
            ref = {}
            Xr, Yr = F_op["X"].ravel(), F_op["Y"].ravel()
            w, kx, ky, kxy, sx, sy, sxy = fe_field_grid(NF, d_f, Xr, Yr)
            ref = dict(w=w, kx=kx, ky=ky, kxy=kxy, sx=sx, sy=sy, sxy=sxy)
            out[tag] = {"op": err_vs(ref, F_op), "bfs": err_vs(ref, F_bfs),
                        "op_wstar": wstar_err(F_op, d_f),
                        "op_mxy_center": mxy_center_max(F_op)}
            print(f"[{op_name}][{tag}] OP={ {k: round(v, 4) for k, v in out[tag]['op'].items()} }"
                  f" BFS={ {k: round(v, 4) for k, v in out[tag]['bfs'].items()} }"
                  f" w*={out[tag]['op_wstar']}")

        evaldofs(d_c, "coarse_FE_DOF")
        evaldofs(d_cf, "fine_proj_DOF")
        R = load_json()
        R.setdefault("supp_review", {}).setdefault("clamped_study", {})[op_name] = out
        save_json(R)
        print("saved clamped_study.", op_name)


if __name__ == "__main__":
    main()
