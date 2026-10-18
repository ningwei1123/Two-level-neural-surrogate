# -*- coding: utf-8 -*-
"""
rev_e.py — 实验 A：非解析载荷（中心 patch）下"代理近似 FE 而非解析解"
回应评审 #1 核心残留：4×4 网格上 BFS-FE ≡ Navier(3e-6) 只是均布载特例；
patch 载荷下粗网格 FE 存在明显离散误差，级Ⅱ（均布族训练）在 OOD DOF 上仍
正确执行 DOF→场恢复，误差主源是粗 DOF 的信息缺失而非标签/算子。

三行对比（均 q=1, patch）：
  BFS(coarse FE DOF)  vs 细 FE 参考  = 粗 FE 离散误差带 δ_fe
  OP (coarse FE DOF)  vs 细 FE 参考  = 端到端（真实闭环）
  OP (细投影 FE DOF)  vs 细 FE 参考  = 信息充足时的恢复能力（OOD 泛化）
落盘 supp_review.patch_study
"""
import os
import json
import numpy as np
import torch
import shell_system as S
import fe_extra as FX
import eval_symavg
from rev_common import (RES, assemble_bfs, l2rel, mxy_center_max,
                        interface_jump, load_json, save_json)

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
NC, NF = 4, 24


def load_ops():
    ops = {}
    for name, fn in [("op_nav", "op_state.pt"), ("op_fe", "op_state_fe.pt")]:
        o = S.PlateOperator(p=64, hidden=64)
        o.load_state_dict(torch.load(os.path.join(RES, fn), weights_only=False))
        o.eval()
        ops[name] = o
    return ops


def ref_at(F, nf, d_f):
    """细 FE 参考场（在 F 的网格采样坐标上）"""
    from rev_common import fe_field_grid
    Xr, Yr = F["X"].ravel(), F["Y"].ravel()
    w, kx, ky, kxy, sx, sy, sxy = fe_field_grid(nf, d_f, Xr, Yr)
    return dict(w=w, kx=kx, ky=ky, kxy=kxy, sx=sx, sy=sy, sxy=sxy)


def err_vs(ref, F, npts=15):
    Xr, Yr = F["X"].ravel(), F["Y"].ravel()
    k = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
              np.concatenate([ref["kx"], ref["ky"], ref["kxy"]]))
    s = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
              np.concatenate([ref["sx"], ref["sy"], ref["sxy"]]))
    w = l2rel(F["W"].ravel(), ref["w"])
    return {"w": float(w), "k": float(k), "s": float(s)}


_CUR = {}


def main():
    ops = load_ops()
    out = {"config": "ss plate, centre patch (half 0.25), q=1",
           "nc": NC, "nf": NF}
    # ---- 细网格参考（patch, ss）----
    d_f, _, _ = FX.fem_solve2(NF, "patch", "ss", q=1.0)
    # 粗网格解
    d_c, _, _ = FX.fem_solve2(NC, "patch", "ss", q=1.0)
    # 细投影到粗节点（信息完整 DOF）
    cn = FX.coarse_nodes_of_fine(NC, NF)
    d_cf = d_f.reshape(-1, 4)[cn].ravel()

    def evaldofs(d_all, tag):
        F_op = eval_symavg._assemble(d_all, ops["op_nav"], scale=None, npts=15)
        F_bfs = assemble_bfs(d_all, n=NC, npts=15)
        ref = ref_at(F_op, NF, d_f)
        eo = err_vs(ref, F_op); eb = err_vs(ref, F_bfs)
        out[tag] = {
            "op": {k: float(v) for k, v in eo.items()},
            "bfs": {k: float(v) for k, v in eb.items()},
            "op_mxy_center": mxy_center_max(F_op),
            "bfs_mxy_center": mxy_center_max(F_bfs),
        }
        print(tag, "OP:", {k: round(v, 4) for k, v in eo.items()},
              "| BFS:", {k: round(v, 4) for k, v in eb.items()})

    evaldofs(d_c, "coarse_FE_DOF")
    evaldofs(d_cf, "fine_proj_DOF")

    R = load_json()
    R.setdefault("supp_review", {})["patch_study"] = out
    save_json(R)
    print("saved supp_review.patch_study")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
