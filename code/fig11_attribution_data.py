# -*- coding: utf-8 -*-
"""
fig11_attribution_data.py — single numerical source for Fig. 11

汇总三样东西到 results/v10_diag.json：
  fam        : 三族（SS/patch/clamp）BFS 插值下限 vs 解码算子位移，比值
  ss_fe      : SS 在 4x4 FE DOF 口径下的同协议对照（复算，随脚本更新）
  stage      : 由 _v10_stage_split.py 已落盘的三段计时
正文/表由本文件读值，避免手抄。
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import torch
import shell_system as S
from rev_e2 import eval_transfer

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)

R = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))
sp = R["supp_review"]
ts = sp["transfer_study"]
out = {}

# ---- SS 在 4x4 FE DOF 口径下（与 patch/clamp 完全同协议）----
op = S.PlateOperator(p=64, hidden=64)
op.load_state_dict(torch.load(os.path.join(RES, "seeds_v9", "seed_2026_op.pt"),
                             weights_only=False))
op.eval()
r0 = eval_transfer(op, "uniform", "ss", [1.0], tag_note="[diag] ")[0]
out["ss_fe"] = {"bfs_w": r0["bfs"]["w"], "op_w_star": r0["op"]["w_star"],
                "bfs_k": r0["bfs"]["k"], "op_k": r0["op"]["k"]}


def _mean(rows, who, key):
    v = [r[who][key] for r in rows if isinstance(r.get(who), dict)
         and isinstance(r[who].get(key), (int, float))]
    return float(np.mean(v)) if v else None


hp = ts["patch"]["test"]["patch_holdout"]
hc = ts["clamp"]["test"]["in_family_holdout"]

fam = {}
fam["ss_pinn"] = {"name": "SS uniform, in-family", "driver": "PINN, six seeds",
                  "bfs_w": None, "op_w": None}
fam["ss_fe"] = {"name": "SS uniform, reference seed", "driver": "4\\u00d74 FE",
                "bfs_w": out["ss_fe"]["bfs_w"], "op_w": out["ss_fe"]["op_w_star"]}
fam["patch"] = {"name": "SS, centre patch (retrained)", "driver": "4\\u00d74 FE",
                "bfs_w": _mean(hp, "bfs", "w"), "op_w": _mean(hp, "op", "w_star")}
fam["clamp"] = {"name": "Clamped uniform (retrained)", "driver": "4\\u00d74 FE",
                "bfs_w": _mean(hc, "bfs", "w"), "op_w": _mean(hc, "op", "w_star")}
for k, v in fam.items():
    if v["bfs_w"] and v["op_w"]:
        v["ratio"] = v["op_w"] / v["bfs_w"]        # >1 = operator less accurate
out["fam"] = fam

# ---- 三段计时 ----
st = json.load(open(os.path.join(RES, "v10_stage.json"), encoding="utf-8"))
out["stage"] = {k: v for k, v in st.items() if k.isdigit()}

# ---- 口径标签（A2）----
out["labels"] = {
    "w_dec_six_seed": None,   # 由构建脚本从 t2 读
    "w_dec_ref_seed": None,
    "note": "single-seed diagnostics are labelled (reference seed); headline numbers are six-seed means",
}

with open(os.path.join(RES, "v10_diag.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:1800])
print("saved results/v10_diag.json")
