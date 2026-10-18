# -*- coding: utf-8 -*-
"""
rev_seed_stats.py — 聚合 results/seeds/seed_*.json → supp_review.seed_stats
==================================================================
对每个指标同时给 跨种子均值(mean)/标准差(std)/最小-最大，并派生每个种子
的"同一输入比例"（k_bf/k_op、s_bf/s_op、w_bf/w_star 等）后做跨种子聚合，
保证正文叙述（"比 BFS 准 X%"、"-43% 闭环误差"）与表格数字同源一致。

落盘字段（含闭环与泛化的字段级 mean±std 与派生 ratio）：
  supp_review.seed_stats = {
    seeds:[...], n:int,
    pinn: {l2:{mean,std}, max_abs:{...}, node:{w:..}},   # 级I 全场
    closed_op: {w,kxx,kyy,kxy,sxx,syy,sxy,w_star -> {mean,std,min,max}},
    closed_bfs: {同键},
    gen: {w_star_mean:{mean,std}, w_star_cv:{mean,std}, k_mean:{...}, k_cv, s_mean, s_cv},
    derived: {k_ratio(bf/op)..., w_ratio_bf_op, w_ratio_bf_star, ...},
  }
用法: python rev_seed_stats.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")
SEED_DIR = os.path.join(RES, "seeds")
from rev_common import load_json, save_json


def agg(vals):
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                min=float(a.min()), max=float(a.max()))


def main():
    files = sorted(f for f in os.listdir(SEED_DIR) if f.startswith("seed_") and f.endswith(".json"))
    recs = [json.load(open(os.path.join(SEED_DIR, f), encoding="utf-8")) for f in files]
    if not recs:
        print("no seed files"); return
    seeds = [r["seed"] for r in recs]
    n = len(recs)

    def col(path):
        vals = []
        for r in recs:
            v = r
            for k in path.split("."):
                v = v[k]
            vals.append(float(v))
        return vals

    # 字段级聚合
    closed_op, closed_bfs, gen = {}, {}, {}
    for k in ("w", "kxx", "kyy", "kxy", "sxx", "syy", "sxy", "w_star", "k", "s"):
        closed_op[k] = agg(col(f"closed.pinn_op.{k}"))
        if k in ("w", "kxx", "kyy", "kxy", "sxx", "syy", "sxy", "k", "s"):
            closed_bfs[k] = agg(col(f"closed.pinn_bfs.{k}"))
    for k in ("w_star_mean", "w_star_std", "w_star_cv", "k_mean", "k_std", "k_cv",
              "s_mean", "s_std", "s_cv"):
        gen[k] = agg(col(f"gen30.{k}"))
    pinn = {k: agg(col(f"pinn.{k}")) for k in ("l2", "max_abs")}
    pinn["node"] = {k: agg(col(f"pinn.node.{k}"))
                    for k in ("w_max", "wx_max", "wy_max", "wxy_max")}

    # 每种子派生：三量平均（与论文 Table 2 "κ/σ (avg.)" 口径一致）
    def avg3(tag, comps):
        return [sum(col(f"closed.{tag}.{c}")[i] for c in comps) / 3.0
                for i in range(n)]
    avgs = {
        "k_op": agg(avg3("pinn_op", ("kxx", "kyy", "kxy"))),
        "k_bf": agg(avg3("pinn_bfs", ("kxx", "kyy", "kxy"))),
        "s_op": agg(avg3("pinn_op", ("sxx", "syy", "sxy"))),
        "s_bf": agg(avg3("pinn_bfs", ("sxx", "syy", "sxy"))),
        "w_head": agg(col("closed.pinn_op.w")),
        "w_bf": agg(col("closed.pinn_bfs.w")),
        "w_star": agg(col("closed.pinn_op.w_star")),
    }
    # 每种子比例（BFS 分母口径与 Table 2 列标题一致）
    def ratio(num_path, den_path):
        return [col(num_path)[i] / col(den_path)[i] for i in range(n)]
    derived = {
        "k_bf_over_k_op": agg(ratio("closed.pinn_bfs.k", "closed.pinn_op.k")),
        "s_bf_over_s_op": agg(ratio("closed.pinn_bfs.s", "closed.pinn_op.s")),
        "w_bf_over_w_op": agg(ratio("closed.pinn_bfs.w", "closed.pinn_op.w")),
        "w_bf_over_w_star": agg(ratio("closed.pinn_bfs.w", "closed.pinn_op.w_star")),
    }
    out = dict(seeds=seeds, n=n, pinn=pinn, closed_op=closed_op,
               closed_bfs=closed_bfs, gen=gen, derived=derived, avg=avgs,
               note="mean/std/min/max across independent random seeds; derived "
                    "ratios and three-component averages computed per seed then "
                    "aggregated.")
    R = load_json()
    R.setdefault("supp_review", {})["seed_stats"] = out
    save_json(R)
    print(json.dumps(out, ensure_ascii=False)[:400])
    print("saved supp_review.seed_stats (n=%d)" % n)


if __name__ == "__main__":
    main()
