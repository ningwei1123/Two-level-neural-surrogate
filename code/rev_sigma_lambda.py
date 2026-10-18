# -*- coding: utf-8 -*-
"""
rev_sigma_lambda.py — 主稿 §4.9 补强：λ₂–₅ 敏感性（±50%） + σ 误差空间结构
=============================================================================
Part 1 (lambda scan): companion 默认协议（140 loads / batch 24 / seed 2026），
  对 L2(曲率监督),L3(角点),L4(应力监督),L5(应力-曲率) 各 ±50% 共 8 变体 @2000ep，
  加 full@2000ep 基线；评测 exact-DOF q=1 npts=11 (k/s/w*)。短程 ranking 探针，
  与 R5 §4.9 配对探针同透明标注；λ1(L1)/λ6(L6) 的 ±50%/10× 证据已有（rev_ablate）。
Part 2 (sigma structure): op_state.pt (companion 主模型) exact q=1 全场 σ 误差按
  空间区域（内部 / 边带 / 角带）分解：各区相对 L2 与质量占比、误差集中度。
输出 results/lambda_scan.json + sigma_spatial.json；控制台表。
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch
import shell_system as S
import rev_b
from rev_common import exact_dof_arr, l2rel

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
EP = 2000
VARIANTS = {"L2": 1.5, "L2lo": 0.5, "L3": 1.5, "L3lo": 0.5,
            "L4": 1.5, "L4lo": 0.5, "L5": 1.5, "L5lo": 0.5}
# name -> (key, value)
VSPEC = [("lam2_hi", "L2", 1.5), ("lam2_lo", "L2", 0.5),
         ("lam3_hi", "L3", 1.5), ("lam3_lo", "L3", 0.5),
         ("lam4_hi", "L4", 1.5), ("lam4_lo", "L4", 0.5),
         ("lam5_hi", "L5", 1.5), ("lam5_lo", "L5", 0.5)]


def make_qs():
    rng = np.random.default_rng(3)
    qs = rng.uniform(S.Q_LO, S.Q_HI, 200)
    return np.concatenate([qs[:137], [1.0, 0.98, 1.02]])


def train_one(weights, qs_train, epochs=EP):
    torch.manual_seed(2026); np.random.seed(2026)
    samples = S.make_operator_dataset(qs_train, n=4, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=epochs, batch=24, Le=S.A / 4,
                     scale=None, lam_w=1.0, weights=weights, log_every=10000)
    return op, time.time() - t0


def ev_op(op):
    d = exact_dof_arr(1.0, 4)
    F = rev_b.assemble_op_grid(d, op, 4, npts=11)
    return rev_b.grid_metrics(F, 1.0, 4, npts=11)


def part1():
    qs = make_qs()
    out = {"meta": {"epochs": EP, "n_train_loads": len(qs), "seed": 2026,
                    "batch": 24, "metric": "exact-DOF q=1 npts=11 rel L2",
                    "note": "short-budget ranking probe; lam1/lam6 evidence in rev_ablate"},
           "runs": {}}
    for name, key, val in [("full", None, None)] + VSPEC:
        wts = {key: val} if key else {}
        op, dt = train_one(wts, qs)
        m = ev_op(op)
        out["runs"][name] = {"weights": wts, "train_s": round(dt, 1),
                             "metrics": {k: float(m[k]) for k in ("k", "s", "w_star")}}
        print(f"[lam] {name:8s} k={m['k']:.4e} s={m['s']:.4e} w*={m['w_star']:.4e} "
              f"({dt/60:.1f} min)", flush=True)
        json.dump(out, open(os.path.join(RES, "lambda_scan.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    # deltas vs full
    f = out["runs"]["full"]["metrics"]
    out["delta_vs_full"] = {n: {k: (r["metrics"][k] / f[k] - 1.0)
                                for k in ("k", "s", "w_star")}
                            for n, r in out["runs"].items()}
    json.dump(out, open(os.path.join(RES, "lambda_scan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[lam] saved results/lambda_scan.json")


def part2():
    """σ 全场误差空间结构：op_state.pt, exact q=1, npts=11 (41×41 唯一网格)。"""
    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"), weights_only=False))
    op.eval()
    d = exact_dof_arr(1.0, 4)
    F = rev_b.assemble_op_grid(d, op, 4, npts=11)
    # 重建唯一网格节点 (grid_metrics 内部同口径)
    from rev_b import unique_idx
    keep = rev_b.unique_idx(4, 11)
    M = keep.size                      # 41
    h = 1.0 / (M - 1)
    xs = np.linspace(0, 1, M)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    # 解析真值 σ (q=1, ζ=+1 顶面)：与 grid_metrics/analytic_metrics 同口径（输入需展平）
    sxa, sya, sxya = S.stress_analytic(X.ravel(), Y.ravel(), 1.0, 1.0)
    Msh = int(np.sqrt(sxa.size))
    sxa = sxa.reshape(Msh, Msh); sya = sya.reshape(Msh, Msh)
    sx_hat = F["SX"][np.ix_(keep, keep)]
    sy_hat = F["SY"][np.ix_(keep, keep)]
    # 分区 mask
    nband = 2                                  # 距离边界 <=2h 为边带
    i_ed = int(nband)
    def zone(name):
        if name == "corner":
            return ((X <= i_ed * h) | (X >= 1 - i_ed * h)) & \
                   ((Y <= i_ed * h) | (Y >= 1 - i_ed * h))
        if name == "edge":
            b = ((X <= i_ed * h) | (X >= 1 - i_ed * h)) | \
                ((Y <= i_ed * h) | (Y >= 1 - i_ed * h))
            c = zone("corner")
            return b & ~c
        return ~(((X <= i_ed * h) | (X >= 1 - i_ed * h)) |
                 ((Y <= i_ed * h) | (Y >= 1 - i_ed * h)))
    res = {"meta": {"model": "op_state.pt", "q": 1.0, "driver": "exact DOF",
                    "npts": 11, "M": M, "edge_band": f"{nband}h",
                    "note": "ζ=+1 top-surface stress"},
           "zones": {}}
    for zn in ("interior", "edge", "corner"):
        mask = zone(zn)
        n = int(mask.sum())
        e_sx = l2rel(sx_hat[mask].ravel(), sxa[mask].ravel())
        e_sy = l2rel(sy_hat[mask].ravel(), sya[mask].ravel())
        # 组合 σ 通道（与全局 σ 指标同源：拼接三个分量）
        h3 = np.concatenate([sx_hat[mask], sy_hat[mask]])
        a3 = np.concatenate([sxa[mask], sya[mask]])
        e_s = l2rel(h3, a3)
        # 该区误差平方占全场误差平方比例（三个分量拼接口径）
        full_mask = np.ones_like(X, dtype=bool)
        e_num_zone = float(((h3 - a3) ** 2).sum())
        res["zones"][zn] = {"nodes": int(n), "frac_nodes": float(n / mask.size),
                            "err_sx": float(e_sx), "err_sy": float(e_sy),
                            "err_s": float(e_s)}
    # 全场口径误差平方（拼接）
    s_full = np.concatenate([sx_hat[full_mask], sy_hat[full_mask]])
    a_full = np.concatenate([sxa[full_mask], sya[full_mask]])
    num_full = float(((s_full - a_full) ** 2).sum())
    sse = {}
    for zn in ("interior", "edge", "corner"):
        msk = zone(zn)
        dzn = np.concatenate([sx_hat[msk], sy_hat[msk]]) - \
              np.concatenate([sxa[msk], sya[msk]])
        sse[zn] = float(((dzn) ** 2).sum()) / num_full
    res["sse_share"] = sse
    json.dump(res, open(os.path.join(RES, "sigma_spatial.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[σ] zones:", {zn: {k: (round(v, 5) if isinstance(v, float) else v)
                              for k, v in d.items() if k != "nodes"}
                         for zn, d in res["zones"].items()})
    print("[σ] SSE share:", {k: round(v, 3) for k, v in res["sse_share"].items()})
    print("[σ] saved results/sigma_spatial.json")


if __name__ == "__main__":
    part2()          # 快，先出
    part1()          # 慢（9×2000ep），后跑
