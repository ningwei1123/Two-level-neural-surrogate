# -*- coding: utf-8 -*-
"""
rev_robust_real.py — 鲁棒性对照实验（回应 V8 审稿 B1）
======================================================
问题：主稿 Fig.12 用**各向同性白噪声**模拟"不精确的驱动 DOF"。白噪声是不相关的
高频扰动，恰恰是"对节点值求二阶导"的最坏情况、对"低通积分"的最好情况，所以该比较
在结构上偏向算子。本脚本用三种扰动结构 + 真实闭环误差场重做，回答两个问题：

  Q1 用真实误差场 Δd = d_PINN − d_exact（每个元素 16 维）驱动时，BFS 与算子谁更差？
  Q2 在**同等幅度**下，扰动结构（白噪 / 逐列白噪 / 光滑相关 / 真实场）各自把两者
     放大多少倍？若结构比幅度更重要，则"85 倍"不能当作方法优势。

输出：results/robust_real.json + results/fig19_noisestructure_en.png

用法：
  python rev_robust_real.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
})

NQ, NPTS = 4, 15
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
ALPHAS = [0.005, 0.01, 0.02, 0.04, 0.08]
NREP = 3
JSON = os.path.join(RES, "robust_real.json")

C_OP, C_BFS = "#c44e52", "#4c72b0"


# --------------------------------------------------------------- 扰动结构
def perturb_global(d0, a, std_all, rng):
    """主稿现有口径：全局 std 归一的白噪声"""
    return d0 + a * std_all * rng.standard_normal(d0.shape)


def perturb_percol(d0, a, col_std, rng):
    """逐列相对白噪声：斜率列按斜率自身 std 归一"""
    return d0 + a * rng.standard_normal(d0.shape) * col_std[None, :]


def _smooth_fields(xs, ys, n_modes, rng):
    """光滑随机场及其一致导数：δw = Σ c_mk sin(mπx)sin(kπy)"""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    f = [np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)]
    for m in range(1, n_modes + 1):
        for k in range(1, n_modes + 1):
            c = rng.standard_normal()
            sm, sk = np.sin(m * np.pi * X), np.sin(k * np.pi * Y)
            cm, ck = np.cos(m * np.pi * X), np.cos(k * np.pi * Y)
            f[0] += c * sm * sk
            f[1] += c * m * np.pi * cm * sk
            f[2] += c * k * np.pi * sm * ck
            f[3] += c * m * k * np.pi ** 2 * cm * ck
    return f


def perturb_smooth(d0, a, col_std, rng, n_modes=2):
    """空间相关的光滑扰动（相关长度 ~ 半个板），各列按自身 std 归一"""
    n = NQ
    xs = np.linspace(0, 1, n + 1)
    ys = np.linspace(0, 1, n + 1)
    f = _smooth_fields(xs, ys, n_modes, rng)
    dd = np.stack([v.ravel() for v in f], axis=1)
    # 逐列缩放到目标 std
    dd = dd / np.maximum(dd.std(axis=0), 1e-30) * (a * col_std)[None, :]
    dd[:, 3] = -dd[:, 3]          # 第四列存 -κxy
    return d0 + dd


# --------------------------------------------------------------- 绘图
def plot(o):
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))

    ax = axes[0]
    b = o["realfield"]
    ax.plot(BETAS, [b["bfs"][str(x)]["k"] * 1e2 for x in BETAS], "-s", ms=3.2,
            lw=1.2, color=C_BFS, label="BFS interpolant")
    ax.plot(BETAS, [b["op"][str(x)]["k"] * 1e2 for x in BETAS], "-o", ms=3.2,
            lw=1.2, color=C_OP, label="element operator")
    ax.axvline(1.0, ls=":", lw=0.9, color="#555")
    ax.text(1.03, ax.get_ylim()[1] * 0.92, "closed loop", fontsize=7,
            color="#444", va="top")
    ax.set_xlabel(r"scale $\beta$ on the measured PINN error field")
    ax.set_ylabel(r"relative $L_2$ error of $\kappa$ ($\times10^{-2}$)")
    ax.set_title(r"(a) real error field, $\Delta d=d_{\mathrm{PINN}}-d_{\mathrm{exact}}$",
                 loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    aq = o["alpha_equiv"]
    names = ["white\n(global)", "white\n(per column)", "smooth\n(correlated)", "measured\nfield"]
    keys = ["white_global", "white_percol", "smooth", "measured"]
    x = np.arange(len(keys))
    wd = 0.36
    for i, (k, lab, col) in enumerate((("bfs", "BFS", C_BFS), ("op", "operator", C_OP))):
        vals = [o["matched"][kk][k] for kk in keys]
        ax.bar(x + (i - 0.5) * wd, vals, wd * 0.92, color=col, label=lab)
        for xi, v in zip(x + (i - 0.5) * wd, vals):
            ax.text(xi, v * 1.04, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.9)
    ax.set_ylabel("error growth factor vs clean input")
    ax.set_title(rf"(b) matched amplitude $\alpha={aq:.3f}$", loc="left")
    ax.set_ylim(0, max(o["matched"][k]["bfs"] for k in keys) * 1.32)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.legend(frameon=False, loc="upper right", ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(os.path.join(RES, "fig19_noisestructure_en.png"), dpi=300)
    plt.close(fig)
    print("[fig] saved fig19_noisestructure_en.png")


# --------------------------------------------------------------- 主流程
def main():
    import torch
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float64)
    import shell_system as S
    import rev_b
    from rev_common import exact_dof_arr, assemble_bfs, analytic_metrics

    def load(path, cls, **kw):
        m = cls(**kw)
        m.load_state_dict(torch.load(os.path.join(RES, path), weights_only=False))
        m.eval()
        return m

    def ev_op(op, d):
        return rev_b.grid_metrics(rev_b.assemble_op_grid(d, op, NQ, npts=NPTS), 1.0, NQ, npts=NPTS)

    def ev_bfs(d):
        return analytic_metrics(assemble_bfs(d, n=NQ, npts=NPTS), 1.0, npts=NPTS, decode=True)

    op = load("op_state.pt", S.PlateOperator, p=64, hidden=64)
    pinn = load("pinn_state.pt", S.PlatePINN, hidden=64, nlayer=4)

    d0 = exact_dof_arr(1.0, NQ)
    dp = rev_b.pinn_dof_grid(pinn, 1.0, NQ)
    delta = dp - d0
    std_all = float(d0.std())
    col_std = d0.std(axis=0)

    clean = {"op": ev_op(op, d0), "bfs": ev_bfs(d0)}
    print("[clean] op k=%.4e bfs k=%.4e" % (clean["op"]["k"], clean["bfs"]["k"]))

    # 真实误差场的等效幅度（两种归一化）
    alpha_eq = float(np.linalg.norm(delta) / np.sqrt(delta.size) / std_all)
    percol_eq = (delta.std(axis=0) / col_std)
    print("[delta] alpha_eq=%.4f  per-column ratios %s" % (alpha_eq, np.round(percol_eq, 4)))

    rng = np.random.default_rng(2026)
    out = {"meta": {"q": 1.0, "n_el": NQ, "npts": NPTS, "betas": BETAS,
                    "alphas": ALPHAS, "n_rep": NREP,
                    "models": "op_state.pt / pinn_state.pt seed-2026"},
           "clean": {k: float(v) for k, v in clean["op"].items()},
           "alpha_equiv": alpha_eq, "percol_equiv": percol_eq.tolist(),
           "realfield": {"op": {}, "bfs": {}}}

    # ---- Q1: 真实误差场 β 扫描 ----
    for b in BETAS:
        d = d0 + b * delta
        mm_o, mm_b = ev_op(op, d), ev_bfs(d)
        out["realfield"]["op"][str(b)] = {k: float(mm_o[k]) for k in ("k", "s", "w_star")}
        out["realfield"]["bfs"][str(b)] = {k: float(mm_b[k]) for k in ("k", "s", "w_star")}
        print("[beta %.2f] op k=%.4e s=%.4e | bfs k=%.4e s=%.4e"
              % (b, mm_o["k"], mm_o["s"], mm_b["k"], mm_b["s"]), flush=True)

    # ---- Q2: 同等幅度下四种结构 ----
    out["sweep"] = {}
    for tag, fn in (("white_global", lambda d: perturb_global(d0, alpha_eq, std_all, rng)),
                    ("white_percol", lambda d: perturb_percol(d0, alpha_eq, col_std, rng)),
                    ("smooth", lambda d: perturb_smooth(d0, alpha_eq, col_std, rng))):
        ko, kb = [], []
        for _ in range(NREP):
            d = fn(None)
            ko.append(ev_op(op, d)["k"]); kb.append(ev_bfs(d)["k"])
        out["sweep"][tag] = {"op": float(np.mean(ko)), "bfs": float(np.mean(kb))}
        print("[sweep %s] op=%.4e bfs=%.4e" % (tag, out["sweep"][tag]["op"],
                                               out["sweep"][tag]["bfs"]), flush=True)

    out["matched"] = {k: {"op": v["op"] / clean["op"]["k"],
                          "bfs": v["bfs"] / clean["bfs"]["k"]}
                      for k, v in out["sweep"].items()}
    out["matched"]["measured"] = {
        "op": out["realfield"]["op"]["1.0"]["k"] / clean["op"]["k"],
        "bfs": out["realfield"]["bfs"]["1.0"]["k"] / clean["bfs"]["k"]}

    json.dump(out, open(JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[matched]", {k: {kk: round(vv, 2) for kk, vv in v.items()}
                        for k, v in out["matched"].items()})
    plot(out)
    print("[OK] saved robust_real.json")


if __name__ == "__main__":
    main()
