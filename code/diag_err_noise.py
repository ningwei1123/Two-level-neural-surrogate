# -*- coding: utf-8 -*-
"""
diag_err_noise.py — 主稿 §4.3 两张诊断图（single-seed 2026, 一致口径）
=========================================================================
Fig. A (err attribution, squared-error waterfall):  BFS(exact) -> op(exact) -> op(PINN closed)
  逐级增量以**平方误差份额**给出（各段可加、总和 100%），而非线性 L2 增量：
  The L2 norm is not additive, so a stacked bar of |e| increments is misleading.
  面板 (a) κ (b) σ (c) w_dec。
Fig. B (noise robustness): 对 exact DOF 注入相对幅度 α·std 的高斯噪声，
  测量 op 与 BFS 的 κ/σ 相对误差随 α 的增长（3 次重复取均值），给出放大因子。

用法:
  python diag_err_noise.py            # 重算 + 出图（约数分钟）
  python diag_err_noise.py --replot   # 只读已有 diag_errnoise.json 重绘图
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 8.2, "axes.titlesize": 8.2, "axes.labelsize": 8.2,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
})

NQ, NPTS = 4, 15
ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
NREP = 3
JSON = os.path.join(RES, "diag_errnoise.json")


# ------------------------------------------------------------------ plotting
def shares(b, e, c):
    """squared-error shares (%, additive): floor / operator / input noise"""
    return (100.0 * b * b / (c * c),
            100.0 * (e * e - b * b) / (c * c),
            100.0 * (c * c - e * e) / (c * c))


def _share_label(ax, i, lo, h, col, fs=8.0):
    """A 1 % segment is thinner than a printed rule: label it beside the bar."""
    if h < 4.0:
        ax.annotate(f"{h:.0f}%", xy=(i + 0.31, lo + h / 2.0), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fs, fontweight="bold", color="black",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#555",
                                    shrinkA=0, shrinkB=0))
    else:
        ax.text(i, lo + h / 2.0, f"{h:.0f}%", ha="center", va="center",
                fontsize=fs, fontweight="bold", color=col)


def plot_attribution(out):
    pa = out["partA"]
    scl = 1e2
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.55))
    lab = [("k", r"(a) $\kappa$", "#c44e52"), ("s", r"(b) $\sigma$", "#4c72b0")]
    names = ["BFS\nfloor", "operator\nresidual", "Level-I\nnoise"]
    for ax, (key, tt, col) in zip(axes[:2], lab):
        b = pa["bfs_exact"][key]; e = pa["op_exact"][key]; c = pa["op_pinn"][key]
        s1, s2, s3 = shares(b, e, c)
        seg = [(0.0, s1), (s1, s2), (s1 + s2, s3)]
        for i, (lo, h) in enumerate(seg):
            ax.bar(i, h, bottom=lo, width=0.62, color=col,
                   alpha=[0.45, 0.72, 0.95][i], edgecolor="white", lw=0.6)
            _share_label(ax, i, lo, h, "white" if i == 2 else "black")
        for i in range(2):
            y = sum(seg[j][1] for j in range(i + 1))
            ax.plot([i + 0.31, i + 0.69], [y, y], ls=":", lw=0.8, color="#555")
        ax.axhline(100, ls="--", lw=0.8, color="#333")
        ax.text(2.35, 102, "100%", fontsize=8.0, color="#333", ha="right", va="bottom")
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=8.0)
        ax.set_ylim(0, 118); ax.set_xlim(-0.55, 2.55)
        ax.set_title(tt, loc="left")
        ax.grid(alpha=.25, axis="y")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0].set_ylabel("share of squared error (%)")

    ax = axes[2]
    b = pa["bfs_exact"]["w_star"]; e = pa["op_exact"]["w_star"]; c = pa["op_pinn"]["w_star"]
    s1, s2, s3 = shares(b, e, c)
    seg = [(0.0, s1), (s1, s2), (s1 + s2, s3)]
    for i, (lo, h) in enumerate(seg):
        ax.bar(i, h, bottom=lo, width=0.62, color="#5B8C5A",
               alpha=[0.45, 0.72, 0.95][i], edgecolor="white", lw=0.6)
        _share_label(ax, i, lo, h, "white" if i == 2 else "black")
    for i in range(2):
        y = sum(seg[j][1] for j in range(i + 1))
        ax.plot([i + 0.31, i + 0.69], [y, y], ls=":", lw=0.8, color="#555")
    ax.axhline(100, ls="--", lw=0.8, color="#333")
    ax.text(2.35, 102, "100%", fontsize=8.0, color="#333", ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.0)
    ax.set_ylim(0, 118); ax.set_xlim(-0.55, 2.55)
    ax.set_title(r"(c) $w_{\mathrm{dec}}$", loc="left")
    ax.grid(alpha=.25, axis="y")
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(os.path.join(RES, "fig15_errattribution_en.png"), dpi=300)
    plt.close(fig)
    print("[figA] saved fig15_errattribution_en.png  | k", tuple(round(x, 1) for x in shares(
        pa["bfs_exact"]["k"], pa["op_exact"]["k"], pa["op_pinn"]["k"])),
        "| s", tuple(round(x, 1) for x in shares(
            pa["bfs_exact"]["s"], pa["op_exact"]["s"], pa["op_pinn"]["s"])),
        "| w", tuple(round(x, 1) for x in shares(
            pa["bfs_exact"]["w_star"], pa["op_exact"]["w_star"], pa["op_pinn"]["w_star"])))


def plot_noise(out):
    """2x2: (a)(b) 白噪声 α 扫描（压力测试口径）；(c) 真实 PINN 误差场 β 扫描；
    (d) 同幅度下四种扰动结构的退化倍数。"""
    scl = 1e2
    rbp = os.path.join(RES, "robust_real.json")
    rb = json.load(open(rbp, encoding="utf-8")) if os.path.exists(rbp) else None
    if rb is None:
        return plot_noise_legacy(out)

    C_OP, C_BFS = "#c44e52", "#4c72b0"
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.45))

    for ax, key, lab in ((axes[0, 0], "k", r"(a) $\kappa$, white noise"),
                         (axes[0, 1], "s", r"(b) $\sigma$, white noise")):
        x = ALPHAS
        ax.plot(x, [out["op"][str(a)][key] * scl for a in x], "-o", ms=3, lw=1.2,
                color=C_OP, label="operator")
        ax.plot(x, [out["bfs"][str(a)][key] * scl for a in x], "-s", ms=3, lw=1.2,
                color=C_BFS, label="BFS interpolant")
        ax.set_xlabel(r"injected noise $\alpha$ (global DOF std)")
        if key == "k":      # both panels share the ordinate; one label is enough
            ax.set_ylabel(r"relative $L_2$ error ($\times10^{-2}$)")
        ax.set_title(lab, loc="left")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    bs = [float(b) for b in rb["meta"]["betas"]]
    ax.plot(bs, [rb["realfield"]["bfs"][str(b)]["k"] * scl for b in bs], "-s", ms=3.2,
            lw=1.2, color=C_BFS, label="BFS interpolant")
    ax.plot(bs, [rb["realfield"]["op"][str(b)]["k"] * scl for b in bs], "-o", ms=3.2,
            lw=1.2, color=C_OP, label="operator")
    ax.axvline(1.0, ls=":", lw=0.9, color="#555")
    ax.annotate("closed loop", xy=(1.0, 0.055), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=8.0, color="#444",
                va="bottom")
    ax.set_xlabel(r"scale $\beta$ on measured error field")
    ax.set_ylabel(r"relative $L_2$ error of $\kappa$ ($\times10^{-2}$)")
    ax.set_title(r"(c) measured field $\Delta\mathbf{d}$", loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    keys = ["white_global", "white_percol", "smooth", "measured"]
    names = ["white\n(global)", "white\n(per DOF)", "smooth\n(correlated)", "measured\nfield"]
    x = np.arange(len(keys))
    wd = 0.36
    for i, (k, lab, col) in enumerate((("bfs", "BFS", C_BFS), ("op", "operator", C_OP))):
        vals = [rb["matched"][kk][k] for kk in keys]
        ax.bar(x + (i - 0.5) * wd, vals, wd * 0.9, color=col, label=lab)
        for xi, v in zip(x + (i - 0.5) * wd, vals):
            ax.text(xi, v * 1.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.0)
    ax.set_ylabel("error growth vs clean input")
    ax.set_title(rf"(d) matched $\alpha={rb['alpha_equiv']:.3f}$", loc="left")
    ax.set_ylim(0, max(rb["matched"][k]["bfs"] for k in keys) * 1.36)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.legend(frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    fig.savefig(os.path.join(RES, "fig16_noiserobust_en.png"), dpi=300)
    plt.close(fig)
    print("[figB] saved fig16_noiserobust_en.png (4 panels)")


def plot_noise_legacy(out):
    scl = 1e2
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.35))
    for ax, key, lab in ((axes[0], "k", r"(a) $\kappa$"), (axes[1], "s", r"(b) $\sigma$")):
        x = ALPHAS
        ax.plot(x, [out["op"][str(a)][key] * scl for a in x], "-o", ms=3, lw=1.2,
                color="#c44e52", label="operator")
        ax.plot(x, [out["bfs"][str(a)][key] * scl for a in x], "-s", ms=3, lw=1.2,
                color="#4c72b0", label="BFS interpolant")
        ax.set_xlabel(r"injected noise $\alpha$ (relative to DOF std)")
        ax.set_ylabel(r"relative $L_2$ error ($\times10^{-2}$)")
        ax.set_title(lab, loc="left")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(RES, "fig16_noiserobust_en.png"), dpi=300)
    plt.close(fig)
    print("[figB] saved fig16_noiserobust_en.png (2 panels, robust_real.json missing)")


# ------------------------------------------------------------------ compute
def main():
    replot = "--replot" in sys.argv
    if replot and os.path.exists(JSON):
        out = json.load(open(JSON, encoding="utf-8"))
        if "partA" in out:
            plot_attribution(out)
            plot_noise(out)
            print("[replot] figures regenerated from", JSON)
            return
        print("[replot] partA missing -> recomputing")

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
        F = rev_b.assemble_op_grid(d, op, NQ, npts=NPTS)
        return rev_b.grid_metrics(F, 1.0, NQ, npts=NPTS)

    def ev_bfs(d):
        Fb = assemble_bfs(d, n=NQ, npts=NPTS)
        return analytic_metrics(Fb, 1.0, npts=NPTS, decode=True)

    op = load("op_state.pt", S.PlateOperator, p=64, hidden=64)
    pinn = load("pinn_state.pt", S.PlatePINN, hidden=64, nlayer=4)
    d0 = exact_dof_arr(1.0, NQ)
    dp = rev_b.pinn_dof_grid(pinn, 1.0, NQ)

    m = {}
    for drv, d in (("exact", d0), ("pinn", dp)):
        m["bfs_" + drv] = ev_bfs(d)
        m["op_" + drv] = ev_op(op, d)
    for tag in ("bfs_exact", "op_exact", "op_pinn"):
        print("[partA]", tag, "k=%.4e s=%.4e w*=%.4e"
              % (m[tag]["k"], m[tag]["s"], m[tag]["w_star"]))

    # ---- Part B: noise robustness ----
    std_d = d0.std()
    out = {"meta": {"q": 1.0, "n_el": NQ, "npts": NPTS, "alphas": ALPHAS,
                    "noise_scale": "alpha*std(exact DOF)", "n_rep": NREP,
                    "model": "op_state.pt seed-2026 single"},
           "op": {}, "bfs": {}}
    rng = np.random.default_rng(2026)
    for a in ALPHAS:
        kv = {"k": [], "s": [], "w_star": []}
        for r in range(NREP):
            d = d0 + a * std_d * rng.standard_normal(d0.shape)
            mm = ev_op(op, d)
            for k in kv:
                kv[k].append(mm[k])
        out["op"][str(a)] = {k: float(np.mean(v)) for k, v in kv.items()}
        print(f"[noise] alpha={a:.3f} op k={out['op'][str(a)]['k']:.4e}", flush=True)
    for a in ALPHAS:
        kv = {"k": [], "s": [], "w_star": []}
        for r in range(NREP):
            d = d0 + a * std_d * rng.standard_normal(d0.shape)
            mm = ev_bfs(d)
            for k in kv:
                kv[k].append(mm[k])
        out["bfs"][str(a)] = {k: float(np.mean(v)) for k, v in kv.items()}
        print(f"[noise] alpha={a:.3f} bfs k={out['bfs'][str(a)]['k']:.4e}", flush=True)

    def amp(block, key):
        k0 = block["0.0"][key]; k1 = block["0.08"][key]
        return (k1 - k0) / k0 / 0.08

    out["amp_kappa_op"] = amp(out["op"], "k")
    out["amp_kappa_bfs"] = amp(out["bfs"], "k")
    out["amp_sigma_op"] = amp(out["op"], "s")
    out["amp_sigma_bfs"] = amp(out["bfs"], "s")
    out["amp_wstar_op"] = amp(out["op"], "w_star")
    out["partA"] = {tag: {k: float(m[tag][k]) for k in ("k", "s", "w_star")}
                    for tag in ("bfs_exact", "op_exact", "op_pinn")}
    json.dump(out, open(JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    plot_attribution(out)
    plot_noise(out)
    print("[amps] kappa op %.2f bfs %.2f | sigma op %.2f bfs %.2f"
          % (out["amp_kappa_op"], out["amp_kappa_bfs"], out["amp_sigma_op"],
             out["amp_sigma_bfs"]))
    print("[OK] saved diag_errnoise.json")


if __name__ == "__main__":
    main()
