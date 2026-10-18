# -*- coding: utf-8 -*-
"""
plot_rev_en.py — 三幅补充结果图（英文标注），支撑 4.6/4.7 新增内容：
  fig12_scaling_en.png   4x4/8x8/16x16 墙钟对比（审稿 #3 / Table 4 可视化）
  fig13_symmetry_en.png  x=0.5 对称线上 kappa_xy 伪影证据剖面（审稿 #2 / 4.6 重写段）
  fig14_transfer_en.png  跨族迁移误差对比（审稿 #1 / Table 5 可视化）
全部数值从 results/results.json + 已训练算子实时重算，无需新训练。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import torch
import shell_system as S
import eval_symavg
from rev_common import exact_dof_arr

RES = S.RES
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 8.2, "axes.labelsize": 8.4,
                     "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
                     "legend.fontsize": 8.0, "axes.linewidth": 0.8})
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)

R = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))
sr = R["supp_review"]
BLUE, ORNG, GRN, RED, PURP = "#1565C0", "#E65100", "#2E7D32", "#C62828", "#6A1B9A"

# ---------------------------------------------------------------- Fig 12
def fig_scaling():
    s1 = sr.get("scaling2", {}).get("grids", {}) or {}
    sc = sr["scaling"]["grids"]
    grids = ["4×4", "8×8", "16×16"]
    def pick(g, key2, key1, dft=0.0):
        return s1[g].get(key2) if (g in s1 and s1[g].get(key2) is not None) else sc[g].get(key1, dft)
    fe = [pick(g, "fe_dense_fresh_ms", "fe_ms") for g in ("4", "8", "16")]
    su = [pick(g, "surrogate_ms", "surrogate_ms") for g in ("4", "8", "16")]
    sp = [pick(g, "fe_sparse_fact_ms", "fe_ms") for g in ("4", "8", "16")]
    spd = [fe[i] / su[i] for i in range(3)]
    nv = [sc[g].get("surrogate_naive_loop_ms") for g in ("4", "8", "16")]
    ne = np.array([4.0, 8.0, 16.0])          # elements per side
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(ne, fe, "s-", color=BLUE, lw=1.5, ms=5.5, label="dense FE, fresh per load")
    ax.plot(ne, sp, "d-", color=GRN, lw=1.5, ms=4.8, label="sparse FE, reused factor")
    ax.plot(ne, su, "o-", color=ORNG, lw=1.5, ms=5.5, label="surrogate, batched")
    xv = [i for i, v in enumerate(nv) if v]
    if xv:
        ax.plot([ne[i] for i in xv], [nv[i] for i in xv], color="#777", lw=1.0,
                marker="^", ms=4.5, ls=":", label="surrogate, unbatched loop")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ne); ax.set_xticklabels(grids)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(3.0, 21.0)
    ax.set_xlabel("working grid (elements per side, log)")
    ax.set_ylabel("per-load wall-clock (ms, log)")
    bb = dict(fc="white", ec="none", pad=0.8)
    for i, v in enumerate(spd):
        dy = -9 if i < 2 else 0
        ha = "center" if i < 2 else "left"
        xx = ne[i] if i < 2 else ne[i] * 1.10
        ax.annotate(f"{v:.0f}\u00d7", (xx, su[i]), textcoords="offset points",
                    xytext=(0, dy), ha=ha, va="top" if i < 2 else "center",
                    fontsize=8.0, color=ORNG, bbox=bb)
    for i in range(3):
        ax.annotate(f"{fe[i]:.1f}", (ne[i] * 0.88, fe[i]), ha="right", va="center",
                    fontsize=8.0, color=BLUE, bbox=bb)
    for i in range(3):
        ax.annotate(f"{sp[i]:.3f}" if sp[i] < 0.1 else f"{sp[i]:.2f}",
                    (ne[i] * 0.88, sp[i]), ha="right", va="center",
                    fontsize=8.0, color=GRN, bbox=bb)
    # fitted exponents of the last segment, as quoted in Section 5
    m_su = np.log(su[2] / su[1]) / np.log(2.0)
    m_sp = np.log(sp[2] / sp[1]) / np.log(2.0)
    ax.annotate(f"$n^{{{m_su:.1f}}}$", (11.0, su[2] * 0.42), fontsize=8.0, color=ORNG,
                ha="center", bbox=bb)
    ax.annotate(f"$n^{{{m_sp:.1f}}}$", (11.0, sp[2] * 0.36), fontsize=8.0, color=GRN,
                ha="center", bbox=bb)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
               fontsize=8.0, bbox_to_anchor=(0.52, 0.005))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(RES, "fig12_scaling_en.png"), dpi=300)
    plt.close(fig)
    print("fig12 saved | fe", fe, "sp", sp, "surr", su, "spd", spd,
          "slopes", round(m_su, 2), round(m_sp, 2))


# ---------------------------------------------------------------- Fig 13
def _centreline_rows(F, tol=1e-9):
    """返回 x≈0.5 的全部行索引（单元界面两侧各一行）"""
    Xg = F["X"]
    X = Xg[:, 0] if Xg.ndim == 2 else Xg
    rows = np.where(np.abs(X - 0.5) < tol)[0]
    return rows


def fig_symmetry():
    d = exact_dof_arr(1.0)
    ops = [
        ("plain control", "op_state_plain4.pt", 16, "#1565C0", "-"),
        ("+ antisym. penalty", "op_state_sym2.pt", 16, "#C62828", "--"),
        ("position-aware", "op_state_pos.pt", 18, "#6A1B9A", "-."),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.1, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    ax, ax2 = axes
    for idx, (tag, fn, din, col, ls) in enumerate(ops):
        op = S.PlateOperator(p=64, hidden=64, d_in=din)
        op.load_state_dict(torch.load(os.path.join(RES, fn), weights_only=False))
        op.eval()
        if din == 18:
            import rev_g
            F = rev_g.assemble_pos(d, op, npts=15)
        else:
            F = eval_symavg._assemble(d, op, scale=None, npts=15)
        rows = _centreline_rows(F)
        Y = F["Y"][0, :] if F["Y"].ndim == 2 else F["Y"]
        KXY = np.abs(F["KXY"][rows, :])
        KXX = F["KX"][rows[0], :]
        prof = KXY.max(axis=0)
        ax.plot(Y, prof, ls, color=col, lw=1.5, label=tag)
        if idx == 0:
            ax2.plot(Y, KXX, ls, color=col, lw=1.4, label=r"$\hat\kappa_{xx}$ (control)")
    # 解析尺度
    xs = np.linspace(0, 1, 200)
    kxx_t = S.curv_analytic(np.full_like(xs, 0.5), xs, 1.0)[0]
    ax2.plot(xs, kxx_t, "-", color=GRN, lw=1.1, label=r"$\kappa_{xx}$ (exact)")
    kxx_c = float(kxx_t.max())
    ax.axhline(8.4e-4, color="#888", lw=0.9, ls=":")
    ax.text(0.03, 8.4e-4 * 1.25, "8.4e-4 = 2.3% of κxx(0.5,0.5)", fontsize=8, color="#555")
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 3e-2)
    ax.set_ylabel(r"$|\hat\kappa_{xy}|$ on x=0.5 (log)")
    ax.legend(fontsize=8, loc="upper left", ncol=1)
    ax2.plot(xs, np.zeros_like(xs), color="#888", lw=0.8, ls=":")
    ax2.set_xlabel("y along the symmetry line")
    ax2.set_ylabel(r"$\kappa_{xx}$")
    ax2.legend(fontsize=8, loc="lower center")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig13_symmetry_en.png"), dpi=300)
    plt.close(fig)
    print("fig13 saved | kxx(0.5,0.5)=", kxx_c)


# ---------------------------------------------------------------- Fig 14
def _hmean(rows, who, key):
    vals = [r[who][key] for r in rows
            if isinstance(r.get(who), dict) and isinstance(r[who].get(key), (int, float))]
    return float(np.mean(vals)) if vals else None


def fig_transfer():
    ts = sr["transfer_study"]
    oo = ts["ood16"]
    rows, labels = [], []
    for fam, name in (("clamp", "Clamped, uniform load"),
                      ("patch", "S.S., centre patch")):
        o = oo[fam]
        h = ts[fam]["test"]
        hh = h.get("in_family_holdout") or h.get("patch_holdout") or []
        rows.append([fam, name, o.get("bfs_k"), o.get("bfs_s"),
                     o.get("op_nav_k"), o.get("op_nav_s"),
                     _hmean(hh, "op", "k"), _hmean(hh, "op", "s")])
    x = np.arange(len(rows))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.95), sharey=True)
    for j, ax in enumerate(axes):
        i0, i1, i2 = (2, 4, 6) if j == 0 else (3, 5, 7)   # k: r2/r4/r6 ; s: r3/r5/r7
        ks = [r[i0] * 100 for r in rows]    # BFS
        oo_ = [r[i1] * 100 for r in rows]   # out-of-family
        rt = [r[i2] * 100 for r in rows]    # retrained
        b1 = ax.bar(x - w, ks, w, color="#90A4AE", label="BFS interpolant")
        b2 = ax.bar(x, oo_, w, color="#E65100", alpha=0.75, label="out-of-family OP")
        b3 = ax.bar(x + w, rt, w, color="#1565C0", label="retrained on FE labels")
        ax.set_xticks(x); ax.set_xticklabels([chr(10).join(r[1].split(", ")) for r in rows], fontsize=8.0)
        ax.set_title((r"$\kappa$" if j == 0 else r"$\sigma$") + " error", fontsize=8)
        ax.set_ylim(0, max(oo_ + ks + rt) * 1.22)
        ax.grid(alpha=0.3, axis="y")
        for bars in (b1, b2, b3):
            for b in bars:
                if b.get_height() and b.get_height() > 0:
                    ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2,
                                                          b.get_height() + 0.4),
                                ha="center", fontsize=8.0)
    axes[0].set_ylabel(r"mean rel. $L_2$ error ($\times10^{-2}$)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=8.0, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.115, 1, 0.985))
    fig.savefig(os.path.join(RES, "fig14_transfer_en.png"), dpi=300)
    plt.close(fig)
    for r in rows:
        print("fig14 |", r[1], "bfs", r[2], r[3], "ood", r[4], r[5], "rt", r[6], r[7])


if __name__ == "__main__":
    fig_scaling()
    fig_symmetry()
    fig_transfer()
    print("ALL 3 FIGURES SAVED")
