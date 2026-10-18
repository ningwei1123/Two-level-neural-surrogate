# -*- coding: utf-8 -*-
"""fig17_diffdesign_en.png — differentiability of the element operator (Section 4.10).

(a) convergence of the element-level target-stress inverse design
(b) target vs recovered top-surface von Mises at the sampled element points
(c) gradient cost: one reverse-mode backward pass vs central finite differences

Canvas is 6.2 in wide (= the column width it is set in), so 7-8 pt text prints at
7-8 pt; see the sizing note in plot_en.py.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
D = json.load(open(os.path.join(RES, "diff_design.json"), encoding="utf-8"))

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 8.2, "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.45))

# ---- (a) inverse-design convergence
h = np.array(D["inverse"]["history"], dtype=float)
h = np.maximum(h, 1e-12)
axes[0].semilogy(np.arange(1, h.size + 1), h, color="#C0392B", lw=1.3, marker="o",
                 ms=2.0, markevery=max(1, h.size // 20))
axes[0].axhline(1e-4, color="#555555", ls=":", lw=0.9)
axes[0].set_xlabel("gradient step")
axes[0].set_ylabel(r"relative field residual $J$")
axes[0].set_title("(a) inverse design", loc="left")
axes[0].grid(alpha=.3, which="both", lw=.4)
axes[0].text(0.97, 0.97, "%d backward\npasses" % D["inverse"]["iters"],
             transform=axes[0].transAxes, ha="right", va="top", fontsize=8.0,
             color="#C0392B", linespacing=1.3)

# ---- (b) target vs recovered von Mises
vt = np.array(D["inverse"]["vm_target"], dtype=float)
vr = np.array(D["inverse"]["vm_recovered"], dtype=float)
idx = np.argsort(vt)
axes[1].plot(vt[idx], vr[idx], "o", ms=2.4, color="#2F6FB3", alpha=.85, lw=0)
lo, hi = vt.min(), vt.max()
axes[1].plot([lo, hi], [lo, hi], "-", color="#888888", lw=1.0)
axes[1].set_xlabel(r"target $\sigma_{vm}$")
axes[1].set_ylabel(r"recovered $\sigma_{vm}$")
axes[1].set_title("(b) prescribed vs recovered", loc="left")
axes[1].grid(alpha=.3, lw=.4)
axes[1].text(0.05, 0.96, "$J=%.1e$" % D["inverse"]["final_field_residual"],
             transform=axes[1].transAxes, fontsize=8.0, color="#2F6FB3", va="top")

# ---- (c) gradient cost
# Lollipops, not bars.  A bar starts at zero, and on a logarithmic axis zero sits at
# minus infinity, so the two bars come out visually similar while the true ratios are
# 16x and 170x.  Marker positions on a log axis encode the ratio as a distance, and
# the multiplier is printed on the lollipop.
c = D["cost"]
y = np.array([0.0, 1.0])
grad = np.array([c["t_grad_element_ms"], c["t_grad_panel_ms"]])
fd = np.array([c["fd_cost_element_ms"], c["fd_cost_panel_ms"]])
ylab = ["16 variables\n(element)", "100 variables\n(panel)"]
for i in range(2):
    axes[2].plot([grad[i], fd[i]], [y[i], y[i]], color="#B0BEC5", lw=1.3, zorder=1)
    axes[2].plot([grad[i]], [y[i]], "o", ms=5.6, color="#C0392B", zorder=3,
                 label="one backward pass" if i == 0 else None)
    axes[2].plot([fd[i]], [y[i]], "s", ms=5.6, color="#7F8C8D", zorder=3,
                 label="central differences" if i == 0 else None)
    axes[2].text(np.sqrt(grad[i] * fd[i]), y[i] + 0.32, "%.0f$\\times$" % (fd[i] / grad[i]),
                 ha="center", va="bottom", fontsize=8.0, color="#333")
    axes[2].annotate("%.1f" % grad[i], (grad[i], y[i]), textcoords="offset points",
                     xytext=(5, 6), ha="left", va="bottom", fontsize=8.0, color="#C0392B")
    axes[2].annotate("%.0f" % fd[i], (fd[i], y[i]), textcoords="offset points",
                     xytext=(0, -7), ha="center", va="top", fontsize=8.0, color="#455A64")
axes[2].set_xscale("log")
axes[2].set_ylim(-0.45, 1.72)
axes[2].set_yticks(y); axes[2].set_yticklabels(ylab, fontsize=8.0)
axes[2].tick_params(axis="y", length=0)
axes[2].set_xlabel("wall-clock (ms)")
axes[2].set_title("(c) gradient cost", loc="left")
axes[2].grid(alpha=.3, axis="x", which="both", lw=.4)
# marker legend is in the caption (circles: one reverse-mode pass, squares:
# central differences); an in-panel legend collided with the design-variable rows

fig.tight_layout(w_pad=1.0)
fig.subplots_adjust(right=0.985)   # (c) 标题/轴标题右端留量
fig.savefig(os.path.join(RES, "fig17_diffdesign_en.png"), dpi=300)
print("[fig17] saved fig17_diffdesign_en.png")
