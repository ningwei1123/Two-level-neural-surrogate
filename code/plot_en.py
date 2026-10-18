# -*- coding: utf-8 -*-
"""
Plot the full figure set with ENGLISH in-figure text for the Thin-Walled Structures
manuscript.  Figures are saved as <name>_en.png alongside the Chinese figures.

V8 sizing policy
----------------
A figure drawn 11.5 in wide and then set in a 6.2 in column is shrunk by 1.85x, so a
10 pt label prints at 5.4 pt.  Every figure is therefore drawn at (approximately) the
width at which it is inserted, and all in-figure text is specified in the 7-8.5 pt band.
Diagram axes use one data unit = one inch, so box sizes below are literal inches.

Requires trained models (pinn_state.pt / op_state.pt) and saved loss histories.
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import shell_system as S

RES = S.RES
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 8.2, "axes.titlesize": 8.2, "axes.labelsize": 8.4,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0, "figure.titlesize": 8.6,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5,
})

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)

A, B, D, NU, H = S.A, S.B, S.D_FLEX, S.NU, S.H_TH
n = S.N_MESH
Lex = A / n
xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
Xnr, Ynr = Xn.ravel(), Yn.ravel()


# ---------------------------------------------------------------- helpers
def diagram(w, h):
    """axes covering the whole canvas, 1 data unit = 1 inch"""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc="#E8F1FB", ec="#2F6FB3", fs=7.4, bold=True, lw=1.1):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#1a1a1a",
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p0, p1, color="#444444", ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def row(w, widths, gap, y, h):
    tot = sum(widths) + gap * (len(widths) - 1)
    x = (w - tot) / 2.0
    out = []
    for bw in widths:
        out.append((x + bw / 2.0, y, bw, h))
        x += bw + gap
    return out


def load_models():
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"),
                                  weights_only=False))
    return pinn, op


def dof_from_field(X, Y, q=1.0, pinn=None):
    if pinn is None:
        w0 = S.w_analytic(X, Y, q); wx0, wy0 = S._w_grad(X, Y, q)
        kxy0 = S.curv_analytic(X, Y, q)[2]
        return np.stack([w0, wx0, wy0, -kxy0], axis=1).ravel()
    f = S.pinn_field(pinn, np.stack([X, Y], axis=1), q)
    return np.stack(f, axis=1).ravel()


def element_predict(op, de, pts, Lex):
    d_t = torch.tensor(de).reshape(1, 16)
    with torch.no_grad():
        wk = op.wk_at(d_t, torch.tensor(pts)).numpy()
        sp = op.s_at(d_t, torch.stack(
            [torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1]),
             torch.full((pts.shape[0],), 1.0)], dim=1)).numpy()
    w_bf, kxx, kyy, kxy = S.bfs_interp(de, pts[:, 0], pts[:, 1], Lex, Lex)
    k_bf = np.stack([kxx, kyy, kxy], axis=1)
    mx = S.D_FLEX * (k_bf[:, 0] + S.NU * k_bf[:, 1])
    my = S.D_FLEX * (k_bf[:, 1] + S.NU * k_bf[:, 0])
    mxy = S.D_FLEX * (1 - S.NU) * k_bf[:, 2]
    s_bf = (6.0 / S.H_TH ** 2) * np.stack([mx, my, mxy], axis=1)
    return (wk[:, 0], wk[:, 1:4], sp, w_bf, k_bf, s_bf)


def assemble_field(op, dof, src="op", npts=15, zeta_top=True):
    n1 = npts
    g = np.linspace(-1, 1, npts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    F = {}
    for k in ["W", "KX", "KY", "KXY", "SX", "SY", "SXY"]:
        F[k] = np.zeros((n * n1, n * n1))
    XW = np.zeros_like(F["W"]); YW = np.zeros_like(F["W"])
    for ei in range(n):
        for ej in range(n):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dof.reshape(-1, 4)[ids].ravel()
            w_op, k_op, s_op, w_bf, k_bf, s_bf = element_predict(op, de, pts, Lex)
            if src == "op":
                w, k, s = w_op, k_op, s_op
            else:
                w, k, s = w_bf, k_bf, s_bf
            si = slice(ei * n1, (ei + 1) * n1); sj = slice(ej * n1, (ej + 1) * n1)
            F["W"][si, sj] = w.reshape(n1, n1)
            F["KX"][si, sj] = k[:, 0].reshape(n1, n1)
            F["KY"][si, sj] = k[:, 1].reshape(n1, n1)
            F["KXY"][si, sj] = k[:, 2].reshape(n1, n1)
            F["SX"][si, sj] = s[:, 0].reshape(n1, n1)
            F["SY"][si, sj] = s[:, 1].reshape(n1, n1)
            F["SXY"][si, sj] = s[:, 2].reshape(n1, n1)
            xg = xs[ei] + 0.5 * Lex * (pts[:, 0] + 1)
            yg = ys[ej] + 0.5 * Lex * (pts[:, 1] + 1)
            XW[si, sj] = xg.reshape(n1, n1); YW[si, sj] = yg.reshape(n1, n1)
    F["X"] = XW; F["Y"] = YW
    return F


def von_mises(sxx, syy, sxy):
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)


def main():
    R = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))
    pinn, op = load_models()
    hist_pinn = np.load(os.path.join(RES, "hist_pinn.npy"))
    z = np.load(os.path.join(RES, "hist_op.npz"))
    hist_op = {k: z[k] for k in ["L1", "L2", "L3", "L4", "L5", "L6"]}

    q = 1.0
    d_ex = dof_from_field(Xnr, Ynr, q)
    d_pin = dof_from_field(Xnr, Ynr, q, pinn=pinn)
    Fop = assemble_field(op, d_pin, "op", npts=15)
    Fbf = assemble_field(op, d_pin, "bfs", npts=15)

    # ---------- Fig 2 (benchmark sketch) ----------
    # 正方形板：set_aspect("equal") 保证 a=b=1 不被拉成矩形；四边同一套 SS 符号；
    # 板内不再画箭头（易被读成节点），改为 4x4 工作网格虚线。
    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    ax.set_xlim(-0.42, 2.55); ax.set_ylim(-0.44, 1.50)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1.0, 1.0, fc="#E8F1FA", ec="#2F6FB3", lw=1.6, zorder=1))
    for u in (0.25, 0.5, 0.75):
        ax.plot([u, u], [0, 1], color="#7FA8CC", lw=0.55, ls=(0, (3.2, 2.6)), zorder=2)
        ax.plot([0, 1], [u, u], color="#7FA8CC", lw=0.55, ls=(0, (3.2, 2.6)), zorder=2)

    # 四边简支符号：边外细线 + 指向板内的小三角形，四边完全同构
    t, hb, gap = 0.052, 0.030, 0.014
    ax.plot([0, 1], [-gap, -gap], color="#546E7A", lw=1.0, zorder=1)
    ax.plot([0, 1], [1 + gap, 1 + gap], color="#546E7A", lw=1.0, zorder=1)
    ax.plot([-gap, -gap], [0, 1], color="#546E7A", lw=1.0, zorder=1)
    ax.plot([1 + gap, 1 + gap], [0, 1], color="#546E7A", lw=1.0, zorder=1)
    for u in np.linspace(0.08, 0.92, 7):
        for apex, b1, b2 in [
            ((u, 0.0), (u - hb, -t), (u + hb, -t)),
            ((u, 1.0), (u - hb, 1.0 + t), (u + hb, 1.0 + t)),
            ((0.0, u), (-t, u - hb), (-t, u + hb)),
            ((1.0, u), (1.0 + t, u - hb), (1.0 + t, u + hb)),
        ]:
            ax.add_patch(plt.Polygon(np.array([apex, b1, b2]), fc="#546E7A",
                                     ec="#37474F", lw=0.5, zorder=3))

    # 均布载荷：只画在上边，红色向下箭头
    for xx in np.linspace(0.10, 0.90, 5):
        ax.annotate("", xy=(xx, 1.135), xytext=(xx, 1.30),
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=0.9,
                                    shrinkA=0, shrinkB=0))
    ax.text(0.5, 1.415, r"$q(x,y)=q$", ha="center", va="center", fontsize=9,
            color="#C0392B")
    ax.text(0.5, 1.345, r"uniform transverse load", ha="center", va="center",
            fontsize=8.0, color="#C0392B")

    # 坐标轴与尺寸
    ax.annotate("", xy=(1.16, -0.13), xytext=(0.0, -0.13),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9))
    ax.annotate("", xy=(-0.13, 1.16), xytext=(-0.13, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.9))
    ax.text(1.23, -0.13, r"$x$", fontsize=9, va="center", ha="left")
    ax.text(-0.13, 1.23, r"$y$", fontsize=9, ha="center", va="bottom")
    ax.text(0.5, -0.235, r"$a=1$", fontsize=8, ha="center", va="center")
    ax.text(-0.255, 0.5, r"$b=1$", fontsize=8, rotation=90, ha="center", va="center")

    # 右侧注记列（不再压住符号）
    ax.text(1.16, 0.90, "simply supported on all four edges", fontsize=8.0,
            color="#37474F", ha="left", va="center")
    ax.text(1.16, 0.78, r"$w=0$,   $M_n=0$", fontsize=8, color="#37474F",
            ha="left", va="center")
    ax.text(1.16, 0.60, r"working grid: $4\times4$ elements", fontsize=8.0,
            color="#2F6FB3", ha="left", va="center")
    ax.text(1.16, 0.40, r"$D=Eh^{3}/[12(1-\nu^{2})]=1$", fontsize=8.0,
            color="#37474F", ha="left", va="center")
    ax.text(1.16, 0.27, r"$h=1,\ \ \nu=0.3$", fontsize=8.0, color="#37474F",
            ha="left", va="center")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f"{RES}/fig1_plate_setup_en.png", dpi=300); plt.close(fig)

    # ---------- Fig 3 (Level-I architecture) ----------
    W, Hh = 6.1, 2.26
    fig, ax = diagram(W, Hh)
    labs = [("Input\n$(x,y)$", 0.85), ("MLP\n$64\\times4$", 1.00),
            ("$\\sin\\pi x\\,\\sin\\pi y$\nhard BC", 1.45),
            ("$\\hat u(x,y)$", 0.85), ("$\\hat w=q\\,\\hat u$", 0.95)]
    cells = row(W, [c[1] for c in labs], 0.17, 1.80, 0.50)
    for (cx, cy, bw, bh), (txt, _) in zip(cells, labs):
        box(ax, cx, cy, bw, bh, txt, fs=7.2)
    for i in range(len(cells) - 1):
        arrow(ax, (cells[i][0] + cells[i][2] / 2, 1.80),
              (cells[i + 1][0] - cells[i + 1][2] / 2, 1.80))
    box(ax, cells[4][0], 1.24, 0.60, 0.30, "$q$", fc="#FFF3E0", ec="#E65100", fs=7.2)
    arrow(ax, (cells[4][0], 1.39), (cells[4][0], 1.55))
    box(ax, W / 2, 0.55, 5.80, 0.72,
        "$\\mathcal{L}[\\hat u]=\\frac{1}{N}\\sum_j\\left\\{\\frac{D}{2}[g(\\hat u)]_j-\\hat u_j\\right\\}\\to\\min$"
        "      (energy loss, Eq. 11)\n"
        "$g(\\hat u)=(\\hat u_{xx}+\\hat u_{yy})^{2}-2(1-\\nu)(\\hat u_{xx}\\hat u_{yy}-\\hat u_{xy}^{2})$",
        fc="#FEF3E2", ec="#C07A00", fs=7.4, bold=False)
    arrow(ax, (cells[3][0], 1.55), (cells[3][0], 0.92), ls=":", color="#B26A00")
    # label kept clear of the dotted arrow (it used to be centred on it and read as a
    # stray vertical tick)
    ax.text(cells[3][0] - 0.09, 1.20, "minimised by Adam", fontsize=8.0,
            color="#B26A00", ha="right", va="center")
    fig.savefig(f"{RES}/fig2_pinn_arch_en.png", dpi=300); plt.close(fig)

    # ---------- Fig 6 (Level-II operator architecture) ----------
    # The displacement head w_e is training-only (at inference the displacement comes
    # from the decoder): it is drawn greyed, above a dotted divider, so the core
    # negative result cannot be misread from the figure.
    W, Hh = 6.2, 2.95
    fig, ax = diagram(W, Hh)
    box(ax, 0.95, 2.40, 1.65, 0.58, "Element DOFs\n$\\mathbf{d}_e\\in\\mathbb{R}^{16}$", fs=7.2)
    box(ax, 2.90, 2.40, 1.60, 0.58, "branch net\n$\\mathbb{R}^{16}\\!\\to\\!\\mathbb{R}^{p}$",
        fc="#EFEBE9", ec="#6D4C41", fs=7.2)
    box(ax, 2.90, 1.40, 1.60, 0.62, "trunk-$w/\\kappa$\n$\\mathbb{R}^{2}\\!\\to\\!\\mathbb{R}^{4p}$  $(\\xi,\\eta)$",
        fc="#E0F7FA", ec="#00838F", fs=7.0)
    box(ax, 2.90, 0.45, 1.60, 0.62, "trunk-$\\sigma$\n$\\mathbb{R}^{3}\\!\\to\\!\\mathbb{R}^{3p}$  $(\\xi,\\eta,\\zeta)$",
        fc="#E0F7FA", ec="#00838F", fs=7.0)
    ax.add_patch(FancyBboxPatch((4.12, 0.70), 1.95, 1.40,
                                boxstyle="round,pad=0.012", fc="#F3E5F5",
                                ec="#6A1B9A", lw=1.1))
    ax.text(5.095, 1.90, "$\\hat w_e=\\langle b,t_w\\rangle$", ha="center", va="center",
            fontsize=8.0, color="#9E9E9E")
    ax.text(5.095, 1.66, "(training only)", ha="center", va="center",
            fontsize=8.0, color="#9E9E9E", style="italic")
    ax.plot([4.20, 6.00], [1.51, 1.51], ls=(0, (1.6, 1.6)), lw=0.8, color="#B39DDB")
    ax.text(5.095, 1.28, "$\\hat\\kappa_{xx},\\hat\\kappa_{yy},\\hat\\kappa_{xy}$",
            ha="center", va="center", fontsize=8.0)
    ax.text(5.095, 0.98, "$\\hat\\sigma_{xx},\\hat\\sigma_{yy},\\hat\\sigma_{xy}$",
            ha="center", va="center", fontsize=8.0)
    arrow(ax, (1.78, 2.40), (2.10, 2.40))
    arrow(ax, (3.70, 2.40), (4.12, 1.95), color="#6D4C41")
    arrow(ax, (3.70, 1.40), (4.12, 1.28))
    arrow(ax, (3.70, 0.45), (4.12, 0.96))
    ax.add_patch(FancyBboxPatch((0.18, 1.02), 1.55, 0.82,
                                boxstyle="round,pad=0.012", fc="#FFF8E1",
                                ec="#C07A00", lw=0.8, ls="--"))
    ax.text(0.955, 1.43, "trained by\n4 consistency terms\n+ data supervision",
            fontsize=8.0, ha="center", va="center", color="#8D5100", linespacing=1.45)
    arrow(ax, (1.75, 1.43), (2.10, 1.43), color="#C07A00", ls=":")
    fig.savefig(f"{RES}/fig3_operator_arch_en.png", dpi=300); plt.close(fig)

    # ---------- Fig 1 (pipeline) ----------
    # Horizontal five-stage chain: each stage carries its own note underneath, so no
    # annotation is pushed against the right margin, and the block is ~5.4 cm tall
    # instead of ~9.5 cm.
    W, Hh = 6.1, 2.12
    fig, ax = diagram(W, Hh)
    bw, gap = 1.06, 0.16
    tot = 5 * bw + 4 * gap
    x0 = (W - tot) / 2.0
    yb, bh = 1.50, 0.66
    boxes = [
        ("Physical\nproblem\nthin plate, $q$", "#E8F1FB", "#2F6FB3"),
        ("Level I\nenergy PINN\n$\\to$ nodal DOFs", "#E3F2FD", "#1565C0"),
        ("Level II\nelement\noperator\n$\\to\\kappa,\\ \\sigma$", "#F3E5F5", "#6A1B9A"),
        ("Poisson\ndecoder\n$\\to w_{\\mathrm{dec}}$", "#E8F5E9", "#2E7D32"),
        ("$w_{\\mathrm{dec}},\\ \\kappa,\\ \\sigma$\non the grid", "#E8F1FB", "#2F6FB3"),
    ]
    notes = [
        "geometry and\nload, no labels",
        "replaces the\nassembly and\nsolve $\\mathbf{K}\\mathbf{d}=\\mathbf{f}$",
        "replaces\n$\\mathbf{N},\\mathbf{B}$\npost-processing",
        "displacement is\nintegrated,\nnot regressed",
        "trained off-line;\nLevel I needs\nno labels",
    ]
    for i, ((txt, fc, ec), nt) in enumerate(zip(boxes, notes)):
        cx = x0 + i * (bw + gap) + bw / 2.0
        box(ax, cx, yb, bw, bh, txt, fc=fc, ec=ec, fs=7.2)
        ax.text(cx, 0.92, nt, fontsize=8.0, ha="center", va="top",
                color="#444444", linespacing=1.38)
        if i < 4:
            arrow(ax, (cx + bw / 2 + 0.015, yb), (cx + bw / 2 + gap - 0.015, yb))
    _p = f"{RES}/fig4_system_pipeline_en.png"
    fig.savefig(_p, dpi=300); plt.close(fig)

    # trim the white margins so the diagram fills the width it is given in the
    # manuscript (matplotlib's tight bbox cannot help: the axes span the canvas)
    from PIL import Image as _Im
    _im = _Im.open(_p).convert("RGB")
    _a = np.array(_im.convert("L")) < 250
    _cs = np.where(_a.any(0))[0]; _rs = np.where(_a.any(1))[0]
    _pad = 10
    _im.crop((max(0, _cs[0] - _pad), max(0, _rs[0] - _pad),
              min(_im.width, _cs[-1] + _pad), min(_im.height, _rs[-1] + _pad))
             ).save(_p, dpi=(300, 300))
    print("[fig1] pipeline cropped to", _Im.open(_p).size)

    # ---------- Fig 4 (Level-I energy history) ----------
    # hist_pinn = n_adam Adam records + ONE final L-BFGS evaluation on a fixed set
    h = np.asarray(hist_pinn).astype(float)
    n_adam = len(h) - 1
    ha = h[:n_adam]
    tail = ha[-500:]
    L_final = float(np.mean(tail))
    sd_tail = float(np.std(tail))
    lbfgs = float(h[-1])
    ep = np.arange(n_adam)
    neg = np.where(ha < 0)[0]
    ep_zero = int(neg[0]) if len(neg) else 0
    last_pos = np.where(ha >= 0)[0]
    ep_settle = int(last_pos[-1]) + 1 if len(last_pos) else ep_zero
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.45),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(ep, ha, color="#1565C0", lw=0.7, alpha=.30, label="raw", zorder=2)
    k = 201
    sm = np.convolve(ha, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k - 1, len(ha)), sm, color="#0D47A1", lw=1.4,
            label="201-epoch mean", zorder=3)
    ax.axhline(0.0, color="#888", lw=0.7, ls="--", zorder=1)
    ax.axvline(ep_settle, color="#C62828", lw=0.7, ls=":", zorder=1)
    ax.text(ep_settle * 2.2, 9.0, f"zero crossing: {ep_zero}\nnegative from {ep_settle}",
            fontsize=8.0, color="#C62828", va="top", ha="left")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.45)
    ax.set_ylim(-3e-3, 40.0)
    ax.set_xscale("symlog", linthresh=10, linscale=0.55)
    ax.set_xlim(-1, n_adam * 1.15)
    ax.set_xlabel("Adam epoch"); ax.set_ylabel(r"$\mathcal{L}$ (symlog)")
    ax.set_title("(a) full Adam phase", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), frameon=True,
              framealpha=.9, edgecolor="none", fontsize=8.0)
    ax.grid(alpha=.3)

    ax = axes[1]
    m = np.arange(n_adam) >= n_adam - 500
    ax.plot(ep[m], tail * 1e4, color="#1565C0", lw=0.7, zorder=2)
    ax.axhline(L_final * 1e4, color="#0D47A1", lw=1.0, ls="--", zorder=3,
               label=f"Adam mean {L_final*1e4:.2f}")
    ax.fill_between(ep[m], (L_final - sd_tail) * 1e4, (L_final + sd_tail) * 1e4,
                    color="#1565C0", alpha=.22, lw=0, zorder=1,
                    label=r"$\pm1$ s.d.")
    ax.plot([n_adam], [lbfgs * 1e4], "D", ms=3.6, color="#2E7D32", zorder=4,
            label=f"L-BFGS {lbfgs*1e4:.2f}")
    ax.set_xlim(ep[m][0], n_adam + 10)
    ax.set_xlabel("Adam epoch / L-BFGS")
    ax.set_ylabel(r"$\mathcal{L}$  ($10^{-4}$)")
    ax.set_title("(b) last 500 epochs, linear", loc="left")
    ax.legend(loc="lower left", frameon=True, framealpha=.9, edgecolor="none",
              fontsize=8.0)
    ax.grid(alpha=.3)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{RES}/fig5_pinn_loss_en.png", dpi=300); plt.close(fig)
    print(f"[fig4] adam {n_adam} zero {ep_zero} settle {ep_settle} "
          f"plateau {L_final:.4e} sd {sd_tail:.2e} lbfgs {lbfgs:.4e}")

    # ---------- Fig 5 (displacement field) ----------
    # (a) is now a single overlay (PINN colours with the exact solution as white
    # contours) instead of two visually identical maps; the freed panel widens the
    # centreline and gives the error field room to show its asymmetry.
    ng = 61
    xg = np.linspace(0, A, ng); yg = np.linspace(0, B, ng)
    XG, YG = np.meshgrid(xg, yg, indexing="ij")
    WE = S.w_analytic(XG.ravel(), YG.ravel(), q).reshape(ng, ng)
    WP = np.array(S.pinn_field(pinn, np.stack([XG.ravel(), YG.ravel()], axis=1), q)[0]).reshape(ng, ng)
    ERR = np.abs(WP - WE)
    i_pk, j_pk = np.unravel_index(ERR.argmax(), ERR.shape)
    fig, axes = plt.subplots(1, 3, figsize=(6.1, 2.50),
                             gridspec_kw={"width_ratios": [1, 1, 1.22]})
    ax = axes[0]
    im = ax.contourf(xg, yg, WP.T, levels=24, cmap="viridis")
    ax.contour(xg, yg, WE.T, levels=9, colors="white", linewidths=0.45)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055,
                      pad=0.20, aspect=11)
    cb.locator = mticker.MaxNLocator(4); cb.update_ticks()
    cb.ax.tick_params(labelsize=8.0)
    ax.set_title("(a) $\\hat w$ (colour) and $w^*$ (white)", loc="left", fontsize=8.0)
    ax = axes[1]
    im = ax.contourf(xg, yg, ERR.T * 1e6, levels=24, cmap="magma")
    ax.plot(xg[i_pk], yg[j_pk], "x", color="white", ms=4.2, mew=1.0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055,
                      pad=0.20, aspect=11)
    cb.locator = mticker.MaxNLocator(4); cb.update_ticks()
    cb.ax.tick_params(labelsize=8.0)
    ax.set_title("(b) $|\\hat w-w^*|$  ($10^{-6}$)", loc="left", fontsize=8.0)
    for ax in axes[:2]:
        ax.set_xlabel("$x$", labelpad=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=8.0)
    axes[0].set_ylabel("$y$")
    axes[1].set_yticklabels([])
    ax = axes[2]
    yc = ng // 2
    ax.plot(xg, WE[:, yc] * 1e3, "k-", lw=1.6, label="exact")
    ax.plot(xg, WP[:, yc] * 1e3, "--", lw=1.4, color="#C0392B", label="PINN field")
    m = np.isclose(Ynr, 0.5)
    ax.plot(Xnr[m], d_ex.reshape(-1, 4)[:, 0][m] * 1e3, "s", ms=3.4, color="#1565C0",
            label="FE nodes (exact)")
    ax.set_xlabel("$x$"); ax.set_ylabel("$w$  ($10^{-3}$)", labelpad=2)
    ax.set_title("(c) centreline $y=0.5$", loc="left", fontsize=8.0)
    ax.set_ylim(-0.9, 6.4)
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=8.0,
              handlelength=1.4, borderpad=0.2, labelspacing=0.35)
    ax.grid(alpha=.3)
    fig.subplots_adjust(left=0.078, right=0.975, top=0.90, bottom=0.095, wspace=0.22)
    fig.savefig(f"{RES}/fig6_displacement_field_en.png", dpi=300); plt.close(fig)
    print(f"[fig5] error peak ({xg[i_pk]:.3f},{yg[j_pk]:.3f}) "
          f"max {ERR.max():.3e} rel {ERR.max()/np.abs(WE).max():.3e}")

    # ---------- Fig 8 (recovery along y=0.5): absolute + residual ----------
    XR, YR = Fop["X"].ravel(), Fop["Y"].ravel()
    uy = np.unique(np.round(YR, 6))
    y0 = uy[np.argmin(np.abs(uy - 0.5))]
    ln = np.isclose(np.round(YR, 6), y0)
    od = np.argsort(XR[ln]); XL = XR[ln][od]
    w_t = S.w_analytic(XR, YR, q); kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, q)
    mx_t = D * (kx_t + NU * ky_t); my_t = D * (ky_t + NU * kx_t)
    w_star_g, x_un = S.wstar_from_field(Fop, npts=15)
    mid_u = int(np.argmin(np.abs(x_un - 0.5)))
    w_star_line = w_star_g[:, mid_u]
    Fd = Fop; Bd = Fbf
    items = [("$w$", w_t, Bd["W"].ravel(), Fd["W"].ravel(), None),
             (r"$M_x$", mx_t, D * (Bd["KX"].ravel() + NU * Bd["KY"].ravel()),
              D * (Fd["KX"].ravel() + NU * Fd["KY"].ravel()), None),
             (r"$M_y$", my_t, D * (Bd["KY"].ravel() + NU * Bd["KX"].ravel()),
              D * (Fd["KY"].ravel() + NU * Fd["KX"].ravel()), None)]
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 4.1),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    def _smooth(v, w=9):
        """centred moving average, edge-safe (keeps the residual bands readable)"""
        v = np.asarray(v, dtype=float)
        if v.size < w:
            return v
        k = np.ones(w) / w
        pad = w // 2
        vp = np.pad(v, pad, mode="edge")
        return np.convolve(vp, k, mode="valid")[:v.size]

    for j, (tt, ye, yb, yo, extra) in enumerate(items):
        ax = axes[0, j]
        ax.plot(XL, ye[ln][od], "k-", lw=1.5, label="exact")
        ax.plot(XL, yb[ln][od], "b--", lw=0.9, marker="s", ms=2.2, markevery=8,
                label="BFS interpolant")
        ax.plot(XL, yo[ln][od], "r-.", lw=1.0, marker="o", ms=1.8, markevery=8,
                label=("$w_{\\mathrm{reg}}$ (regression head)" if j == 0 else "operator"))
        if j == 0:
            ax.plot(x_un, w_star_line, "g-", lw=1.5,
                    label="$w_{\\mathrm{dec}}$ (physics decoder)")
        ax.set_title("(" + "abc"[j] + ") " + tt, loc="left", fontsize=8.0)
        ax.grid(alpha=.3)
        ax.tick_params(labelbottom=False)
        if j == 0:
            ax.set_ylabel("field")
        ax = axes[1, j]
        pk = np.max(np.abs(ye[ln]))
        ax.axhline(0, color="#888", lw=0.6, ls="--")
        rb = (yb[ln][od] - ye[ln][od]) / pk * 100
        ro = (yo[ln][od] - ye[ln][od]) / pk * 100
        # faint raw samples + smoothed trend, so the ordering of the two surrogates is visible
        ax.plot(XL, rb, "b--", lw=0.5, alpha=0.35)
        ax.plot(XL, ro, "r-.", lw=0.5, alpha=0.35)
        ax.plot(XL, _smooth(rb), "b-", lw=1.3, label="BFS interpolant")
        ax.plot(XL, _smooth(ro), "r-", lw=1.3, label="operator")
        ax.set_title("(" + "def"[j] + ") residual", loc="left", fontsize=8.0)
        ax.set_xlabel("$x$"); ax.grid(alpha=.3)
        if j == 0:
            ax.set_ylabel("residual (% of peak)")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.4)
    fig.savefig(f"{RES}/fig7_field_recovery_en.png", dpi=300); plt.close(fig)

    # ---------- Fig 7 (six-term operator loss), two panels ----------
    cols = {"L1": "#1565C0", "L2": "#00838F", "L3": "#2E7D32",
            "L4": "#6A1B9A", "L5": "#C62828", "L6": "#E65100"}
    names = {"L1": "L1 disp.", "L2": "L2 curv.", "L3": "L3 corner",
             "L4": "L4 stress", "L5": "L5 stress\u2013curv.",
             "L6": "L6 curv.\u2013disp."}
    k = 51
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.72))
    ratios = {}
    for kk in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        v = np.asarray(hist_op[kk], float)
        sm = np.convolve(v, np.ones(k) / k, mode="valid")
        ratios[kk] = float(sm[0] / max(sm[-1], 1e-30))
        axes[0].plot(np.arange(len(v)), np.abs(v) + 1e-18, color=cols[kk],
                     lw=0.5, alpha=.20)
        axes[0].semilogy(np.arange(k - 1, len(v)), sm, color=cols[kk], lw=1.1,
                         label=names[kk])
        axes[1].semilogy(np.arange(k - 1, len(v)), sm / sm[0], color=cols[kk], lw=1.1,
                         label=f"{names[kk]}  ({ratios[kk]:.0f}\u00d7)")
    axes[0].set_title("(a) absolute", loc="left")
    axes[1].set_title("(b) normalised by own epoch-50 value", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("loss / initial")
    h0, l0 = axes[1].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=8.0, columnspacing=1.2,
               handlelength=1.7)
    fig.subplots_adjust(left=0.097, right=0.992, top=0.885, bottom=0.315, wspace=0.26)
    fig.savefig(f"{RES}/fig8_operator_loss_en.png", dpi=300); plt.close(fig)
    print("[fig7] reduction factors:", {k_: round(v_, 1) for k_, v_ in ratios.items()})

    # ---------- Fig 13 (generalisation over 30 unseen loads) ----------
    # Mean +/- std are drawn as markers with error bars, NOT as bars: a bar starts at
    # zero, and on a logarithmic axis zero is at minus infinity, so every bar is clipped
    # to the axis frame and all bars come out identical (this is what V8 shipped).
    gm = R["generalization"]
    pe = gm.get("percase_exact")
    g30 = R.get("supp_review", {}).get("gen30_pinn_dof", {})
    pp = g30.get("percase_pinn")
    ok = bool(pe and pp and len(pp["w_star"]) == 30 and len(pe["w_star"]) == 30)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    c1, c2 = "#1565C0", "#E65100"
    lo, hi = 1e9, 0.0
    if not ok:
        labels = [r"$w^{\ast}$", r"$\kappa$", r"$\sigma$"]
        means = np.array([gm["w_star_mean"], gm["k_mean"], gm["s_mean"]], float)
        stds = np.array([gm["w_star_std"], gm["k_std"], gm["s_std"]], float)
        xb = np.arange(3, dtype=float)
        ax.errorbar(xb, means, yerr=stds, fmt="o", ms=5.5, lw=1.3, capsize=3,
                    color=c1, label="exact DOF")
        for xi, m_, s_ in zip(xb, means, stds):
            ax.annotate(f"{m_:.1e} (CV {s_/m_*100:.0f}%)", (xi, m_),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.0)
        lo, hi = (means - stds).min(), (means + stds).max()
        ax.set_xticks(xb); ax.set_xticklabels(labels)
        ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    else:
        ex = dict(w_star=np.array(pe["w_star"]), k=np.array(pe["k"]), s=np.array(pe["s"]))
        e2e = dict(w_star=np.array(pp["w_star"]), k=np.array(pp["k"]), s=np.array(pp["s"]))
        groups = [(r"$w^{\ast}$ (decoded)", ex["w_star"], e2e["w_star"]),
                  (r"$\kappa$", ex["k"], e2e["k"]),
                  (r"$\sigma$", ex["s"], e2e["s"])]
        xb = np.arange(3, dtype=float); off = 0.17
        for i, (name, m1, m2) in enumerate(groups):
            ax.errorbar(xb[i] - off, m1.mean(), yerr=m1.std(), fmt="o", ms=5.5,
                        lw=1.3, capsize=2.6, color=c1,
                        label="exact DOF" if i == 0 else None)
            ax.annotate(f"{m1.mean():.1e}\nCV {m1.std()/m1.mean()*100:.0f}%",
                        (xb[i] - off, m1.mean()), textcoords="offset points",
                        xytext=(-2, -5), ha="center", va="top", fontsize=8.0,
                        bbox=dict(fc="white", ec="none", pad=0.8))
            ax.errorbar(xb[i] + off, m2.mean(), yerr=m2.std(), fmt="s", ms=5.5,
                        lw=1.3, capsize=2.6, color=c2,
                        label="end-to-end (PINN DOF)" if i == 0 else None)
            ax.annotate(f"{m2.mean():.1e}\nCV {m2.std()/m2.mean()*100:.0f}%",
                        (xb[i] + off, m2.mean()), textcoords="offset points",
                        xytext=(2, 8), ha="center", va="bottom", fontsize=8.0,
                        bbox=dict(fc="white", ec="none", pad=0.8))
            lo = min(lo, m1.mean() - m1.std(), m2.mean() - m2.std())
            hi = max(hi, m1.mean() + m1.std(), m2.mean() + m2.std())
        ax.set_xticks(xb); ax.set_xticklabels([g[0] for g in groups])
        ax.legend(frameon=False, fontsize=8.0, loc="lower right", ncol=1)
    ax.set_ylim(lo * 0.62, hi * 1.55)
    ax.set_xlim(-0.62, 2.62)      # room for the left-most value label
    ax.set_ylabel("relative $L_2$ error (log)"); ax.set_yscale("log")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{RES}/fig9_generalization_en.png", dpi=300); plt.close(fig)

    # ---------- Fig 10 (stress residual and map-to-map difference) ----------
    # A parity plot cannot separate two maps that lie on the same diagonal: the two
    # point clouds cover each other.  Plotting the pointwise residual against the
    # local stress level (a) and the difference between the two maps (b) shows what
    # the parity plot hid, and states the parity numerically.
    Xr = Fop["X"].ravel(); Yr = Fop["Y"].ravel()
    sx_t, sy_t, sxy_t = S.stress_analytic(Xr, Yr, 1.0, q)
    vm_t = von_mises(sx_t, sy_t, sxy_t)
    vm_op = von_mises(Fop["SX"].ravel(), Fop["SY"].ravel(), Fop["SXY"].ravel())
    vm_bf = von_mises(Fbf["SX"].ravel(), Fbf["SY"].ravel(), Fbf["SXY"].ravel())
    sp = vm_t / vm_t.max()
    keep = sp > 0.08
    # residuals in % of the peak top-surface stress, the same convention as the
    # field-recovery residuals of Fig. 8 (a pointwise relative residual is dominated
    # by the near-zero-stress corner and says more about the metric than the map)
    pk_vm = float(vm_t.max())
    r_bf = (vm_bf - vm_t) / pk_vm * 100.0
    r_op = (vm_op - vm_t) / pk_vm * 100.0
    d_ob = (vm_op - vm_bf) / pk_vm * 100.0
    ledges = np.linspace(np.log10(0.08), 0.0, 9)
    cen = 10 ** (0.5 * (ledges[1:] + ledges[:-1]))
    bidx = np.clip(np.digitize(np.log10(sp[keep]), ledges) - 1, 0, len(cen) - 1)

    def bands(v):
        med, lo, hi = [], [], []
        for i in range(len(cen)):
            s = v[bidx == i]
            if s.size < 20:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                med.append(np.median(s))
                lo.append(np.percentile(s, 16)); hi.append(np.percentile(s, 84))
        return np.array(med), np.array(lo), np.array(hi)

    C_BF, C_OP = "#1565C0", "#C0392B"
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.62), sharex=True)
    rng = np.random.default_rng(0)
    sub = rng.choice(np.count_nonzero(keep), size=min(900, np.count_nonzero(keep)),
                     replace=False)
    spk = sp[keep][sub]
    ax = axes[0]
    for v, c, mk, lb in ((r_bf[keep], C_BF, "s", "BFS interpolant"),
                         (r_op[keep], C_OP, "o", "neural operator")):
        ax.plot(spk, v[sub], mk, ms=1.4, color=c, alpha=0.22, mew=0)
        md, lo, hi = bands(v)
        ax.fill_between(cen, lo, hi, color=c, alpha=0.16, lw=0)
        ax.plot(cen, md, "-", color=c, lw=1.5, marker=mk, ms=2.6, label=lb)
    ax.axhline(0, color="#555", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("exact $\\sigma_{vm}^{*}$ / peak")
    ax.set_ylabel("residual (% of peak)")
    ax.set_title("(a) each map against the exact field", loc="left", fontsize=8.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8.0, handlelength=1.8)
    ax.set_ylim(-9, 9)
    ax.text(0.03, 0.05, f"RMS residual\nBFS {np.sqrt((r_bf[keep] ** 2).mean()):.2f}%,  "
                        f"operator {np.sqrt((r_op[keep] ** 2).mean()):.2f}%",
            transform=ax.transAxes, fontsize=8.0, va="bottom", color="#333",
            linespacing=1.45)
    ax = axes[1]
    ax.plot(spk, d_ob[keep][sub], "o", ms=1.4, color="#6A1B9A", alpha=0.25, mew=0)
    md, lo, hi = bands(d_ob[keep])
    ax.fill_between(cen, lo, hi, color="#6A1B9A", alpha=0.16, lw=0)
    ax.plot(cen, md, "-", color="#6A1B9A", lw=1.5, marker="o", ms=2.6,
            label="operator $-$ BFS")
    ax.axhline(0, color="#555", lw=0.8, ls="--")
    rms_d = float(np.sqrt((d_ob[keep] ** 2).mean()))
    ax.axhspan(-rms_d, rms_d, color="#6A1B9A", alpha=0.07, lw=0)
    ax.text(0.03, 0.05, f"RMS difference {rms_d:.1f}%\nof the peak stress",
            transform=ax.transAxes, fontsize=8.0, va="bottom", color="#333",
            linespacing=1.45)
    ax.set_xscale("log")
    ax.set_xlabel("exact $\\sigma_{vm}^{*}$ / peak")
    ax.set_ylabel("operator $-$ BFS (% of peak)")
    ax.set_title("(b) the two maps, fed identical DOFs", loc="left", fontsize=8.0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.0, handlelength=1.8)
    ax.set_ylim(-9, 9)
    for ax in axes:
        ax.grid(alpha=.28)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(f"{RES}/fig10_stress_parity_en.png", dpi=300); plt.close(fig)
    json.dump({"rms_bf": float(np.sqrt((r_bf[keep] ** 2).mean())),
               "rms_op": float(np.sqrt((r_op[keep] ** 2).mean())),
               "rms_d": rms_d, "mean_d": float(d_ob[keep].mean()),
               "pk_vm": pk_vm},
              open(os.path.join(RES, "v13_fig10.json"), "w", encoding="utf-8"), indent=1)
    print(f"[fig10] median|r| bfs {np.median(np.abs(r_bf[keep])):.2f}% "
          f"op {np.median(np.abs(r_op[keep])):.2f}%  mean op-bfs {d_ob[keep].mean():+.3f}% "
          f"rms {rms_d:.2f}%")

    # ---------- Fig 16 (where the decoder pays; the data of Table 8) ----------
    vd = json.load(open(os.path.join(RES, "v10_diag.json"), encoding="utf-8"))["fam"]
    rows = [("ss_fe", "Simply supported,\nuniform load", "right"),
            ("patch", "Simply supported,\ncentre patch", "right"),
            ("clamp", "Clamped,\nuniform load", "left")]
    fig, ax = plt.subplots(figsize=(6.1, 2.22))
    ypos = np.arange(len(rows))[::-1]
    for y, (key, lab, side) in zip(ypos, rows):
        b = vd[key]["bfs_w"]; o = vd[key]["op_w"]; r = o / b
        ax.plot([b, o], [y, y], "-", color="#9E9E9E", lw=1.4, zorder=1)
        ax.plot([b], [y], "s", ms=5.6, color="#1565C0", zorder=3,
                label="BFS interpolant" if y == ypos[0] else None)
        ax.plot([o], [y], "o", ms=5.6, color="#C0392B", zorder=3,
                label="operator + decoder" if y == ypos[0] else None)
        if r > 1:
            txt, col = f"BFS {r:.1f}$\\times$ better", "#1565C0"
        else:
            txt, col = f"decoder {1 / r:.1f}$\\times$ better", "#A93226"
        if side == "right":
            ax.annotate(txt, (max(b, o), y), textcoords="offset points",
                        xytext=(9, 0), ha="left", va="center", fontsize=8.0, color=col)
        else:
            ax.annotate(txt, (min(b, o), y), textcoords="offset points",
                        xytext=(-9, 0), ha="right", va="center", fontsize=8.0, color=col)
    ax.set_xscale("log")
    ax.set_xlim(1.02e-3, 5.6e-2)
    ax.set_xticks([1e-3, 3e-3, 1e-2, 3e-2])
    ax.set_xticklabels(["$10^{-3}$", "$3\\times10^{-3}$", "$10^{-2}$", "$3\\times10^{-2}$"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(-0.62, 2.62)
    ax.set_yticks(ypos); ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=8.0)
    ax.set_xlabel("mean relative $L_2$ displacement error (log, 16$\\times$16 FE reference)")
    ax.grid(alpha=.28, axis="x")
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper left", bbox_to_anchor=(0.0, 0.995), ncol=2,
               frameon=False, fontsize=8.0, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.178, right=0.995, top=0.795, bottom=0.215)
    fig.savefig(f"{RES}/fig18_decoder_floor_en.png", dpi=300); plt.close(fig)
    print("[fig16] decoder-vs-floor", [(k, round(vd[k]["bfs_w"], 6), round(vd[k]["op_w"], 6))
                                       for k, _, _ in rows])

    # ---------- Fig 9 (von Mises field), horizontal colorbars ----------
    n1 = Fop["W"].shape[0]
    VM_T = vm_t.reshape(n1, n1); VM_O = vm_op.reshape(n1, n1)
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.55))
    for j, (ax, Z, tt, cm) in enumerate([(axes[0], VM_T, "(a) exact", "plasma"),
                                         (axes[1], VM_O, "(b) operator", "plasma"),
                                         (axes[2], np.abs(VM_O - VM_T) * 1e3,
                                          "(c) $|$error$|$ ($10^{-3}$)", "viridis")]):
        im = ax.contourf(Fop["X"], Fop["Y"], Z, levels=26, cmap=cm)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06,
                          pad=0.19, aspect=14)
        cb.locator = mticker.MaxNLocator(4)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=8.0)
        ax.set_title(tt, loc="left", fontsize=8.0)
        ax.set_xlabel("$x$", labelpad=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=8.0)
        if j == 0:
            ax.set_ylabel("$y$")
        else:
            ax.set_yticklabels([])
    fig.subplots_adjust(left=0.078, right=0.99, top=0.895, bottom=0.21, wspace=0.10)
    fig.savefig(f"{RES}/fig11_mises_field_en.png", dpi=300); plt.close(fig)

    print("[plot_en] figures written to", RES)


if __name__ == "__main__":
    main()
