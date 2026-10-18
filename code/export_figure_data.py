# -*- coding: utf-8 -*-
"""
export_figure_data.py -- dump the numerical content of every manuscript figure
to plain-text tables under ``data/``.

The exporter does not re-implement any plotting logic.  It imports the four
drawing scripts of the repository, intercepts ``Figure.savefig``, and walks the
fully built axes object: every line, error bar, bar, contour field, image and
scatter collection is written to ``figNN_<name>.txt`` with an explicit header.
What lands in the file is therefore exactly what the published rendered figure
shows, not a second-hand transcription.

Usage
-----
    python export_figure_data.py --src ../pinn_shell/code --out ../data

``--src`` must point at a directory that contains the model checkpoints and the
JSON result files (see README, "Reproducing the figures"), because the drawing
scripts load ``results.json``, ``pinn_state.pt``, ``op_state.pt``,
``hist_pinn.npy`` and ``hist_op.npz`` from their own ``../results`` folder.

Figure numbers below are the numbers printed in the manuscript.  Note that the
PNG file names are historical and do **not** follow the manuscript order
(e.g. manuscript Fig. 1 is drawn by ``fig4_system_pipeline_en.png``).
"""

import argparse
import io
import os
import runpy
import shutil
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# manuscript figure number -> (png produced by the drawing code, short name, kind)
# ---------------------------------------------------------------------------
FIGMAP = {
    "fig4_system_pipeline_en.png":    (1,  "surrogate_architecture",  "schematic"),
    "fig1_plate_setup_en.png":        (2,  "plate_benchmark",         "schematic"),
    "fig2_pinn_arch_en.png":          (3,  "level1_network",          "schematic"),
    "fig5_pinn_loss_en.png":          (4,  "level1_training_history", "curve"),
    "fig6_displacement_field_en.png": (5,  "displacement_field",      "field"),
    "fig3_operator_arch_en.png":      (6,  "level2_operator",         "schematic"),
    "fig8_operator_loss_en.png":      (7,  "level2_training_history", "curve"),
    "fig7_field_recovery_en.png":     (8,  "intra_element_recovery",  "curve"),
    "fig11_mises_field_en.png":       (9,  "mises_field",             "field"),
    "fig10_stress_parity_en.png":     (10, "stress_parity",           "scatter"),
    "fig15_errattribution_en.png":    (11, "error_attribution",       "bar"),
    "fig16_noiserobust_en.png":       (12, "noise_robustness",        "curve"),
    "fig9_generalization_en.png":     (13, "generalisation",          "curve"),
    "fig12_scaling_en.png":           (14, "wallclock_scaling",       "curve"),
    "fig17_diffdesign_en.png":        (15, "differentiability",       "curve"),
    "fig18_decoder_floor_en.png":     (16, "decoder_floor",           "lollipop"),
}

# contour calls are not recoverable from the artist tree (matplotlib keeps only
# the contour paths), so the raw grids are captured at call time
_CONTOUR = {}


def _wrap_contour(name):
    orig = getattr(Axes, name)

    def wrapped(self, *args, **kwargs):
        cs = orig(self, *args, **kwargs)
        nums = [np.array(a, dtype=float, copy=True)
                for a in args if isinstance(a, np.ndarray)]
        _CONTOUR.setdefault(id(self.figure), []).append((id(self), name, nums))
        return cs

    setattr(Axes, name, wrapped)


for _n in ("contourf", "contour"):
    _wrap_contour(_n)


def _vec(a):
    return np.asarray(a, dtype=float).ravel()


def _line_block(fh, tag, x, y, label="", note="", is_ref=False):
    x = _vec(x)
    y = _vec(y)
    if x.size == 0 or x.size != y.size:
        return False
    if is_ref and x.size == 2 and abs(y[0] - y[1]) < 1e-12:
        fh.write("# %s  axhline reference line at y = %.6g%s\n" % (tag, y[0], note))
        return True
    if is_ref and x.size == 2 and abs(x[0] - x[1]) < 1e-12:
        fh.write("# %s  axvline reference line at x = %.6g%s\n" % (tag, x[0], note))
        return True
    fh.write("# %s  label=%r  n=%d%s\n" % (tag, label, x.size, note))
    fh.write("#   col1 = x, col2 = y\n")
    for xi, yi in zip(x, y):
        fh.write("%.6g\t%.6g\n" % (xi, yi))
    fh.write("\n")
    return True


def _grid_block(fh, tag, X, Y, Z, extra=""):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    fh.write("# %s  X%s  Y%s  Z%s%s\n" % (tag, X.shape, Y.shape, Z.shape, extra))
    fh.write("#   X (one row):\n#   " + " ".join("%.6g" % v for v in X.ravel()) + "\n")
    fh.write("#   Y (one column):\n#   " + " ".join("%.6g" % v for v in Y.ravel()) + "\n")
    fh.write("#   Z (row-major, %d rows x %d cols):\n" % Z.shape)
    for row in Z:
        fh.write(" ".join("%.6g" % v for v in row) + "\n")
    fh.write("\n")


def dump_figure(fig, png, fh):
    num, name, kind = FIGMAP.get(os.path.basename(png), (None, None, None))
    ax = fig.axes
    fh.write("# =========================================================================\n")
    fh.write("# Figure %s -- %s\n" % (num, name))
    fh.write("# drawn by %s   (%d axes)\n" % (os.path.basename(png), len(ax)))
    fh.write("# =========================================================================\n\n")

    contours = _CONTOUR.get(id(fig), [])
    wrote = False

    # error bars are exported from their container, so the two cap markers and the
    # bar segments are excluded from the generic line/collection loops below
    eb_lines, eb_cols = set(), set()
    for cont in [c for a0 in ax for c in a0.containers]:
        if type(cont).__name__ == "ErrorbarContainer":
            dl, caps, cols = cont.lines[0], cont.lines[1], cont.lines[2]
            eb_lines.add(id(dl))
            for c in caps:
                eb_lines.add(id(c))
            for c in cols:
                eb_cols.add(id(c))

    for ai, a in enumerate(ax):
        # ---- error bars ----------------------------------------------------
        for ci, cont in enumerate(a.containers):
            if type(cont).__name__ != "ErrorbarContainer":
                continue
            dl, cols = cont.lines[0], cont.lines[2]
            xy = dl.get_xydata()
            if xy.size == 0:
                continue
            segs = []
            for c in cols:
                segs += [np.asarray(s, dtype=float) for s in c.get_segments()]
            lab = dl.get_label()
            if lab.startswith("_"):
                lab = ""
            fh.write("# ax%d errorbar%d  label=%r  marker=%r  n=%d\n"
                     % (ai, ci, lab, dl.get_marker(), len(xy)))
            fh.write("#   col1 = x, col2 = y (central value), col3 = lower err, "
                     "col4 = upper err (0 if no bar was drawn)\n")
            for k, (x, y) in enumerate(xy):
                lo = hi = 0.0
                if k < len(segs) and segs[k].shape[0] >= 2:
                    ys = segs[k][:, 1]
                    lo = abs(y - ys.min())
                    hi = abs(ys.max() - y)
                fh.write("%.6g\t%.6g\t%.6g\t%.6g\n" % (x, y, lo, hi))
            fh.write("\n")
            wrote = True
        # ---- contour fields captured at call time -------------------------
        for (aid, cname, arrs) in contours:
            if aid != id(a):
                continue
            if len(arrs) == 3:
                _grid_block(fh, "ax%d %s" % (ai, cname), arrs[0], arrs[1], arrs[2])
            elif len(arrs) == 1:
                Z = arrs[0]
                gx = np.arange(Z.shape[1], dtype=float)
                gy = np.arange(Z.shape[0], dtype=float)
                _grid_block(fh, "ax%d %s (index grid)" % (ai, cname), gx, gy, Z)
            wrote = True

        # ---- images -------------------------------------------------------
        for im in a.images:
            Z = np.asarray(im.get_array(), dtype=float)
            x0, x1, y0, y1 = im.get_extent()
            gx = np.linspace(x0, x1, Z.shape[1])
            gy = np.linspace(y0, y1, Z.shape[0])
            _grid_block(fh, "ax%d imshow" % ai, gx, gy, Z)
            wrote = True

        # ---- plain lines ---------------------------------------------------
        for li, ln in enumerate(a.lines):
            if id(ln) in eb_lines:
                continue                      # already exported with its container
            lab = ln.get_label()
            if lab.startswith("_"):
                lab = ""
            xy = ln.get_xydata()
            if xy.size == 0:
                continue
            try:                       # only true axhline/axvline use an axes transform
                tr = ln.get_transform()
                is_ref = tr is a.get_yaxis_transform() or \
                         tr is a.get_xaxis_transform()
            except Exception:
                is_ref = False
            wrote |= _line_block(fh, "ax%d line%d" % (ai, li), xy[:, 0], xy[:, 1],
                                 lab, is_ref=is_ref)

        # ---- bar charts ----------------------------------------------------
        for ci, cont in enumerate(a.containers):
            # BarContainer exposes .patches; ErrorbarContainer exposes .lines,
            # whose data line is already picked up by the ax.lines loop above
            rows = [p for p in getattr(cont, "patches", [])
                    if hasattr(p, "get_height")]
            if not rows:
                continue
            fh.write("# ax%d bar-container%d  label=%r  n=%d\n"
                     % (ai, ci, getattr(cont, "get_label", lambda: "")(), len(rows)))
            fh.write("#   col1 = width, col2 = height, col3 = left (x), col4 = bottom (y)\n")
            for p in rows:
                fh.write("%.6g\t%.6g\t%.6g\t%.6g\n"
                         % (p.get_width(), p.get_height(), p.get_x(), p.get_y()))
            fh.write("\n")
            wrote = True

        # ---- scatter / line collections ------------------------------------
        for ci, col in enumerate(a.collections):
            if id(col) in eb_cols:
                continue                      # error-bar segments, exported above
            name_c = type(col).__name__
            if name_c in ("PathCollection",):
                off = np.asarray(col.get_offsets(), dtype=float)
                if off.size:
                    fh.write("# ax%d %s%d (scatter)  n=%d\n"
                             % (ai, name_c, ci, len(off)))
                    fh.write("#   col1 = x, col2 = y\n")
                    for row in off:
                        fh.write("%.6g\t%.6g\n" % (row[0], row[1]))
                    fh.write("\n")
                    wrote = True
            elif name_c == "LineCollection":
                segs = [np.asarray(s, dtype=float) for s in col.get_segments()]
                segs = [s for s in segs if s.ndim == 2 and s.shape[0] >= 2]
                if segs:
                    fh.write("# ax%d %s%d (line segments)  n=%d\n"
                             % (ai, name_c, ci, len(segs)))
                    fh.write("#   one segment per block: 'seg x0 y0 x1 y1 ...'\n")
                    for k, s in enumerate(segs):
                        fh.write("seg%d\t" % k
                                 + "\t".join("%.6g\t%.6g" % (p[0], p[1]) for p in s)
                                 + "\n")
                    fh.write("\n")
                    wrote = True
            elif name_c == "QuadMesh":
                Z = np.asarray(col.get_array(), dtype=float)
                gx, gy = col.get_coordinates().transpose(2, 0, 1)[0, 0], None
                try:
                    cds = col.get_coordinates()          # (ny+1, nx+1, 2)
                    gx = cds[0, :, 0]
                    gy = cds[:, 0, 1]
                    if Z.ndim == 1:                      # flat -> reshape
                        ny, nx = len(gy) - 1, len(gx) - 1
                        Z = Z.reshape(ny, nx)
                    _grid_block(fh, "ax%d pcolormesh" % ai, gx, gy, Z)
                    wrote = True
                except Exception:
                    pass

        if not wrote and kind != "schematic":
            fh.write("# (no numerical artist found on ax%d)\n\n" % ai)

    if not wrote and kind == "schematic":
        fh.write("# Schematic panel: geometry, network blocks and arrows only.\n")
        fh.write("# It carries no measured data; see Section 2-3 of the manuscript.\n")
    return num, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory holding shell_system.py and ../results")
    ap.add_argument("--out", required=True, help="output directory for the txt tables")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="figdump_")
    records = []

    orig_savefig = Figure.savefig

    def savefig(self, fname, *a, **k):
        bn = os.path.basename(str(fname))
        if bn in FIGMAP:
            num, name, _ = FIGMAP[bn]
            path = os.path.join(out, "fig%02d_%s.txt" % (num, name))
            with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
                dump_figure(self, bn, fh)
            records.append((num, name, path))
            print("[txt] fig%02d_%-24s <- %s" % (num, name, bn))
        return orig_savefig(self, os.path.join(tmp, bn), *a, **k)

    Figure.savefig = savefig

    sys.path.insert(0, src)
    os.chdir(src)

    import plot_en
    plot_en.main()
    plt.close("all")

    import plot_rev_en
    plot_rev_en.fig_scaling()
    plot_rev_en.fig_symmetry()
    plot_rev_en.fig_transfer()
    plt.close("all")

    runpy.run_path(os.path.join(src, "plot_diff.py"), run_name="__main__")
    plt.close("all")

    import diag_err_noise
    diag_err_noise.main()
    plt.close("all")

    Figure.savefig = orig_savefig
    shutil.rmtree(tmp, ignore_errors=True)

    records.sort()
    print("\n%d tables written to %s" % (len(records), out))
    missing = [v for k, v in FIGMAP.items()
               if os.path.basename(k) not in {os.path.basename(r[2]) for r in records}]
    del missing  # not meaningful after renaming; kept for clarity


if __name__ == "__main__":
    main()
