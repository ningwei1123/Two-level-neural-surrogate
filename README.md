# Two-level neural surrogate for thin-walled plate components

Code and figure data for the manuscript

> **Two-level neural surrogate for thin-walled plate components: where physics
> decoding beats shape-function interpolation**
> Wei Ning, *Thin-Walled Structures* (submitted)

The repository holds the analysis code that produces every number in the paper,
the four drawing scripts that render the figures, and the numerical content of
each figure as plain-text tables so that any figure can be redrawn without
PyTorch.

---

## 1. What the method is

A Kirchhoff–Love plate is analysed as an assembly of element-scale components.

| Level | Object | What it does |
|---|---|---|
| I | energy-based PINN on the whole plate | returns the 16 nodal DOFs $\mathbf{d}_e$ of every element for a given load |
| II | element operator (DeepONet-style) | maps $\mathbf{d}_e$ to the intra-element fields $\{\hat w,\hat\kappa,\hat\sigma\}$ |

Level II replaces the classical BFS shape-function interpolation that is
normally used to recover curvature and stress from nodal displacements. The
paper's claim is that the operator pays off where the *derived* fields matter,
and that it is differentiable with respect to the design variables.

---

## 2. Layout

```
.
├── code/     analysis scripts, drawing scripts, data exporter
├── data/     numerical content of each figure, one plain-text table per figure
└── README.md
```

`code/` is deliberately flat: the modules import each other by plain module
name (`import shell_system as S`, `from rev_common import exact_dof_arr`), so
keeping them in one directory is what makes the scripts runnable as they are.

### Analysis code

| File | Role |
|---|---|
| `shell_system.py` | core library: `MLP`, `PlatePINN` (Level I), `PlateOperator` (Level II), Navier reference solution, BFS interpolation, loss terms, training loops. Running it as a script reproduces the main results file. |
| `rev_common.py` | shared helpers: analytic DOF vectors, FE reference fields, $L_2$ metrics, JSON I/O |
| `rect_plate.py` | aspect-ratio (rectangular-panel) study |
| `eval_op.py`, `eval_poisson.py`, `eval_symavg.py` | evaluation utilities: operator metrics, Poisson recovery step, symmetry averaging |
| `sym_aug.py`, `fe_extra.py` | D4 symmetry augmentation; extra FE reference machinery |
| `supp_experiments.py` | driver for the main experiment set (closed loop, generalisation, wall-clock) |
| `gen30_percase.py` | per-load errors over the 30 unseen loads |
| `rev_seed.py`, `rev_seed_stats.py` | six-seed repeats and their field-level statistics |
| `rev_b.py` | operator training / grid scaling |
| `rev_c.py`, `rev_aug.py`, `rev_h.py`, `rev_g.py` | operator variants: plain, symmetry-augmented, symmetry-averaged, interface study |
| `rev_a_fe.py` | operator retrained on FE-generated labels, independent FE validation |
| `rev_ablate.py` | term-by-term ablation of the six-term loss |
| `rev_d.py` | classical baselines (POD, supervised-only, …) |
| `rev_e.py`, `rev_f.py`, `rev_e2.py`, `transfer_ood.py` | family transfer: patch load, clamped edges, out-of-distribution |
| `rev_cross.py` | direct evaluation of a 4×4-trained operator on 6×6 and 8×8 grids |
| `rev_robust_real.py` | sensitivity to imprecise driving DOFs (measured fields) |
| `rev_diff.py` | differentiability evidence: Jacobian vs finite differences, gradient cost |
| `rev_scale2.py` | per-grid wall-clock of the FE and surrogate chains |
| `timing_stage_split.py` | stage-by-stage split of one surrogate call (Level I / element / Poisson) |
| `rev_sigma_lambda.py` | λ-scan, spatial stress statistics |
| `rev_n8.py` | 8×8 operator used by the rectangular-panel comparison |
| `rev_r5.py` | supplementary ablation / cost records |
| `fig11_attribution_data.py` | assembles the error-attribution numbers behind Fig. 11 |

### Drawing code

| File | Figures |
|---|---|
| `plot_en.py` | Figs. 1–10, 13, 16 |
| `diag_err_noise.py` | Figs. 11, 12 |
| `plot_rev_en.py` | Fig. 14 |
| `plot_diff.py` | Fig. 15 |
| `png_rgb.py` | post-processing: strip alpha, re-tag 300 dpi |
| `export_figure_data.py` | regenerates everything in `data/` (see §5) |

Note that the PNG file names are historical and do **not** follow the
manuscript order — manuscript Fig. 1 is drawn by
`fig4_system_pipeline_en.png`, and `FIGMAP` in `export_figure_data.py` holds the
authoritative mapping.

---

## 3. Environment

Measured on the workstation used for the paper:

| Package | Version |
|---|---|
| Python | 3.13.14 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.1 |
| Pillow | (any recent) |

```bash
pip install -r requirements.txt
```

All computations are float64 on CPU; no GPU is required.

---

## 4. Paths and data

Every script reads and writes `../results/` relative to its own location, i.e.
a `results/` folder next to `code/`:

```
.
├── code/
└── results/      <- created by you; holds JSON results and .pt checkpoints
```

```bash
mkdir -p results
```

The JSON result files and the trained checkpoints (`pinn_state.pt`,
`op_state.pt`, `hist_pinn.npy`, `hist_op.npz`, …) are **not** shipped in this
repository, because the checkpoints are large binaries and the JSON files are
regenerated by the scripts below. To rebuild them:

```bash
cd code
python shell_system.py          # Level I + Level II + closed loop + generalisation
python supp_experiments.py      # wall-clock and the main experiment set
python gen30_percase.py         # per-load errors on the 30 unseen loads
python rev_seed.py              # six seeds (one JSON record per seed)
python rev_seed_stats.py        # mean ± std over the seeds
python rev_diff.py              # differentiability evidence
python rev_scale2.py            # per-grid timings
python timing_stage_split.py    # stage split of one call
```

The remaining `rev_*.py` scripts each add one block of the supplementary
comparison and write into `results/results.json` under `supp_review.<name>`.

Training the two networks takes roughly 7 min (Level I, 8 000 Adam + 1 L-BFGS)
and 15–40 min (Level II) on an Intel Core i9-14900K at 8 threads.

---

## 5. Figure data

`data/figNN_<name>.txt` holds the numbers actually drawn in manuscript Fig. NN.
The files were produced by running the real drawing code and dumping the artist
tree, so they are a transcription of the figure, not a re-derivation:

```bash
# --src is a directory that holds shell_system.py and, next to it, a
#       results/ folder with results.json and the checkpoints
python export_figure_data.py --src /path/to/analysis/code --out ../data
```

Header lines start with `#`. Body lines are tab- or space-separated numeric
columns. Four block types occur:

| Header | Content |
|---|---|
| `# axN lineK` | `x`, `y` — one curve |
| `# axN errorbarK` | `x`, `y`, lower error, upper error |
| `# axN bar-containerK` | patch width, height, left edge, bottom edge |
| `# axN contourf` / `imshow` | grid: `X` row, `Y` column, then `Z` row-major |

Figures 1, 2, 3 and 6 are schematic panels (geometry, network blocks, arrows);
they carry no measured data and the corresponding files say so explicitly.

---

## 6. Figure – code – data map

| Fig. | Content | Drawn by | Data file |
|---|---|---|---|
| 1 | two-level surrogate architecture | `plot_en.main` | `fig01_surrogate_architecture.txt` (schematic) |
| 2 | benchmark plate and boundary conditions | `plot_en.main` | `fig02_plate_benchmark.txt` (schematic) |
| 3 | Level-I network and hard constraint | `plot_en.main` | `fig03_level1_network.txt` (schematic) |
| 4 | Level-I training history | `plot_en.main` | `fig04_level1_training_history.txt` |
| 5 | displacement field, PINN vs exact | `plot_en.main` | `fig05_displacement_field.txt` |
| 6 | Level-II element operator | `plot_en.main` | `fig06_level2_operator.txt` (schematic) |
| 7 | Level-II training losses | `plot_en.main` | `fig07_level2_training_history.txt` |
| 8 | intra-element recovery along $y=0.5$ | `plot_en.main` | `fig08_intra_element_recovery.txt` |
| 9 | von Mises stress on the top surface | `plot_en.main` | `fig09_mises_field.txt` |
| 10 | stress parity / residual by stress level | `plot_en.main` | `fig10_stress_parity.txt` |
| 11 | attribution of the closed-loop error | `diag_err_noise.plot_attribution` | `fig11_error_attribution.txt` |
| 12 | sensitivity to imprecise driving DOFs | `diag_err_noise.plot_noise` | `fig12_noise_robustness.txt` |
| 13 | generalisation over 30 unseen loads | `plot_en.main` | `fig13_generalisation.txt` |
| 14 | wall-clock against the working grid | `plot_rev_en.fig_scaling` | `fig14_wallclock_scaling.txt` |
| 15 | differentiability of the operator | `plot_diff` (module level) | `fig15_differentiability.txt` |
| 16 | where the decoder pays | `plot_en.main` | `fig16_decoder_floor.txt` |

---

## 7. Table – code map

| Table | Content | Produced by |
|---|---|---|
| 1 | hyper-parameters and loss weights | definitions in `shell_system.py` |
| 2 | closed-loop recovery, six seeds | `rev_seed.py` → `rev_seed_stats.py` |
| 3 | generalisation over 30 unseen loads | `gen30_percase.py` |
| 4 | per-load wall-clock, three grids | `rev_scale2.py`, `timing_stage_split.py` |
| 5 | transfer across load families and supports | `rev_e2.py`, `transfer_ood.py`, `rev_f.py` |
| 6 | reuse-or-retrain decision guide | derived from Tables 4–5 |
| 7 | cost of one design gradient | `rev_diff.py` |
| 8 | where the physics decoder pays | `diag_err_noise.py` |
| 9 | break-even query counts | derived from Table 4 and measured training times |
| A.1 | problem setting, data pipeline, randomness, evaluation | — (descriptive) |

---

## 8. Notes on reproducibility

* Every reported error is a relative $L_2$ norm over the field, computed by
  `l2rel` in `rev_common.py`, using the analytic Navier solution as reference.
* The main accuracy protocol uses 120 training loads with a hold-out set; the
  ablation, timing and transfer protocols use 140 loads without hold-out. The
  protocol in force is stated in each table caption of the paper.
* Timings are medians of repeated runs on a single workstation after warm-up.
  Absolute values scale with the CPU; only ratios between methods are meant to
  be transferred.
* Seeds are fixed explicitly in every script (`seed=2026` for the main run);
  the six-seed spread is reported in Table 2 rather than hidden.

---

## 9. Citation and contact

Please cite the article if you use this code. Correspondence:
Wei Ning, School of Mechanical Engineering, Shaanxi University of Technology —
`ningwei@snut.edu.cn`.
