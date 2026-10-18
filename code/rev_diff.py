# -*- coding: utf-8 -*-
"""
rev_diff.py — Differentiable-design evidence for the TWS manuscript (Section 4.10).

Three measurements, all on the *reference* trained operator (results/op_state.pt):

  (A) Jacobian check. The 7 x 16 Jacobian of the element outputs
      {w, kxx, kyy, kxy, sxx, syy, sxy} at a fixed intra-element point with respect to
      the 16 element DOFs, computed by reverse-mode autograd, is compared with
      central finite differences of the same operator.

  (B) Cost of one gradient. Wall-clock of (i) a forward evaluation, (ii) a backward
      pass returning the gradient of a scalar element objective w.r.t. all 16 DOFs,
      and (iii) one central-difference gradient (2 x 16 forward evaluations).
      Reported for the element-level objective (16 design variables) and for the
      panel-level objective (all nodal DOFs of the 4 x 4 working grid).

  (C) Target-stress inverse design at the element level. Starting from a perturbed
      state, the element DOF vector that reproduces a prescribed top-surface von
      Mises distribution is recovered by gradient descent through the operator.
      Reported: iterations to a fixed tolerance, final state error, final field
      residual, and the forward-pass count a finite-difference descent would need.

Writes results/diff_design.json.
"""
import json
import os
import time

import numpy as np
import torch

import shell_system as S

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
torch.manual_seed(2026)

RES = S.RES
n = S.N_MESH                      # 4 x 4 working grid
Lex = S.A / n
NPTS = 11                         # intra-element sampling per direction


# ---------------------------------------------------------------- utilities
def load_op():
    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"),
                                  weights_only=False))
    op.eval()
    return op


def node_dof(q=1.0):
    """exact nodal DOF vector (4(n+1)^2,) at load amplitude q."""
    xs = np.linspace(0, S.A, n + 1)
    ys = np.linspace(0, S.B, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Xr, Yr = X.ravel(), Y.ravel()
    w0 = S.w_analytic(Xr, Yr, q)
    wx0, wy0 = S._w_grad(Xr, Yr, q)
    kxy0 = S.curv_analytic(Xr, Yr, q)[2]
    return np.stack([w0, wx0, wy0, -kxy0], axis=1).ravel()


def element_dof(dof, ei, ej):
    ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
           [(0, 0), (1, 0), (1, 1), (0, 1)]]
    return dof.reshape(-1, 4)[ids].ravel()


def sample_pts(npts=NPTS):
    g = np.linspace(-1, 1, npts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    return np.stack([XI.ravel(), ETA.ravel()], axis=1)


def mises(sxx, syy, sxy):
    return torch.sqrt(torch.clamp(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2,
                                  min=1e-30))


def element_outputs(op, d, xi, zeta=1.0):
    """d: (16,) tensor | xi: (N,2) tensor -> (N,7) tensor"""
    d = d.reshape(1, 16)
    pts = xi.reshape(-1, 2)
    wk = op.wk_at(d, pts)                                   # (N,4)
    zz = torch.full((pts.shape[0], 1), float(zeta))
    sp = op.s_at(d, torch.cat([pts, zz], dim=1))            # (N,3)
    return torch.cat([wk, sp], dim=1)


def panel_mises(op, dof, npts=NPTS):
    """peak top-surface von Mises over the assembled 4x4 panel (differentiable)"""
    xi = torch.tensor(sample_pts(npts))
    vals = []
    for ei in range(n):
        for ej in range(n):
            de = element_dof(dof, ei, ej)
            out = element_outputs(op, de, xi, 1.0)
            vals.append(mises(out[:, 4], out[:, 5], out[:, 6]))
    return torch.cat(vals).max()


# ---------------------------------------------------------------- (A) Jacobian
def jacobian_check(op):
    d0 = torch.tensor(element_dof(node_dof(1.0), 1, 1))
    xi = torch.tensor(np.array([[0.35, -0.25]]))            # one intra-element point
    zeta = 1.0

    def f(d):
        return element_outputs(op, d, xi, zeta).reshape(-1)   # (7,)

    J_ad = torch.autograd.functional.jacobian(f, d0)           # (7,16)

    # central differences with a relative step per DOF
    h = 1e-6 * torch.clamp(d0.abs(), min=1e-6)
    J_fd = torch.zeros_like(J_ad)
    for k in range(16):
        dp = d0.clone(); dp[k] += h[k]
        dm = d0.clone(); dm[k] -= h[k]
        J_fd[:, k] = (f(dp) - f(dm)) / (2.0 * h[k])

    scale = J_fd.abs().clamp(min=1e-12)
    rel = (J_ad - J_fd).abs() / scale
    return {
        "point": [float(xi[0, 0]), float(xi[0, 1]), zeta],
        "n_outputs": 7, "n_inputs": 16,
        "max_rel_dev": float(rel.max()),
        "median_rel_dev": float(rel.median()),
        "jac_fd_absmax": float(J_fd.abs().max()),
        "jac_ad_absmax": float(J_ad.abs().max()),
    }


# ---------------------------------------------------------------- (B) cost
def _time(fn, reps=200):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def cost_probe(op):
    d0 = torch.tensor(element_dof(node_dof(1.0), 1, 1))
    xi = torch.tensor(sample_pts(NPTS))
    dof0 = torch.tensor(node_dof(1.0))

    def fwd_element():
        return element_outputs(op, d0, xi, 1.0)

    def scalar_element():
        out = element_outputs(op, d0, xi, 1.0)
        return mises(out[:, 4], out[:, 5], out[:, 6]).mean()

    def grad_element():
        d = d0.clone().requires_grad_(True)
        out = element_outputs(op, d, xi, 1.0)
        j = mises(out[:, 4], out[:, 5], out[:, 6]).mean()
        (g,) = torch.autograd.grad(j, d)
        return g

    def fwd_panel():
        return panel_mises(op, dof0, NPTS)

    def grad_panel():
        dof = dof0.clone().requires_grad_(True)
        j = panel_mises(op, dof, NPTS)
        (g,) = torch.autograd.grad(j, dof)
        return g

    t_fwd_el = _time(fwd_element)
    t_bwd_el = _time(grad_element)
    t_fwd_pn = _time(fwd_panel, reps=20)
    t_bwd_pn = _time(grad_panel, reps=20)
    g = grad_element()
    gp = grad_panel()
    return {
        "t_fwd_element_ms": 1e3 * t_fwd_el,
        "t_grad_element_ms": 1e3 * t_bwd_el,
        "t_fwd_panel_ms": 1e3 * t_fwd_pn,
        "t_grad_panel_ms": 1e3 * t_bwd_pn,
        "n_design_element": 16,
        "n_design_panel": int(dof0.numel()),
        "fd_forwards_element": 32,
        "fd_forwards_panel": 2 * int(dof0.numel()),
        "fd_cost_element_ms": 1e3 * 2 * 16 * t_fwd_el,
        "fd_cost_panel_ms": 1e3 * 2 * int(dof0.numel()) * t_fwd_pn,
        "speedup_element": (2 * 16 * t_fwd_el) / t_bwd_el,
        "speedup_panel": (2 * int(dof0.numel()) * t_fwd_pn) / t_bwd_pn,
        "grad_element_absmax": float(g.abs().max()),
        "grad_panel_absmax": float(gp.abs().max()),
        "grad_panel_nonzero_frac": float((gp.abs() > 1e-12).double().mean()),
    }


# ---------------------------------------------------------------- (C) inverse design
def inverse_design(op, iters=400, lr=0.02):
    """recover the element DOF state that reproduces a prescribed von Mises field"""
    xi = torch.tensor(sample_pts(NPTS))
    d_star = torch.tensor(element_dof(node_dof(1.3), 1, 1))      # target state
    with torch.no_grad():
        target = element_outputs(op, d_star, xi, 1.0)
        target_vm = mises(target[:, 4], target[:, 5], target[:, 6]).detach()

    rng = np.random.default_rng(7)
    d0 = torch.tensor(element_dof(node_dof(0.75), 3, 2))         # different element / load
    d = d0.clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)

    hist = []
    fwd_count = 0
    for it in range(iters):
        opt.zero_grad()
        out = element_outputs(op, d, xi, 1.0)
        pred = mises(out[:, 4], out[:, 5], out[:, 6])
        J = ((pred - target_vm) ** 2).mean() / (target_vm ** 2).mean()
        J.backward()
        fwd_count += 1
        opt.step()
        hist.append(float(J.item()))
        if hist[-1] < 1e-4:
            break
    n_it = len(hist)
    with torch.no_grad():
        out = element_outputs(op, d, xi, 1.0)
        pred = mises(out[:, 4], out[:, 5], out[:, 6])
        res = float((((pred - target_vm) ** 2).mean() / (target_vm ** 2).mean()).item())
        # relative L2 error of the recovered 16-DOF state against the target state
        er = float((torch.norm(d.detach() - d_star) / torch.norm(d_star)).item())
        vm_t = target_vm.detach().numpy().copy()
        vm_r = pred.detach().numpy().copy()
    return {
        "target_state_norm": float(torch.norm(d_star)),
        "init_state_rel_err": float((torch.norm(d0 - d_star) / torch.norm(d_star)).item()),
        "iters": n_it,
        "final_field_residual": res,
        "final_state_rel_err": er,
        "loss_first": hist[0],
        "loss_last": hist[-1],
        "backward_passes": n_it,
        "fd_forwards_equivalent": 2 * 16 * n_it,
        "history": hist,
        "vm_target": vm_t.tolist(),
        "vm_recovered": vm_r.tolist(),
    }


def main():
    op = load_op()
    out = {"config": {"grid": "%dx%d" % (n, n), "npts": NPTS,
                      "operator": "results/op_state.pt (reference 4x4 model)"}}
    out["jacobian"] = jacobian_check(op)
    out["cost"] = cost_probe(op)
    out["inverse"] = inverse_design(op)
    with open(os.path.join(RES, "diff_design.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("[OK] saved diff_design.json")


if __name__ == "__main__":
    main()
