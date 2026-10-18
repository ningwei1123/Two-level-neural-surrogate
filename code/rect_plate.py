# -*- coding: utf-8 -*-
"""
rect_plate.py — R3 #1 范围扩展：2:1 矩形板 (a=2, b=1) 闭环演示
==================================================================
目的：回应"仅单一几何/单一方板"质疑——
  · 几何：a/b=2 矩形板，16×8 网格（方单元 Le=1/8），SSSS 均布载；
  · 级I ：矩形域能量 PINN 重训（新 ansatz sin(πx/a)sin(πy/b)），网格无关；
  · 级II：直接复用"方板 8×8 训练"的单元算子 op_state_n8.pt（同 Le、同单元），
          演示单元级算子的跨几何复用（方板 64 单元 → 矩形 128 单元）；
  · 解码：矩形 Poisson（零 Dirichlet）物理解码；
  · 对照：BFS 同输入插值（与主稿完全一致的口径）。
产物：supp_review.aspect_ratio{meta, rect{op,bfs}, square8{op,bfs}, analytic_check}
用法：python rect_plate.py --ep-pinn 8000 --lbfgs 1200 [--op-n8 路径]
先决：results/op_state_n8.pt（方板 8×8、Le=1/8 训练、140 loads no-holdout 4000ep）
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import shell_system as S
import rev_b
from rev_common import load_json, save_json, l2rel

torch.set_default_dtype(torch.float64)

A_R, B_R = 2.0, 1.0          # 矩形板尺寸
N_TERM = 25


# ---------------------------------------------------------------------------
# 矩形 Navier 解析解（符号约定与 shell_system 一致：κxx=-w_xx, κxy=-w_xy）
# ---------------------------------------------------------------------------
def _rect_coefs(a, b, nterm=N_TERM):
    od = np.arange(1, 2 * nterm, 2).astype(np.float64)
    mm, nn = np.meshgrid(od, od, indexing="ij")
    al = (mm * np.pi / a).ravel(); be = (nn * np.pi / b).ravel()
    c = (16.0 / (np.pi ** 6 * S.D_FLEX * mm * nn
                 * (mm ** 2 / a ** 2 + nn ** 2 / b ** 2) ** 2)).ravel()
    return al, be, c, (al ** 2), (be ** 2), (al * be)


class RectNavier:
    """SSSS 均布载矩形板：w/一阶导/曲率κ/应力σ（与 shell_system 同符号约定）"""
    def __init__(self, a=A_R, b=B_R, nterm=N_TERM):
        self.a, self.b = a, b
        (self.al, self.be, self.c,
         self.a2, self.b2, self.ab) = _rect_coefs(a, b, nterm)

    def _tri(self, x, y):
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        sx = np.sin(self.al[:, None] * x[None, :]); cx = np.cos(self.al[:, None] * x[None, :])
        sy = np.sin(self.be[:, None] * y[None, :]); cy = np.cos(self.be[:, None] * y[None, :])
        return sx, sy, cx, cy

    def w(self, x, y, q):
        sx, sy, _, _ = self._tri(x, y)
        return q * ((sx * self.c[:, None]) * sy).sum(axis=0)

    def grad(self, x, y, q):
        sx, sy, cx, cy = self._tri(x, y)
        wx = q * ((cx * (self.c * self.al)[:, None]) * sy).sum(axis=0)
        wy = q * ((sx * (self.c * self.be)[:, None]) * cy).sum(axis=0)
        wxy = q * ((cx * (self.c * self.ab)[:, None]) * cy).sum(axis=0)
        return wx, wy, wxy

    def curv(self, x, y, q):
        sx, sy, cx, cy = self._tri(x, y)
        # κ = (-w_xx, -w_yy, -w_xy)
        kxx = q * ((sx * (self.c * self.a2)[:, None]) * sy).sum(axis=0)
        kyy = q * ((sx * (self.c * self.b2)[:, None]) * sy).sum(axis=0)
        kxy = -q * ((cx * (self.c * self.ab)[:, None]) * cy).sum(axis=0)
        return kxx, kyy, kxy

    def moment(self, x, y, q):
        kx, ky, kxy = self.curv(x, y, q)
        return (S.D_FLEX * (kx + S.NU * ky), S.D_FLEX * (ky + S.NU * kx),
                S.D_FLEX * (1 - S.NU) * kxy)

    def stress(self, x, y, zeta, q):
        mx, my, mxy = self.moment(x, y, q)
        z = np.atleast_1d(np.asarray(zeta, float))
        f = 6.0 * z / S.H_TH ** 2
        return f * mx, f * my, f * mxy

    def node_dof(self, xs, ys, q):
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Xr, Yr = X.ravel(), Y.ravel()
        w = self.w(Xr, Yr, q)
        wx, wy, wxy = self.grad(Xr, Yr, q)
        return np.stack([w, wx, wy, wxy], axis=1), Xr, Yr


# ---------------------------------------------------------------------------
# 矩形域能量 PINN（硬约束 sin(πx/a)sin(πy/b)，乘子网格无关可换）
# ---------------------------------------------------------------------------
class PlatePINN_Rect(nn.Module):
    def __init__(self, a=A_R, b=B_R, hidden=64, nlayer=4):
        super().__init__()
        self.a, self.b = a, b
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(nlayer - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        m = torch.sin(np.pi * x / self.a) * torch.sin(np.pi * y / self.b)
        return m * self.net(xy)


def _grad_hess(u, xy):
    g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    hx = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0]
    hy = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True)[0]
    return g, torch.stack([hx, hy], dim=1)


def _energy_rect(pinn, xy):
    u = pinn(xy).reshape(-1)
    _, H = _grad_hess(u, xy)
    uxx, uyy = H[:, 0, 0], H[:, 1, 1]
    uxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    bend = (uxx + uyy) ** 2 - 2.0 * (1.0 - S.NU) * (uxx * uyy - uxy ** 2)
    return 0.5 * S.D_FLEX * bend.mean() - u.mean()


def train_rect_pinn(pinn, a, b, n_epochs=8000, n_int=4000, lr=1e-3,
                    lbfgs_iter=1200, log_every=1000):
    opt = torch.optim.Adam(pinn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for it in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(n_int, 2)
        xy[:, 0] *= a; xy[:, 1] *= b
        xy.requires_grad_(True)
        loss = _energy_rect(pinn, xy)
        loss.backward()
        opt.step(); sched.step()
        if (it + 1) % log_every == 0:
            print(f"  [RPINN-Adam] {it+1:5d}  L={loss.item():.6e}", flush=True)
    xy_fix = torch.rand(max(n_int, 4096), 2)
    xy_fix[:, 0] *= a; xy_fix[:, 1] *= b
    xy_fix.requires_grad_(True)
    opt2 = torch.optim.LBFGS(pinn.parameters(), max_iter=lbfgs_iter,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-14, tolerance_change=1e-16)
    cnt = {"n": 0}
    def closure():
        opt2.zero_grad()
        loss = _energy_rect(pinn, xy_fix)
        loss.backward(); cnt["n"] += 1
        if cnt["n"] % 200 == 0:
            print(f"  [RPINN-LBFGS] {cnt['n']:5d}  L={loss.item():.6e}", flush=True)
        return loss
    opt2.step(closure)
    return _energy_rect(pinn, xy_fix).item()


def rect_pinn_dof(pinn, xs, ys, q):
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), requires_grad=True)
    u = pinn(xy).reshape(-1)
    g, H = _grad_hess(u, xy)
    uxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    out = [u.detach().numpy(), g[:, 0].detach().numpy(),
           g[:, 1].detach().numpy(), uxy.detach().numpy()]
    return np.stack([q * o for o in out], axis=1)   # (nnode,4)


def rect_fem_solve(na, nb, q, a=A_R, b=B_R):
    """矩形 BFS 参考求解（自检用）：返回节点自由度与坐标"""
    Lex, Ley = a / na, b / nb
    nnode = (na + 1) * (nb + 1); ndof = 4 * nnode
    K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
    Ke, fe = S.bfs_element(Lex, Ley, q)
    for ei in range(na):
        for ej in range(nb):
            ids = np.array([4 * ((ei + di) * (nb + 1) + (ej + dj)) + k
                            for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)]
                            for k in range(4)])
            K[np.ix_(ids, ids)] += Ke; F[ids] += fe
    fixed = set()
    for i in range(na + 1):
        for j in range(nb + 1):
            nid = i * (nb + 1) + j
            if i == 0 or i == na:
                fixed.update([4 * nid + 0, 4 * nid + 2])
            if j == 0 or j == nb:
                fixed.update([4 * nid + 0, 4 * nid + 1])
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)
    d = np.zeros(ndof)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return d


def assemble_rect(d_all, op, nav, na=16, nb=8, npts=15, src="op"):
    """矩形装配：op(算子) 或 bfs(同输入 BFS)。d_all 节点自由度(nnode*4)"""
    a, b = nav.a, nav.b
    Lex, Ley = a / na, b / nb
    n1 = npts
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    W = np.zeros((na * n1, nb * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W); XG = np.zeros_like(W); YG = np.zeros_like(W)
    for ei in range(na):
        for ej in range(nb):
            ids = [(ei + di) * (nb + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d_all.reshape(-1, 4)[ids].ravel()
            if src == "op":
                d_t = torch.tensor(de, dtype=torch.float64).reshape(1, 16)
                with torch.no_grad():
                    wk = op.wk_at(d_t, torch.tensor(pts, dtype=torch.float64)).numpy()
                    sp = op.s_at(d_t, torch.stack(
                        [torch.tensor(pts[:, 0], dtype=torch.float64),
                         torch.tensor(pts[:, 1], dtype=torch.float64),
                         torch.ones(pts.shape[0])], dim=1)).numpy()
                w, k, s = wk[:, 0], wk[:, 1:4], sp
            else:
                w, kxx, kyy, kxy = S.bfs_interp(de, pts[:, 0], pts[:, 1], Lex, Ley)
                mx = S.D_FLEX * (kxx + S.NU * kyy); my = S.D_FLEX * (kyy + S.NU * kxx)
                mxy = S.D_FLEX * (1 - S.NU) * kxy
                s = (6.0 / S.H_TH ** 2) * np.stack([mx, my, mxy], axis=1)
                k = np.stack([kxx, kyy, kxy], axis=1)
                w = np.asarray(w)
            sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = w.reshape(n1, n1)
            KX[sl_i, sl_j] = k[:, 0].reshape(n1, n1); KY[sl_i, sl_j] = k[:, 1].reshape(n1, n1)
            KXY[sl_i, sl_j] = k[:, 2].reshape(n1, n1)
            SX[sl_i, sl_j] = s[:, 0].reshape(n1, n1); SY[sl_i, sl_j] = s[:, 1].reshape(n1, n1)
            SXY[sl_i, sl_j] = s[:, 2].reshape(n1, n1)
            XG[sl_i, sl_j] = (Lex * (ei + 0.5 * (pts[:, 0] + 1.0))).reshape(n1, n1)
            YG[sl_i, sl_j] = (Ley * (ej + 0.5 * (pts[:, 1] + 1.0))).reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=XG, Y=YG)


def rect_coords(na, nb, npts):
    """装配场唯一网格坐标（不含重复边界）"""
    x = np.concatenate([np.linspace(e / na, (e + 1) / na, npts)[:-1]
                        for e in range(na)] + [np.array([1.0])]) * A_R
    y = np.concatenate([np.linspace(e / nb, (e + 1) / nb, npts)[:-1]
                        for e in range(nb)] + [np.array([1.0])]) * B_R
    return x, y


def poisson_rect(kx, ky, x, y):
    """∇²w = -(κxx+κyy)，Dirichlet 0，矩形非均匀?（均匀）网格"""
    Mx, My = kx.shape
    dx = x[1] - x[0]; dy = y[1] - y[0]
    N = (Mx - 2) * (My - 2)
    rows, cols, vals, rhs = [], [], [], []
    for i in range(1, Mx - 1):
        for j in range(1, My - 1):
            r = (i - 1) * (My - 2) + (j - 1)
            rows.append(r); cols.append(r); vals.append(-2 / dx ** 2 - 2 / dy ** 2)
            for di in (-1, 1):
                ii = i + di
                if 1 <= ii <= Mx - 2:
                    rows.append(r); cols.append((ii - 1) * (My - 2) + (j - 1))
                    vals.append(1 / dx ** 2)
            for dj in (-1, 1):
                jj = j + dj
                if 1 <= jj <= My - 2:
                    rows.append(r); cols.append((i - 1) * (My - 2) + (jj - 1))
                    vals.append(1 / dy ** 2)
            rhs.append(-(kx[i, j] + ky[i, j]))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    w = spla.spsolve(A, np.asarray(rhs))
    W = np.zeros((Mx, My)); W[1:Mx - 1, 1:My - 1] = w.reshape(Mx - 2, My - 2)
    return W


def rect_metrics(F, nav, q, npts):
    """全装配点场误差 + 唯一坐标网格解码 w*"""
    if "X" in F and "Y" in F:
        XR, YR = F["X"].ravel(), F["Y"].ravel()
    else:
        x = np.linspace(0, A_R, F["W"].shape[0]); y = np.linspace(0, B_R, F["W"].shape[1])
        XR, YR = np.meshgrid(x, y, indexing="ij")
        XR = XR.ravel(); YR = YR.ravel()
    kx_t, ky_t, kxy_t = nav.curv(XR, YR, q)
    sx_t, sy_t, sxy_t = nav.stress(XR, YR, 1.0, q)
    rec = dict(
        w=l2rel(F["W"].ravel(), nav.w(XR, YR, q)),
        k=l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                np.concatenate([kx_t, ky_t, kxy_t])),
        s=l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                np.concatenate([sx_t, sy_t, sxy_t])))
    # 唯一坐标网格（去重复边界）后解码 w*
    Mx_full, My_full = F["KX"].shape
    ix = np.concatenate([np.arange(e * npts, e * npts + npts - 1)
                         for e in range(Mx_full // npts)] + [[Mx_full - 1]])
    iy = np.concatenate([np.arange(e * npts, e * npts + npts - 1)
                         for e in range(My_full // npts)] + [[My_full - 1]])
    kxu = F["KX"][np.ix_(ix, iy)]; kyu = F["KY"][np.ix_(ix, iy)]
    xu = np.linspace(0, A_R, ix.size); yu = np.linspace(0, B_R, iy.size)
    w_star = poisson_rect(kxu, kyu, xu, yu)
    XU, YU = np.meshgrid(xu, yu, indexing="ij")
    rec["w_star"] = l2rel(w_star.ravel(), nav.w(XU.ravel(), YU.ravel(), q))
    return rec


def rect_dataset(nav, qs, na=16, nb=8, n_pts=10, seed=1):
    """矩形网格单元算子数据集（解析标签，约定与 make_operator_dataset 一致）。
    样本 = (de(16), pts(N,2)∈[-1,1]², zeta(N,), w_ref, k_ref(3), s_ref(3), q)"""
    xs = np.linspace(0, nav.a, na + 1); ys = np.linspace(0, nav.b, nb + 1)
    Lex, Ley = nav.a / na, nav.b / nb
    g = np.linspace(-1, 1, n_pts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    XI = XI.ravel(); ETA = ETA.ravel(); npts = XI.size
    PX = np.concatenate([xs[ei] + 0.5 * Lex * (XI + 1.0)
                         for ei in range(na) for _ in range(nb)])
    PY = np.concatenate([ys[ej] + 0.5 * Ley * (ETA + 1.0)
                         for _ in range(na) for ej in range(nb)])
    rng = np.random.default_rng(seed)
    samples = []
    for q in qs:
        dnode, _, _ = nav.node_dof(xs, ys, q)          # ((na+1)(nb+1),4)
        w_all = nav.w(PX, PY, q)
        kx_a, ky_a, kxy_a = nav.curv(PX, PY, q)
        zeta = rng.choice([-1.0, 1.0], size=PX.size)
        sx_a, sy_a, sxy_a = nav.stress(PX, PY, zeta, q)
        for c, (ei, ej) in enumerate([(i, j) for i in range(na) for j in range(nb)]):
            ids = [(ei + di) * (nb + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dnode[ids].ravel()
            sl = slice(c * npts, (c + 1) * npts)
            samples.append((de, np.stack([XI, ETA], axis=1), zeta[sl],
                            w_all[sl],
                            np.stack([kx_a[sl], ky_a[sl], kxy_a[sl]], axis=1),
                            np.stack([sx_a[sl], sy_a[sl], sxy_a[sl]], axis=1),
                            float(q)))
    return samples


def train_rect_op(samples, epochs=3000, threads=8, out_name="op_state_r16x8.pt",
                  save=True):
    torch.set_num_threads(threads)
    torch.manual_seed(2026); np.random.seed(2026)
    op = S.PlateOperator(p=64, hidden=64)
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=epochs, batch=24, Le=1.0 / 8.0,
                     scale=None, lam_w=1.0, log_every=1000)
    dt = time.time() - t0
    if save:
        torch.save(op.state_dict(), os.path.join(RES, out_name))
    print(f"[ROPN] trained {dt/60:.1f} min -> {out_name}", flush=True)
    return op, dt



def square8_metrics(op, nav_check=None, npts=15):
    """方板 8×8 同输入闭环（级I 2026 节点自由度 @8×8，同 op）——几何对照列"""
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    dp = rev_b.pinn_dof_grid(pinn, 1.0, 8)
    Fop = rev_b.assemble_op_grid(dp, op, 8, npts=npts)
    keep = rev_b.unique_idx(8, npts)
    kx = Fop["KX"][np.ix_(keep, keep)]; ky = Fop["KY"][np.ix_(keep, keep)]
    M = keep.size
    x = np.linspace(0.0, 1.0, M); h = x[1] - x[0]
    w_star = S.poisson_reconstruct_w(kx, ky, h)
    XU, YU = np.meshgrid(x, x, indexing="ij")
    XR, YR = Fop["X"].ravel(), Fop["Y"].ravel()
    kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, 1.0)
    sx_t, sy_t, sxy_t = S.stress_analytic(XR, YR, 1.0, 1.0)
    return dict(
        k=l2rel(np.concatenate([Fop["KX"].ravel(), Fop["KY"].ravel(), Fop["KXY"].ravel()]),
                np.concatenate([kx_t, ky_t, kxy_t])),
        s=l2rel(np.concatenate([Fop["SX"].ravel(), Fop["SY"].ravel(), Fop["SXY"].ravel()]),
                np.concatenate([sx_t, sy_t, sxy_t])),
        w_star=l2rel(w_star.ravel(), S.w_analytic(XU.ravel(), YU.ravel(), 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-pinn", type=int, default=8000)
    ap.add_argument("--lbfgs", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--op-n8", default=os.path.join(RES, "op_state_n8.pt"))
    ap.add_argument("--retrain-pinn", action="store_true", default=False)
    ap.add_argument("--rect-op-path", default=os.path.join(RES, "op_state_r16x8.pt"))
    ap.add_argument("--retrain-rect-op", action="store_true", default=False)
    ap.add_argument("--rect-op-epochs", type=int, default=3000)
    ap.add_argument("--rect-op-loads", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026); np.random.seed(2026)

    nav = RectNavier()
    # ---- 自检：矩形 FEM vs Navier（中心挠度/节点误差） ----
    na, nb = 16, 8
    xs = np.linspace(0, A_R, na + 1); ys = np.linspace(0, B_R, nb + 1)
    d_fem = rect_fem_solve(na, nb, 1.0)
    w_fem = d_fem.reshape(-1, 4)[:, 0]
    dnode, Xr, Yr = nav.node_dof(xs, ys, 1.0)
    e_fem = float(np.abs(w_fem - dnode[:, 0]).max())

    # ---- 级I：矩形能量 PINN ----
    pinn_path = os.path.join(RES, "pinn_rect_state.pt")
    pinn = PlatePINN_Rect(a=A_R, b=B_R)
    if os.path.exists(pinn_path) and not args.retrain_pinn:
        pinn.load_state_dict(torch.load(pinn_path, weights_only=False))
        print("[RPINN] loaded")
    else:
        print("===== 矩形级I 训练 =====", flush=True)
        t0 = time.time()
        train_rect_pinn(pinn, A_R, B_R, n_epochs=args.ep_pinn,
                        lbfgs_iter=args.lbfgs)
        torch.save(pinn.state_dict(), pinn_path)
        print(f"[RPINN] trained {(time.time()-t0)/60:.1f} min", flush=True)
    # 全场误差 + 节点 DOF
    ng = 101
    xg = np.linspace(0, A_R, ng); yg = np.linspace(0, B_R, ng)
    XG, YG = np.meshgrid(xg, yg, indexing="ij")
    xy = torch.tensor(np.stack([XG.ravel(), YG.ravel()], axis=1))
    w_pin = pinn(xy).detach().numpy().ravel()
    e_pin_l2 = float(l2rel(w_pin, nav.w(XG.ravel(), YG.ravel(), 1.0)))
    e_pin_max = float(np.abs(w_pin - nav.w(XG.ravel(), YG.ravel(), 1.0)).max())
    dp = rect_pinn_dof(pinn, xs, ys, 1.0).ravel()
    dnode, _, _ = nav.node_dof(xs, ys, 1.0)
    e_node = {k: float(np.abs(dp.reshape(-1, 4)[:, c] - dnode[:, c]).max())
              for k, c in [("w", 0), ("wx", 1), ("wy", 2), ("wxy", 3)]}

    # ---- 级II：几何族对照 ----
    # ① 方板训练算子 OOD 直测（元素同尺寸，但训练 DOF 分布为方板）——诚实报告族外退化
    op_sq = S.PlateOperator(p=64, hidden=64)
    op_sq.load_state_dict(torch.load(args.op_n8, weights_only=False)); op_sq.eval()
    npts = 15
    m_ood = rect_metrics(assemble_rect(dp, op_sq, nav, na=na, nb=nb,
                                       npts=npts, src="op"), nav, 1.0, npts)
    m_bf = rect_metrics(assemble_rect(dp, op_sq, nav, na=na, nb=nb,
                                      npts=npts, src="bfs"), nav, 1.0, npts)
    sq = square8_metrics(op_sq, npts=npts)   # 方板 8×8 对照列

    # ② 矩形族算子（解析标签、族内重训；等价于 4.7 的 FE 标签重训但无需 FE）
    if os.path.exists(args.rect_op_path) and not args.retrain_rect_op:
        op_r = S.PlateOperator(p=64, hidden=64)
        op_r.load_state_dict(torch.load(args.rect_op_path, weights_only=False))
        op_r.eval()
        print("[ROPN] loaded")
        tr_sec = None
    else:
        rngq = np.random.default_rng(3)
        qa = rngq.uniform(S.Q_LO, S.Q_HI, 400)
        qs_r = qa[(qa < 0.9) | (qa > 1.1)][:args.rect_op_loads]
        samples = rect_dataset(nav, qs_r, na=na, nb=nb, n_pts=10, seed=1)
        op_r, dt = train_rect_op(samples, epochs=args.rect_op_epochs,
                                 threads=args.threads)
        tr_sec = round(dt, 1)
    m_fam = rect_metrics(assemble_rect(dp, op_r, nav, na=na, nb=nb,
                                       npts=npts, src="op"), nav, 1.0, npts)

    out = {
        "meta": {"a": A_R, "b": B_R, "mesh": f"{na}x{nb}",
                 "element_Le": 1.0 / 8.0, "q": 1.0, "npts": npts,
                 "op_source": os.path.basename(args.op_n8),
                 "note": "2:1 rectangular plate closed loop. Level-I energy PINN "
                         "retrained for the rectangular domain (label-free). "
                         "op_ood = square-trained element operator applied "
                         "out-of-the-box (same element size, different DOF "
                         "distribution) — reported honestly as distributional "
                         "failure. op_family = same architecture retrained on "
                         "rectangular analytic labels (no FE involved). "
                         "BFS = same-input bicubic interpolation."},
        "analytic_check": {"fem_w_max_err": e_fem, "pinn_l2": e_pin_l2,
                           "pinn_max": e_pin_max, "node_err": e_node,
                           "w_center": float(nav.w(np.array([1.0]), np.array([0.5]), 1.0)[0])},
        "rect": {"bfs": m_bf, "op_ood": m_ood, "op_family": m_fam},
        "square8": sq,
        "rect_op_train_sec": tr_sec,
    }
    print("[rect] bfs k=%.2e w*=%.2e | op_ood k=%.2e | op_family k=%.2e s=%.2e w*=%.2e"
          % (m_bf["k"], m_bf["w_star"], m_ood["k"], m_fam["k"], m_fam["s"],
             m_fam["w_star"]))
    R = load_json()
    R.setdefault("supp_review", {})["aspect_ratio"] = out
    save_json(R)
    print("saved supp_review.aspect_ratio")


if __name__ == "__main__":
    main()
