# -*- coding: utf-8 -*-
"""
基于 PINN 与神经算子的板壳结构仿真替代体系 —— 四边简支矩形薄板数值验证
================================================================================
模型      : Kirchhoff–Love 薄板（平板壳元），矩形域 a×b，厚度 h，泊松比 ν
控制方程  : D ∇⁴ w = q(x,y)，  D = E h³ / [12(1-ν²)]
边界      : 四边简支  w = 0（本质边界，硬约束）; M_n = 0（自然边界，能量法自动满足）
参考解    : Navier 双正弦级数解（奇数项 m,n = 1,3,5,...）
参考 FEM  : 16 自由度双三次 Hermite 协调板单元（Bogner–Fox–Schmit, BFS）

级 I  : 能量型 PINN（Deep Ritz 型）求解节点自由度 {w, w_x, w_y, w_xy}
        —— 替代  K d = f
级 II : DeepONet 神经算子 由单元节点自由度恢复单元内
        位移 w / 曲率 (κ_xx, κ_yy, κ_xy) / 应力 (σ_xx, σ_yy, σ_xy)
        —— 替代  N、B 形函数后处理

无量纲化: a = b = 1, D = 1, h = 1, ν = 0.3；载荷幅值 q 为唯一参数

输出:
  results/fig*.png        —— 数值验证图
  results/results.json    —— 全部量化指标（供论文引用）
  results/error_report.txt
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# 无量纲参数
# ---------------------------------------------------------------------------
A, B = 1.0, 1.0          # 板边长
D_FLEX = 1.0             # 归一化抗弯刚度
NU = 0.3                 # 泊松比
H_TH = 1.0               # 归一化厚度
N_TERM = 25              # Navier 级数奇数项数（m,n = 1,3,...,2N-1）
Q_LO, Q_HI = 0.5, 2.0    # 载荷幅值范围
Q_TEST = 1.0             # 基准验证工况
N_MESH = 4               # 工作网格 n×n 单元

# ===========================================================================
# 0. Navier 解析解
# ===========================================================================
_ODD = np.arange(1, 2 * N_TERM, 2).astype(np.float64)
_MM, _NN = np.meshgrid(_ODD, _ODD, indexing="ij")
# w = Σ C_mn sin(α_m x) sin(β_n y)，展平为 (m,n) → k 的一维索引
_ALPHA = (_MM * np.pi / A).ravel()
_BETA = (_NN * np.pi / B).ravel()
_C = (16.0 / (np.pi ** 6 * D_FLEX * _MM * _NN
              * (_MM ** 2 / A ** 2 + _NN ** 2 / B ** 2) ** 2)).ravel()
_A2 = _ALPHA ** 2
_B2 = _BETA ** 2
_AB = _ALPHA * _BETA


def _basis(x, y):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    SX = np.sin(x[:, None] * _ALPHA[None, :])              # (N, K2)
    SY = np.sin(y[:, None] * _BETA[None, :])
    CX = np.cos(x[:, None] * _ALPHA[None, :])
    CY = np.cos(y[:, None] * _BETA[None, :])
    return SX, SY, CX, CY


def _sum2(P, Q, coef):
    """Σ_k coef_k · P[:,k] · Q[:,k] —— 避免 einsum 退化为 O(N²K)"""
    return ((P * coef) * Q).sum(axis=1)


def w_analytic(x, y, q):
    """挠度 w（向下为正）"""
    SX, SY, _, _ = _basis(x, y)
    return q * _sum2(SX, SY, _C)


def curv_analytic(x, y, q):
    """曲率分量 κ_xx = -w_xx, κ_yy = -w_yy, κ_xy = -w_xy"""
    SX, SY, CX, CY = _basis(x, y)
    w_xx = -q * _sum2(SX, SY, _C * _A2)
    w_yy = -q * _sum2(SX, SY, _C * _B2)
    w_xy = q * _sum2(CX, CY, _C * _AB)
    return -w_xx, -w_yy, -w_xy


def _w_grad(x, y, q):
    """解析一阶导 w_x, w_y"""
    SX, SY, CX, CY = _basis(x, y)
    wx = q * _sum2(CX, SY, _C * _ALPHA)
    wy = q * _sum2(SX, CY, _C * _BETA)
    return wx, wy


def moment_analytic(x, y, q):
    """弯矩 M_x = D(κ_xx+ν κ_yy), M_y = D(κ_yy+ν κ_xx), M_xy = D(1-ν) κ_xy"""
    kx, ky, kxy = curv_analytic(x, y, q)
    return (D_FLEX * (kx + NU * ky),
            D_FLEX * (ky + NU * kx),
            D_FLEX * (1.0 - NU) * kxy)


def stress_analytic(x, y, zeta, q):
    """纤维应力 σ = 6 ζ M / h²（ζ = z/(h/2) ∈ [-1,1]；h = 1）"""
    mx, my, mxy = moment_analytic(x, y, q)
    z = np.atleast_1d(np.asarray(zeta, dtype=np.float64))
    f = 6.0 * z / H_TH ** 2
    return f * mx, f * my, f * mxy


def mises(sxx, syy, sxy):
    """平面应力 von Mises 等效应力"""
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)


# ===========================================================================
# 1. 参考 FEM：Bogner–Fox–Schmit 16 自由度矩形板单元
# ===========================================================================
def _hermite(t, Le):
    """1D Hermite 形函数及其对 x 的一阶/二阶导数，t∈[0,1] 局部坐标"""
    t = np.asarray(t, dtype=np.float64)
    H = np.stack([1 - 3 * t ** 2 + 2 * t ** 3,
                  Le * (t - 2 * t ** 2 + t ** 3),
                  3 * t ** 2 - 2 * t ** 3,
                  Le * (-t ** 2 + t ** 3)], axis=-1)
    dH = np.stack([(-6 * t + 6 * t ** 2) / Le,
                   1 - 4 * t + 3 * t ** 2,
                   (6 * t - 6 * t ** 2) / Le,
                   -2 * t + 3 * t ** 2], axis=-1)
    d2H = np.stack([(-6 + 12 * t) / Le ** 2,
                    (-4 + 6 * t) / Le,
                    (6 - 12 * t) / Le ** 2,
                    (-2 + 6 * t) / Le], axis=-1)
    return H, dH, d2H


# 单元局部节点 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)
_I_OF = [0, 1, 1, 0]
_J_OF = [0, 0, 1, 1]


def bfs_shape(t, s, Lex, Ley):
    """BFS 形函数与二阶导。t,s ∈[0,1]（形状 (N,)）
    返回 N(16), d2N/dx2, d2N/dy2, d2N/dxdy，均为 (N,16)"""
    Ht, dHt, d2Ht = _hermite(t, Lex)
    Hs, dHs, d2Hs = _hermite(s, Ley)
    N = np.zeros((len(np.atleast_1d(t)), 16))
    d2x = np.zeros_like(N); d2y = np.zeros_like(N); dxy = np.zeros_like(N)
    for nd in range(4):
        iv, idv = 2 * _I_OF[nd], 2 * _I_OF[nd] + 1
        jv, jdv = 2 * _J_OF[nd], 2 * _J_OF[nd] + 1
        vx, dx = Ht[:, iv], Ht[:, idv]
        vy, dy = Hs[:, jv], Hs[:, jdv]
        vxx, dxx = d2Ht[:, iv], d2Ht[:, idv]
        vyy, dyy = d2Hs[:, jv], d2Hs[:, jdv]
        vxp, dxp = dHt[:, iv], dHt[:, idv]
        vyp, dyp = dHs[:, jv], dHs[:, jdv]
        N[:, 4 * nd + 0] = vx * vy
        N[:, 4 * nd + 1] = dx * vy
        N[:, 4 * nd + 2] = vx * dy
        N[:, 4 * nd + 3] = dx * dy
        d2x[:, 4 * nd + 0] = vxx * vy
        d2x[:, 4 * nd + 1] = dxx * vy
        d2x[:, 4 * nd + 2] = vxx * dy
        d2x[:, 4 * nd + 3] = dxx * dy
        d2y[:, 4 * nd + 0] = vx * vyy
        d2y[:, 4 * nd + 1] = dx * vyy
        d2y[:, 4 * nd + 2] = vx * dyy
        d2y[:, 4 * nd + 3] = dx * dyy
        dxy[:, 4 * nd + 0] = vxp * vyp
        dxy[:, 4 * nd + 1] = dxp * vyp
        dxy[:, 4 * nd + 2] = vxp * dyp
        dxy[:, 4 * nd + 3] = dxp * dyp
    return N, d2x, d2y, dxy


_GP, _GW = np.polynomial.legendre.leggauss(4)


def bfs_element(Lex, Ley, q):
    """单元刚度矩阵 (16,16) 与等效载荷向量 (16,)（均布横向载荷 q）"""
    Ke = np.zeros((16, 16)); fe = np.zeros(16)
    Dm = D_FLEX * np.array([[1.0, NU, 0.0],
                            [NU, 1.0, 0.0],
                            [0.0, 0.0, (1.0 - NU) / 2.0]])
    for gp, gw in zip(_GP, _GW):
        for gq, gv in zip(_GP, _GW):
            t = 0.5 * (gp + 1.0); s = 0.5 * (gq + 1.0)
            wJ = 0.25 * gw * gv * Lex * Ley
            N, d2x, d2y, dxy = bfs_shape(np.array([t]), np.array([s]), Lex, Ley)
            Bm = np.vstack([d2x, d2y, 2.0 * dxy])                 # (3,16)
            Ke += wJ * (Bm.T @ Dm @ Bm)
            fe += wJ * q * N[0]
    return Ke, fe


def fem_solve(n, q):
    """n×n 网格 BFS 单元求解四边简支板，返回节点自由度 (4(n+1)²,) 与节点坐标"""
    Lex, Ley = A / n, B / n
    nnode = (n + 1) * (n + 1)
    ndof = 4 * nnode
    K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
    Ke, fe = bfs_element(Lex, Ley, q)
    for ei in range(n):
        for ej in range(n):
            ids = np.array([4 * ((ei + di) * (n + 1) + (ej + dj)) + k
                            for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)]
                            for k in range(4)])
            K[np.ix_(ids, ids)] += Ke
            F[ids] += fe
    fixed = set()
    for i in range(n + 1):
        for j in range(n + 1):
            nid = i * (n + 1) + j
            if i == 0 or i == n:                 # x = 0, x = a 边：w = 0, w_y = 0
                fixed.update([4 * nid + 0, 4 * nid + 2])
            if j == 0 or j == n:                 # y = 0, y = b 边：w = 0, w_x = 0
                fixed.update([4 * nid + 0, 4 * nid + 1])
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)
    d = np.zeros(ndof)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
    return d, xs, ys


def node_dof_from_field(field_fn, xs, ys, q):
    """由连续场（解析或 PINN）采样节点自由度 {w, w_x, w_y, w_xy}"""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    w, wx, wy, wxy = field_fn(X.ravel(), Y.ravel())
    return np.stack([q * w, q * wx, q * wy, q * wxy], axis=1).ravel()


def bfs_interp(de, xi, eta, Lex, Ley):
    """传统后处理：由单元 16 自由度作 BFS 插值，返回 w 与曲率分量
    xi, eta ∈ [-1,1] 单元局部坐标"""
    t = 0.5 * (np.atleast_1d(xi) + 1.0)
    s = 0.5 * (np.atleast_1d(eta) + 1.0)
    N, d2x, d2y, dxy = bfs_shape(t, s, Lex, Ley)
    w = N @ de
    kxx = -(d2x @ de); kyy = -(d2y @ de); kxy = -(dxy @ de)
    return w, kxx, kyy, kxy


# ===========================================================================
# 2. 级 I：能量型 PINN（线性工况下载荷 q 可分离：ŵ = q·û）
# ===========================================================================
class PlatePINN(nn.Module):
    """û(x,y) = sin(πx)·sin(πy)·NN(x,y) —— 四边 w = 0 的硬约束
    选择正弦乘子而非多项式乘子：Navier 解本身即双正弦级数，
    û = w/[sin(πx)sin(πy)] 在域内处处光滑有界，网络更易拟合"""

    def __init__(self, hidden=64, nlayer=4):
        super().__init__()
        layers = []
        layers += [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(nlayer - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        x = xy[:, 0:1]; y = xy[:, 1:2]
        return (torch.sin(np.pi * x) * torch.sin(np.pi * y)) * self.net(xy)


def _grad_hess(u, xy):
    """返回一阶导 (N,2) 与 Hessian (N,2,2)"""
    g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    hx = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0]
    hy = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True)[0]
    return g, torch.stack([hx, hy], dim=1)


def _energy(pinn, xy):
    u = pinn(xy).reshape(-1)
    _, H = _grad_hess(u, xy)
    uxx, uyy = H[:, 0, 0], H[:, 1, 1]
    uxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    bend = (uxx + uyy) ** 2 - 2.0 * (1.0 - NU) * (uxx * uyy - uxy ** 2)
    return 0.5 * D_FLEX * bend.mean() - u.mean()


def train_pinn(pinn, n_epochs=10000, n_int=4000, lr=1e-3, lbfgs_iter=1200,
               log_every=1000):
    """能量损失（对 û，已除 q²）：
       L = ½ D·mean[(û_xx+û_yy)² − 2(1−ν)(û_xx û_yy − û_xy²)] − mean[û]
       先 Adam 粗调，再 LBFGS 精调（能量泛函为标量泛函，拟牛顿法收敛更快）"""
    opt = torch.optim.Adam(pinn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    hist = []
    for it in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(n_int, 2, requires_grad=True)
        loss = _energy(pinn, xy)
        loss.backward()
        opt.step(); sched.step()
        hist.append(loss.item())
        if (it + 1) % log_every == 0:
            print(f"  [PINN-Adam] epoch {it+1:5d}  L={loss.item():.6e}")
    # ---- LBFGS 精调（固定配点集，保证线搜索一致性）----
    xy_fix = torch.rand(max(n_int, 4096), 2, requires_grad=True)
    opt2 = torch.optim.LBFGS(pinn.parameters(), max_iter=lbfgs_iter,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-14, tolerance_change=1e-16)
    cnt = {"n": 0}

    def closure():
        opt2.zero_grad()
        loss = _energy(pinn, xy_fix)
        loss.backward()
        cnt["n"] += 1
        if cnt["n"] % 200 == 0:
            print(f"  [PINN-LBFGS] iter {cnt['n']:5d}  L={loss.item():.6e}")
        return loss

    opt2.step(closure)
    hist.append(_energy(pinn, xy_fix).item())
    return np.array(hist)


def pinn_field(pinn, xy_np, q_val):
    """返回 q·(û, û_x, û_y, û_xy)"""
    xy = torch.tensor(np.asarray(xy_np, dtype=np.float64), requires_grad=True)
    u = pinn(xy).reshape(-1)
    g, H = _grad_hess(u, xy)
    uxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    out = [u.detach().numpy(), g[:, 0].detach().numpy(),
           g[:, 1].detach().numpy(), uxy.detach().numpy()]
    return [q_val * o for o in out]


# ===========================================================================
# 3. 级 II：DeepONet 神经算子（单元内 位移 / 曲率 / 应力）
# ===========================================================================
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64, nlayer=3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.Tanh()]
        for _ in range(nlayer - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class PlateOperator(nn.Module):
    """G_φ: d_e(16) ↦ {ŵ, κ̂_xx, κ̂_yy, κ̂_xy, σ̂_xx, σ̂_yy, σ̂_xy}
    branch 输入 = 单元 16 自由度（按特征归一化：w~10⁻³ 与 w_x~10⁻²、w_xy~10⁻¹ 量级
    相差百倍，直接喂给 MLP 会使小分量被 tanh 饱和掩盖，故在输入层之前
    对每个特征除以训练集统计的幅值参考）。
    branch: R^16 → R^p
    trunk_wk: R²(ξ,η) → R^{4p}   —— 位移与三个曲率分量共享 trunk，分块内积
    trunk_s : R³(ξ,η,ζ) → R^{3p} —— 三个应力分量共享 trunk
    """

    def __init__(self, p=64, hidden=64, d_in=16):
        super().__init__()
        self.p = p
        self.branch = MLP(d_in, p, hidden)
        self.trunk_wk = MLP(2, 4 * p, hidden)
        self.trunk_s = MLP(3, 3 * p, hidden)
        self.b0 = nn.Parameter(torch.zeros(7))

    def _branch(self, d):
        return self.branch(d).reshape(-1)

    def wk_at(self, d, xi):
        """xi: (N,2) 或 (2,) → 输出 (N,4): ŵ, κ̂_xx, κ̂_yy, κ̂_xy"""
        b = self._branch(d)
        xi = xi.reshape(-1, 2)
        T = self.trunk_wk(xi).reshape(-1, 4, self.p)
        out = torch.einsum("nkp,p->nk", T, b)
        return out + self.b0[0:4]

    def s_at(self, d, xiz):
        """xiz: (N,3) = (ξ,η,ζ) → 输出 (N,3): σ̂_xx, σ̂_yy, σ̂_xy"""
        b = self._branch(d)
        xiz = xiz.reshape(-1, 3)
        T = self.trunk_s(xiz).reshape(-1, 3, self.p)
        out = torch.einsum("nkp,p->nk", T, b)
        return out + self.b0[4:7]


def make_operator_dataset(qs, n=N_MESH, n_pts=10, seed=0):
    """多工况: 解析节点自由度 -> 单元内采样 (d_e, ξ, η, ζ) -> 标签
    返回样本列表，每个样本 = (de(16), pts(N,2), zeta(N,), 标签数组)"""
    xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
    Lex = A / n; Ley = B / n
    g = np.linspace(-1, 1, n_pts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    XI = XI.ravel(); ETA = ETA.ravel()
    rng = np.random.default_rng(seed)
    # 预计算：每个单元的采样全局坐标（一次性拼成大数组，避免逐单元调用解析解）
    npts = XI.size
    PX = np.concatenate([xs[ei] + 0.5 * Lex * (XI + 1.0) for ei in range(n) for _ in range(n)])
    PY = np.concatenate([ys[ej] + 0.5 * Ley * (ETA + 1.0) for _ in range(n) for ej in range(n)])
    samples = []
    for q in qs:
        # 节点自由度（解析）
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Xr, Yr = X.ravel(), Y.ravel()
        w = w_analytic(Xr, Yr, q)
        kx, ky, kxy = curv_analytic(Xr, Yr, q)
        wx, wy = _w_grad(Xr, Yr, q)
        dnode = np.stack([w, wx, wy, -kxy], axis=1)          # ((n+1)²,4)
        # 全部单元采样点的解析标签（一次性）
        w_all = w_analytic(PX, PY, q)
        kx_a, ky_a, kxy_a = curv_analytic(PX, PY, q)
        zeta = rng.choice([-1.0, 1.0], size=PX.size)
        sx_a, sy_a, sxy_a = stress_analytic(PX, PY, zeta, q)
        for c, (ei, ej) in enumerate([(i, j) for i in range(n) for j in range(n)]):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = dnode[ids].ravel()                          # 16（真实尺度）
            sl = slice(c * npts, (c + 1) * npts)
            samples.append((de, np.stack([XI, ETA], axis=1), zeta[sl],
                            w_all[sl], np.stack([kx_a[sl], ky_a[sl], kxy_a[sl]], axis=1),
                            np.stack([sx_a[sl], sy_a[sl], sxy_a[sl]], axis=1), float(q)))
    return samples


def compute_feature_scale(samples, qtest=1.0):
    """按特征统计归一化系数：取训练样本各自由度分量 95% 分位幅值的倒数，
    使 q=1 验证工况的输入量级 O(1)。返回 (16,) 缩放向量"""
    A = np.stack([s[0] for s in samples])                    # (N,16)
    amp = np.percentile(np.abs(A), 95, axis=0)
    amp = np.maximum(amp, 1e-12)
    # 验证工况 q=1 量级介于训练 q∈[0.5,0.9]∪[1.1,2.0] 之间，取 95% 分位即可 O(1)
    return 1.0 / amp


CORNER_XI = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                         dtype=torch.float64)


def train_operator(op, samples, n_epochs=4000, batch=24, lr=1e-3, log_every=250,
                   Le=0.25, scale=None, lam_w=1.0, weights=None):
    """损失六项（位移监督权重 lam_w 可调，默认 1）：
    ① 位移监督 ② 曲率监督 ③ 角点插值自洽（w/w_x/w_y/w_xy）
    ④ 应力监督 ⑤ 应力-曲率一致性 ⑥ 曲率-位移二阶导自洽
    weights: 可选 dict，如 {'L1':0.0,'L3':0.0,'L6':1e-1} 覆盖各项权重（消融用）。
    默认 {L1:lam_w, L2:1, L3:1, L4:1, L5:1, L6:1e-2}。"""
    n_func = len(samples)
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    keys = ["L1", "L2", "L3", "L4", "L5", "L6"]
    history = {k: [] for k in keys}
    scl = torch.tensor(scale if scale is not None else np.ones(16), dtype=torch.float64)
    wts = {"L1": lam_w, "L2": 1.0, "L3": 1.0, "L4": 1.0, "L5": 1.0, "L6": 1e-2}
    if weights:
        wts.update({k: float(v) for k, v in weights.items()})
    for it in range(n_epochs):
        opt.zero_grad()
        idx = np.random.choice(n_func, batch, replace=False)
        acc = {k: torch.tensor(0.0) for k in keys}
        for i in idx:
            de, pts, zeta, w_ref, k_ref, s_ref, q = samples[i]
            d_t = (torch.tensor(de) * scl).reshape(1, 16)
            xy = torch.tensor(pts, requires_grad=True)
            wk = op.wk_at(d_t, xy)                       # (N,4)
            w_hat = wk[:, 0]; k_hat = wk[:, 1:4]
            sz = torch.tensor(zeta)
            s_hat = op.s_at(d_t, torch.stack([xy[:, 0], xy[:, 1], sz], dim=1))
            w_r = torch.tensor(w_ref); k_r = torch.tensor(k_ref); s_r = torch.tensor(s_ref)
            acc["L1"] = acc["L1"] + ((w_hat - w_r) ** 2).mean()
            acc["L2"] = acc["L2"] + ((k_hat - k_r) ** 2).mean()
            # ③ 角点插值自洽（仅用前 16 自由度）
            xc = CORNER_XI.clone().requires_grad_(True)
            wc = op.wk_at(d_t, xc)[:, 0]
            gc = torch.autograd.grad(wc.sum(), xc, create_graph=True)[0]
            hx = torch.autograd.grad(gc[:, 0].sum(), xc, create_graph=True)[0]
            dnode = torch.tensor(de).reshape(4, 4)
            acc["L3"] = acc["L3"] + ((wc - dnode[:, 0]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Le) * gc[:, 0] - dnode[:, 1]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((2.0 / Le) * gc[:, 1] - dnode[:, 2]) ** 2).mean()
            acc["L3"] = acc["L3"] + (((4.0 / Le ** 2) * hx[:, 1] - dnode[:, 3]) ** 2).mean()
            # ④ 应力监督
            acc["L4"] = acc["L4"] + ((s_hat - s_r) ** 2).mean()
            # ⑤ 应力-曲率一致性：σ = 6ζ M/h², M = D(κ+νκ)
            zz = sz
            mx = D_FLEX * (k_hat[:, 0] + NU * k_hat[:, 1])
            my = D_FLEX * (k_hat[:, 1] + NU * k_hat[:, 0])
            mxy = D_FLEX * (1.0 - NU) * k_hat[:, 2]
            f = 6.0 * zz / H_TH ** 2
            s_cons = torch.stack([f * mx, f * my, f * mxy], dim=1)
            acc["L5"] = acc["L5"] + ((s_hat - s_cons) ** 2).mean()
            # ⑥ 曲率-位移二阶导自洽
            g1 = torch.autograd.grad(w_hat.sum(), xy, create_graph=True)[0]
            h1 = torch.autograd.grad(g1[:, 0].sum(), xy, create_graph=True)[0]
            h2 = torch.autograd.grad(g1[:, 1].sum(), xy, create_graph=True)[0]
            c = 4.0 / Le ** 2
            k_from_w = torch.stack([-c * h1[:, 0], -c * h2[:, 1],
                                    -c * 0.5 * (h1[:, 1] + h2[:, 0])], dim=1)
            acc["L6"] = acc["L6"] + ((k_hat - k_from_w) ** 2).mean()
        loss = (wts["L1"] * acc["L1"] + wts["L2"] * acc["L2"] + wts["L3"] * acc["L3"]
                + wts["L4"] * acc["L4"] + wts["L5"] * acc["L5"]
                + wts["L6"] * acc["L6"]) / batch
        loss.backward()
        opt.step(); sched.step()
        for k in keys:
            history[k].append((acc[k] / batch).item())
        if (it + 1) % log_every == 0:
            print("  [OP] epoch %5d  " % (it + 1) +
                  "  ".join(f"{k}={history[k][-1]:.3e}" for k in keys))
    return history


def l2rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


# ---------------------------------------------------------------------------
# 4b. 位移物理解码（动作 B）：w* = ∇⁻²(κ_xx+κ_yy) 五点离散 Poisson + 简支 BC
#     assemble 输出场每单元含共享边界点（4×npts 行），先压到唯一坐标网格
# ---------------------------------------------------------------------------
def _unique_grid_indices(npts):
    """去共享边界重复点 → 唯一坐标网格索引（行/列同用）。
    每单元 npts 点含两端，单元 i 末点 = 单元 i+1 首点；
    保留 0..npts-2（每单元前 npts-1 点）+ 最后一单元末点 → 1+n*(npts-1) 唯一坐标"""
    keep = []
    for ei in range(N_MESH):
        keep += list(range(ei * npts, ei * npts + npts - 1))
    keep.append(N_MESH * npts - 1)
    return np.asarray(keep, dtype=np.int64)


def poisson_reconstruct_w(kxx, kyy, h):
    """五点离散 ∇²w = -(κ_xx+κ_yy)，简支 w|∂Ω=0。
    kxx/kyy: (M,M) 均匀网格场（行沿 x、列沿 y）→ 返回 w* 同形状。"""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    M = kxx.shape[0]
    s = -(kxx + kyy)
    N = (M - 2) ** 2
    rows, cols, vals, rhs = [], [], [], []
    for i in range(1, M - 1):
        for j in range(1, M - 1):
            r = (i - 1) * (M - 2) + (j - 1)
            rows.append(r); cols.append(r); vals.append(-4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 1 <= ii <= M - 2 and 1 <= jj <= M - 2:
                    rows.append(r)
                    cols.append((ii - 1) * (M - 2) + (jj - 1))
                    vals.append(1.0)
            rhs.append(s[i, j] * h * h)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    w = spla.spsolve(A, np.asarray(rhs))
    W = np.zeros((M, M))
    W[1:M - 1, 1:M - 1] = w.reshape(M - 2, M - 2)
    return W


def wstar_from_field(F, npts=15):
    """从 assemble 输出场 F 重构位移 w*（自动压到唯一坐标网格）。
    返回 (w*(M,M), x 坐标 (M,))"""
    keep = _unique_grid_indices(npts)
    kx = F["KX"][np.ix_(keep, keep)]
    ky = F["KY"][np.ix_(keep, keep)]
    M = keep.size
    x = np.linspace(0.0, A, M)
    h = x[1] - x[0]
    return poisson_reconstruct_w(kx, ky, h), x


# ===========================================================================
# 4. 架构示意图
# ===========================================================================
def _box(ax, cx, cy, w, h, text, fc="#E8F1FB", ec="#2F6FB3", fs=11,
         tc="#1a1a1a", bold=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def _arrow(ax, p0, p1, color="#444444"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color))


def fig_plate_setup():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.set_xlim(-0.55, 1.75); ax.set_ylim(-0.55, 1.35); ax.axis("off")
    # 板面
    ax.add_patch(Rectangle((0, 0), 1.0, 1.0, fc="#DCE9F5", ec="#2F6FB3", lw=2.2, zorder=1))
    # 均布载荷箭头（9 个，向下）
    for xx in np.linspace(0.12, 0.88, 5):
        for yy in np.linspace(0.15, 0.85, 5):
            ax.annotate("", xy=(xx, yy - 0.055), xytext=(xx, yy + 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.3))
    ax.text(0.5, 1.10, r"$q(x,y)=q$（横向均布载荷）", ha="center",
            fontsize=12.5, color="#C0392B")
    # 简支边界：四边小三角（每边 5 个）
    tri = np.array([[-0.05, 0.0], [0.05, 0.0], [0.0, -0.075]])
    for u in np.linspace(0.15, 0.85, 5):
        for (px, py), rot in [((u, 0.0), 0), ((u, 1.0), 180)]:
            th = np.deg2rad(rot)
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            p = tri @ Rm.T + np.array([px, py])
            ax.add_patch(plt.Polygon(p, fc="#455A64", ec="#37474F", lw=0.8, zorder=3))
        for (px, py), rot in [((0.0, u), 90), ((1.0, u), 270)]:
            th = np.deg2rad(rot)
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            p = tri @ Rm.T + np.array([px, py])
            ax.add_patch(plt.Polygon(p, fc="#455A64", ec="#37474F", lw=0.8, zorder=3))
    # 坐标轴
    ax.annotate("", xy=(1.32, -0.16), xytext=(0.0, -0.16),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    ax.annotate("", xy=(-0.16, 1.32), xytext=(-0.16, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    ax.text(1.35, -0.20, r"$x$", fontsize=13)
    ax.text(-0.20, 1.35, r"$y$", fontsize=13)
    ax.text(-0.30, 0.5, r"$b=1$", fontsize=12, rotation=90, ha="center")
    ax.text(0.5, -0.30, r"$a=1$", fontsize=12, ha="center")
    ax.text(1.15, 0.92, r"$w=0,\ M_n=0$（四边简支）", fontsize=11.5, color="#455A64")
    ax.text(0.5, -0.46, r"$D=\dfrac{Eh^{3}}{12(1-\nu^{2})}=1,\quad h=1,\quad \nu=0.3$",
            ha="center", fontsize=12.5)
    ax.set_title("四边简支矩形薄板基准算例（Kirchhoff–Love 平板壳模型）", fontsize=13.5)
    fig.tight_layout(); fig.savefig(f"{RES}/fig1_plate_setup.png", dpi=150); plt.close(fig)


def fig_pinn_arch():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(-0.6, 10.8); ax.set_ylim(0, 4.9); ax.axis("off")
    # 主链（推理）：输入 -> MLP -> 硬约束 -> û -> ŵ=qû
    _box(ax, 0.9, 3.35, 1.5, 0.85, r"输入 $(x,y)$")
    _box(ax, 2.85, 3.35, 1.45, 1.6, "MLP\n64×4\ntanh")
    _box(ax, 4.95, 3.35, 1.7, 0.85, r"$\sin(\pi x)\sin(\pi y)$" + "\n硬约束")
    _box(ax, 7.1, 3.35, 1.3, 0.85, r"$\hat u(x,y)$")
    _box(ax, 9.4, 3.35, 1.5, 0.85, r"$\hat w=q\,\hat u$")
    for x0, x1 in [(1.65, 2.125), (3.575, 4.1), (5.8, 6.45), (7.75, 8.65)]:
        _arrow(ax, (x0, 3.35), (x1, 3.35))
    # 载荷 q 输入（置于 ŵ 正下方，避开能量框）
    _box(ax, 9.4, 2.0, 0.95, 0.7, r"$q$")
    _arrow(ax, (9.4, 2.35), (9.4, 2.925))
    # 能量损失框（公式分行，右侧不越界，q 框不重叠）
    _box(ax, 6.8, 0.95, 7.2, 1.2,
         r"能量损失：$\mathcal{L}[\hat u]=\frac{1}{N}\sum_j\left\{\frac{D}{2}\left[g(\hat u)\right]_j-\hat u_j\right\}\to\min$" + "\n" +
         r"$g(\hat u)=(\hat u_{xx}+\hat u_{yy})^{2}-2(1-\nu)(\hat u_{xx}\hat u_{yy}-\hat u_{xy}^{2})$（与式(11)同）",
         fc="#FEF3E2", ec="#C07A00", fs=9.5)
    _arrow(ax, (7.1, 1.58), (7.1, 2.9))
    ax.text(7.28, 2.2, "梯度训练", fontsize=8, color="#B26A00", ha="left")
    _box(ax, 1.0, 1.4, 1.9, 0.95, r"变分原理" + "\n" + r"$\Pi[\hat u]\to\min$",
         fc="#E8F5E9", ec="#2E7D32", fs=10)
    _arrow(ax, (1.95, 1.4), (3.2, 1.4))
    ax.set_title(r"级 I：能量型 PINN 节点自由度求解器（线性工况 $\hat w=q\cdot\hat u$，替代 $\mathbf{K}\mathbf{d}=\mathbf{f}$）",
                 fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{RES}/fig2_pinn_arch.png", dpi=150); plt.close(fig)


def fig_operator_arch():
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.set_xlim(-0.6, 10.8); ax.set_ylim(0, 5.2); ax.axis("off")
    _box(ax, 0.5, 3.0, 1.6, 1.0, r"单元节点自由度" + "\n" + r"$\mathbf{d}_e\in\mathbb{R}^{16}$", fs=9.5)
    _box(ax, 2.7, 3.0, 1.5, 0.9, r"branch 网络" + "\n" + r"$\mathbb{R}^{16}\to\mathbb{R}^{p}$",
         fc="#EFEBE9", ec="#6D4C41", fs=9.5)
    _box(ax, 5.4, 4.45, 1.5, 0.55, r"查询位置 $(\xi,\eta)$", fc="#E0F7FA", ec="#00838F", fs=9)
    _box(ax, 5.4, 1.55, 1.5, 0.55, r"查询位置 $(\xi,\eta,\zeta)$", fc="#E0F7FA", ec="#00838F", fs=9)
    _box(ax, 5.4, 3.55, 1.5, 0.75, r"trunk-$w/\kappa$" + "\n" + r"$\mathbb{R}^{2}\!\to\!\mathbb{R}^{4p}$",
         fc="#E0F7FA", ec="#00838F", fs=9)
    _box(ax, 5.4, 2.45, 1.5, 0.75, r"trunk-$\sigma$" + "\n" + r"$\mathbb{R}^{3}\!\to\!\mathbb{R}^{3p}$",
         fc="#E0F7FA", ec="#00838F", fs=9)
    _box(ax, 8.7, 3.0, 3.0, 1.5,
         r"$\hat w_e=\langle b,t_w\rangle$" + "\n" +
         r"$\hat\kappa_{xx},\hat\kappa_{yy},\hat\kappa_{xy}=\langle b,t_{\kappa}\rangle$" + "\n" +
         r"$\hat\sigma_{xx},\hat\sigma_{yy},\hat\sigma_{xy}=\langle b,t_{\sigma}\rangle$",
         fc="#F3E5F5", ec="#6A1B9A", fs=9.5)
    _box(ax, 8.7, 1.0, 3.0, 0.65, "损失：数据监督 + 角点自洽\n+ 应力-曲率一致性",
         fc="#FFF3E0", ec="#E65100", fs=9)
    _arrow(ax, (1.3, 3.0), (1.95, 3.0))
    _arrow(ax, (3.45, 3.0), (4.65, 3.55))
    _arrow(ax, (3.45, 3.0), (4.65, 2.45))
    _arrow(ax, (5.4, 4.17), (5.4, 3.92))
    _arrow(ax, (5.4, 1.83), (5.4, 2.08))
    _arrow(ax, (6.15, 3.55), (7.2, 3.0))
    _arrow(ax, (6.15, 2.45), (7.2, 3.0))
    _arrow(ax, (8.7, 2.25), (8.7, 1.33))
    ax.set_title(r"级 II：DeepONet 神经算子（替代 $\mathbf{N},\mathbf{B}$ 形函数后处理）", fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{RES}/fig3_operator_arch.png", dpi=150); plt.close(fig)


def fig_system_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 12.0); ax.set_ylim(0, 5.6); ax.axis("off")
    # ---- 行 1：推理主链（实线）----
    _box(ax, 1.4, 4.5, 2.4, 0.95, r"物理问题" + "\n" + r"薄板/壳 + 载荷 $q(x,y)$")
    _box(ax, 4.6, 4.5, 2.6, 0.95, r"级 I：能量型 PINN" + "\n" + r"节点自由度 $\{w,w_x,w_y,w_{xy}\}$",
         fc="#E3F2FD", ec="#1565C0")
    _box(ax, 8.1, 4.5, 2.8, 0.95, r"级 II：神经算子" + "\n" + r"单元内 $\hat\kappa,\hat\sigma$",
         fc="#F3E5F5", ec="#6A1B9A")
    _box(ax, 11.0, 4.5, 1.3, 0.95, r"仿真结果")
    for a in [(2.6, 4.5, 3.3, 4.5), (5.9, 4.5, 6.7, 4.5), (9.5, 4.5, 10.35, 4.5)]:
        _arrow(ax, (a[0], a[1]), (a[2], a[3]))
    # ---- 行 2：被替代的传统 FE 环节（点线连接主框，红框文字自明）----
    _box(ax, 4.6, 3.35, 2.6, 0.9, r"被替代：刚度组装 + 求解" + "\n" + r"$\mathbf{K}\mathbf{d}=\mathbf{f}$",
         fc="#FBE9E7", ec="#C62828", fs=10)
    _box(ax, 8.1, 3.35, 2.8, 0.9, r"被替代：形函数 + 本构后处理" + "\n" + r"$\sigma=\frac{6\zeta}{h^{2}}\mathbf{M}$",
         fc="#FBE9E7", ec="#C62828", fs=10)
    for cx in (5.9, 9.5):
        ax.annotate("", xy=(cx, 3.82), xytext=(cx, 4.02),
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2,
                                    linestyle=":"))
    ax.text(6.02, 3.93, "替代", ha="left", fontsize=8.5, color="#C62828")
    ax.text(9.62, 3.93, "替代", ha="left", fontsize=8.5, color="#C62828")
    # ---- 训练标签流（虚线，仅指向级 II：级 I 为无标签能量训练）----
    ax.annotate("", xy=(9.2, 2.88), xytext=(9.2, 2.16),
                arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.3,
                                linestyle="--"))
    ax.text(9.38, 2.5, "离线标签", ha="left", fontsize=9, color="#555")
    _box(ax, 10.35, 1.5, 2.9, 1.25,
         r"离线参照场（训练期）" + "\n" + r"Navier 级数 / 高保真 FE" + "\n" + "→ 级 II 训练标签",
         fc="#F1F8E9", ec="#558B2F", fs=9.5)
    # ---- 物理一致性约束（训练期，施加于两级网络）----
    _box(ax, 3.5, 0.8, 5.6, 1.1,
         "物理一致性约束（训练期，作用于两级网络）" + "\n" +
         "变分原理 · 角点插值自洽 · 应力-曲率一致 · 曲率-位移自洽",
         fc="#E8F5E9", ec="#2E7D32", fs=9.5)

    ax.set_title("两级神经网络替代体系总体架构", fontsize=13.5)
    fig.tight_layout(); fig.savefig(f"{RES}/fig4_system_pipeline.png", dpi=150); plt.close(fig)


# ===========================================================================
# 5. 主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="忽略已保存模型，重新训练")
    ap.add_argument("--ep-pinn", type=int, default=8000)
    ap.add_argument("--ep-op", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=1200)
    ap.add_argument("--no-holdout", action="store_true",
                    help="算子训练覆盖 q∈[0.5,2.0]（含 q=1），不做留出")
    ap.add_argument("--scale", action="store_true",
                    help="对 branch 输入做特征归一化（默认关闭）")
    ap.add_argument("--lw", type=float, default=1.0,
                    help="位移监督 L1 权重（默认 1.0）")
    ap.add_argument("--tag", type=str, default="",
                    help="模型文件名后缀（并行实验用）")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试（极少 epoch）")
    args = ap.parse_args()
    if args.smoke:
        args.ep_pinn, args.ep_op, args.lbfgs = 200, 40, 30

    t0 = time.time()
    n = N_MESH
    Lex = A / n; Ley = B / n
    xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    Xnr, Ynr = Xn.ravel(), Yn.ravel()

    print("[绘图] 架构示意图 ...")
    fig_plate_setup(); fig_pinn_arch(); fig_operator_arch(); fig_system_pipeline()

    # ---------- 解析解自检 ----------
    wc = float(w_analytic(np.array([0.5]), np.array([0.5]), 1.0)[0])
    print(f"[解析解] 中心挠度 w(a/2,b/2) = {wc:.6e}  (理论 4.062e-3)")
    mx_c = moment_analytic(np.array([0.5]), np.array([0.5]), 1.0)[0][0]
    print(f"[解析解] 中心弯矩 M_x = {mx_c:.6e}  (理论 4.789e-2)")

    # ---------- FEM 基准 ----------
    d_fem, _, _ = fem_solve(n, Q_TEST)
    w_ex_nodes = w_analytic(Xnr, Ynr, Q_TEST)
    wx_ex, wy_ex = _w_grad(Xnr, Ynr, Q_TEST)
    kxy_ex = curv_analytic(Xnr, Ynr, Q_TEST)[2]
    d_ex = np.stack([w_ex_nodes, wx_ex, wy_ex, -kxy_ex], axis=1).ravel()
    e_fem_node = float(np.abs(d_fem[0::4] - w_ex_nodes).max())
    print(f"[FEM] {n}×{n} BFS 单元 节点挠度最大误差 = {e_fem_node:.3e}")

    # ---------- 级 I: PINN ----------
    pinn_path = os.path.join(RES, "pinn_state.pt")
    hist_pinn_path = os.path.join(RES, "hist_pinn.npy")
    pinn = PlatePINN(hidden=64, nlayer=4)
    if os.path.exists(pinn_path) and not args.retrain:
        pinn.load_state_dict(torch.load(pinn_path, weights_only=False))
        hist_pinn = (np.load(hist_pinn_path)
                     if os.path.exists(hist_pinn_path) else np.array([0.0]))
        print("[PINN] 载入已训练模型（含训练历史）" if hist_pinn.size > 1 else "[PINN] 载入已训练模型")
    else:
        print("\n===== 级 I: 能量型 PINN 训练 =====")
        hist_pinn = train_pinn(pinn, n_epochs=args.ep_pinn, n_int=4000,
                               lbfgs_iter=args.lbfgs)
        torch.save(pinn.state_dict(), pinn_path)
        np.save(hist_pinn_path, hist_pinn)

    # 全场评估
    ng = 61
    xg = np.linspace(0, A, ng); yg = np.linspace(0, B, ng)
    XG, YG = np.meshgrid(xg, yg, indexing="ij")
    XGr, YGr = XG.ravel(), YG.ravel()
    w_pin, _, _, _ = pinn_field(pinn, np.stack([XGr, YGr], axis=1), Q_TEST)
    w_ex_grid = w_analytic(XGr, YGr, Q_TEST)
    e_pinn_L2 = l2rel(w_pin, w_ex_grid)
    e_pinn_max = float(np.abs(w_pin - w_ex_grid).max())
    # 节点自由度
    d_pin = np.array(pinn_field(pinn, np.stack([Xnr, Ynr], axis=1), Q_TEST)).T.ravel()
    e_pinn_node_w = float(np.abs(d_pin[0::4] - w_ex_nodes).max())
    e_pinn_node_gx = float(np.abs(d_pin[1::4] - wx_ex).max())
    e_pinn_node_gy = float(np.abs(d_pin[2::4] - wy_ex).max())
    print(f"[PINN] 全场 L2={e_pinn_L2:.3e}  max|Δw|={e_pinn_max:.3e}  "
          f"节点 max|Δw|={e_pinn_node_w:.3e}  max|Δw_x|={e_pinn_node_gx:.3e}  "
          f"max|Δw_y|={e_pinn_node_gy:.3e}")
    print(f"[PINN] w(中心): PINN={float(w_pin[ng*ng//2]):.6e}  解析={float(w_ex_grid[ng*ng//2]):.6e}")

    # ---------- 级 II: 算子 ----------
    op_tag = f"op_state{args.tag}.pt"
    op_path = os.path.join(RES, op_tag)
    qs_all = np.random.default_rng(3).uniform(Q_LO, Q_HI, 200)
    if args.no_holdout:
        # 训练覆盖全域含 q=1：140 个 + 显式补 1.0/0.98/1.02（闭环基准属训练区间）
        qs_train = np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])
    else:
        qs_train = qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:120]
    samples = make_operator_dataset(qs_train, n=n, n_pts=10, seed=1)
    scale = compute_feature_scale(samples) if args.scale else None
    print(f"\n[数据] 算子训练工况 {len(qs_train)} 个（工作网格 {n}×{n}），"
          f"样本单元 {len(samples)} 个；特征归一化: {args.scale}；位移权重: {args.lw}")
    op = PlateOperator(p=64, hidden=64)
    hist_op_path = os.path.join(RES, "hist_op.npz")
    if os.path.exists(op_path) and not args.retrain:
        op.load_state_dict(torch.load(op_path, weights_only=False))
        if os.path.exists(hist_op_path):
            z = np.load(hist_op_path)
            hist_op = {k: z[k].tolist()
                       for k in ["L1", "L2", "L3", "L4", "L5", "L6"]}
            print("[OP] 载入已训练模型（含训练历史）")
        else:
            hist_op = {k: [0.0] for k in ["L1", "L2", "L3", "L4", "L5", "L6"]}
            print("[OP] 载入已训练模型")
    else:
        print("===== 级 II: DeepONet 神经算子训练 =====")
        hist_op = train_operator(op, samples, n_epochs=args.ep_op, batch=24, Le=Lex,
                                 scale=scale, lam_w=args.lw)
        torch.save(op.state_dict(), op_path)
        np.savez(hist_op_path,
                 **{k: np.asarray(v) for k, v in hist_op.items()})

    # ---------- 闭环：PINN 节点自由度 → 算子 → 单元内场 ----------
    print("\n===== 闭环验证 (q=1) =====")

    def element_predict(de, npts=15):
        """单元 16 自由度 → 算子输出（branch 输入按特征归一化）与 BFS 输出"""
        g = np.linspace(-1, 1, npts)
        XI, ETA = np.meshgrid(g, g, indexing="ij")
        pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
        d_t = (torch.tensor(de) * torch.tensor(
            scale if scale is not None else np.ones(16))).reshape(1, 16)
        with torch.no_grad():
            wk = op.wk_at(d_t, torch.tensor(pts)).numpy()
            sp = op.s_at(d_t, torch.stack([torch.tensor(pts[:, 0]),
                                           torch.tensor(pts[:, 1]),
                                           torch.full((pts.shape[0],), 1.0)], dim=1)).numpy()
        w_op, k_op, s_op = wk[:, 0], wk[:, 1:4], sp
        w_bf, kxx_bf, kyy_bf, kxy_bf = bfs_interp(de, pts[:, 0], pts[:, 1], Lex, Ley)
        k_bf = np.stack([kxx_bf, kyy_bf, kxy_bf], axis=1)
        mx_bf = D_FLEX * (k_bf[:, 0] + NU * k_bf[:, 1])
        my_bf = D_FLEX * (k_bf[:, 1] + NU * k_bf[:, 0])
        mxy_bf = D_FLEX * (1 - NU) * k_bf[:, 2]
        s_bf = (6.0 / H_TH ** 2) * np.stack([mx_bf, my_bf, mxy_bf], axis=1)
        return pts, w_op, k_op, s_op, w_bf, k_bf, s_bf

    def assemble(dof, src, npts=15):
        """src: 'op' 或 'bfs'；返回沿结构化的全场数组与坐标"""
        n1 = npts
        W = np.zeros((n * n1, n * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
        KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
        SXY = np.zeros_like(W); WX = np.zeros_like(W); WY = np.zeros_like(W)
        for ei in range(n):
            for ej in range(n):
                ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                       [(0, 0), (1, 0), (1, 1), (0, 1)]]
                de = dof.reshape(-1, 4)[ids].ravel()
                pts, w_op, k_op, s_op, w_bf, k_bf, s_bf = element_predict(de, npts)
                if src == "op":
                    w, k, s = w_op, k_op, s_op
                else:
                    w, k, s = w_bf, k_bf, s_bf
                sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
                W[sl_i, sl_j] = w.reshape(n1, n1)
                KX[sl_i, sl_j] = k[:, 0].reshape(n1, n1)
                KY[sl_i, sl_j] = k[:, 1].reshape(n1, n1)
                KXY[sl_i, sl_j] = k[:, 2].reshape(n1, n1)
                SX[sl_i, sl_j] = s[:, 0].reshape(n1, n1)
                SY[sl_i, sl_j] = s[:, 1].reshape(n1, n1)
                SXY[sl_i, sl_j] = s[:, 2].reshape(n1, n1)
                xg = xs[ei] + 0.5 * Lex * (pts[:, 0] + 1)
                yg = ys[ej] + 0.5 * Ley * (pts[:, 1] + 1)
                WX[sl_i, sl_j] = xg.reshape(n1, n1); WY[sl_i, sl_j] = yg.reshape(n1, n1)
        return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)

    closed = {}
    for tag, dof in [("pinn", d_pin), ("exact", d_ex), ("fem", d_fem)]:
        for src in (["op"] if tag == "pinn" else ["op", "bfs"]):
            F = assemble(dof, src, npts=15)
            XR, YR = F["X"].ravel(), F["Y"].ravel()
            w_t = w_analytic(XR, YR, Q_TEST)
            kx_t, ky_t, kxy_t = curv_analytic(XR, YR, Q_TEST)
            sx_t, sy_t, sxy_t = stress_analytic(XR, YR, 1.0, Q_TEST)
            key = f"{tag}_{src}"
            rec = {
                "w": l2rel(F["W"].ravel(), w_t),
                "kxx": l2rel(F["KX"].ravel(), kx_t),
                "kyy": l2rel(F["KY"].ravel(), ky_t),
                "kxy": l2rel(F["KXY"].ravel(), kxy_t),
                "sxx": l2rel(F["SX"].ravel(), sx_t),
                "syy": l2rel(F["SY"].ravel(), sy_t),
                "sxy": l2rel(F["SXY"].ravel(), sxy_t),
                "w_center": float(F["W"][F["W"].shape[0] // 2, F["W"].shape[1] // 2]),
            }
            if src == "op":
                # 位移物理解码：w* = ∇⁻²(κ_xx+κ_yy)（简支 BC，唯一坐标网格）
                w_star, x_u = wstar_from_field(F, npts=15)
                XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
                w_t_u = w_analytic(XU.ravel(), YU.ravel(), Q_TEST).reshape(XU.shape)
                rec["w_star"] = l2rel(w_star.ravel(), w_t_u.ravel())
                rec["w_star_center"] = float(
                    w_star[w_star.shape[0] // 2, w_star.shape[1] // 2])
            closed[key] = rec
            print(f"[{key:10s}] w={closed[key]['w']:.3e}  κxx={closed[key]['kxx']:.3e}  "
                  f"κyy={closed[key]['kyy']:.3e}  κxy={closed[key]['kxy']:.3e}  "
                  f"σxx={closed[key]['sxx']:.3e}  σyy={closed[key]['syy']:.3e}  "
                  f"σxy={closed[key]['sxy']:.3e}"
                  + (f"  w*={closed[key]['w_star']:.3e}" if src == "op" else ""))
    Fop = assemble(d_pin, "op", npts=15)
    Fbf = assemble(d_pin, "bfs", npts=15)      # 公平对照：BFS 亦以 PINN 节点自由度为输入

    # ---------- 跨工况泛化 ----------
    print("\n===== 跨工况泛化（30 个未见载荷工况）=====")
    rng = np.random.default_rng(11)
    qs_test = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    gen = {"w": [], "w_star": [], "k": [], "s": []}
    for qv in qs_test:
        d = np.zeros_like(d_ex)
        wq = w_analytic(Xnr, Ynr, qv); wxq, wyq = _w_grad(Xnr, Ynr, qv)
        kxyq = curv_analytic(Xnr, Ynr, qv)[2]
        dq = np.stack([wq, wxq, wyq, -kxyq], axis=1).ravel()
        Fq = assemble(dq, "op", npts=11)
        XR, YR = Fq["X"].ravel(), Fq["Y"].ravel()
        kx_t, ky_t, kxy_t = curv_analytic(XR, YR, qv)
        sx_t, sy_t, sxy_t = stress_analytic(XR, YR, 1.0, qv)
        gen["w"].append(l2rel(Fq["W"].ravel(), w_analytic(XR, YR, qv)))
        # 位移物理解码（动作 B）
        w_star, x_u = wstar_from_field(Fq, npts=11)
        XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
        gen["w_star"].append(l2rel(w_star.ravel(),
                                   w_analytic(XU.ravel(), YU.ravel(), qv)))
        gen["k"].append(l2rel(np.concatenate([Fq["KX"].ravel(), Fq["KY"].ravel(),
                                              Fq["KXY"].ravel()]),
                              np.concatenate([kx_t, ky_t, kxy_t])))
        gen["s"].append(l2rel(np.concatenate([Fq["SX"].ravel(), Fq["SY"].ravel(),
                                              Fq["SXY"].ravel()]),
                              np.concatenate([sx_t, sy_t, sxy_t])))
        del d
    print(f"[泛化] 30 工况: w={np.mean(gen['w']):.2e}±{np.std(gen['w']):.1e}  "
          f"w*(重构)={np.mean(gen['w_star']):.2e}±{np.std(gen['w_star']):.1e}  "
          f"κ={np.mean(gen['k']):.2e}±{np.std(gen['k']):.1e}  "
          f"σ={np.mean(gen['s']):.2e}±{np.std(gen['s']):.1e}")

    # ---------- 绘图 ----------
    print("\n[绘图] 结果图 ...")
    # (5) PINN 损失：Deep Ritz 势能会跨零，故不能用 |L| 画 log，会形成"反单调"假象；
    # 改为线性主轴 + 残差 log 副轴，并标出过零点。
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    h = np.asarray(hist_pinn).astype(float)
    L_final = h[-1]
    ax.plot(np.arange(len(h)), h, color="#1565C0", lw=1.4, label=r"$\mathcal{L}$（线性）")
    ax.axhline(0.0, color="#888", lw=0.8, ls="--", label="零线")
    sign_change = int(np.argmax(np.cumsum(h < 0) > 0)) if (h < 0).any() else None
    if sign_change is not None and sign_change > 0:
        ax.axvline(sign_change, color="#C62828", lw=0.8, ls=":",
                   label=f"过零点 iter {sign_change}")
    ax.set_xlabel("epoch（Adam）"); ax.set_ylabel(r"$\mathcal{L}$（线性）")
    ax.set_title("级 I：能量型 PINN 训练损失\n（线性主轴，跨零并收敛到小负值）", fontsize=10)
    ax.grid(alpha=.3); ax.legend(loc="upper right", fontsize=9)
    ax2 = ax.twinx()
    ax2.semilogy(np.arange(len(h)), np.abs(h - L_final) + 1e-16,
                 color="#2E7D32", lw=1.0, alpha=.6,
                 label=r"$|\mathcal{L}-\mathcal{L}_{\mathrm{final}}|$（对数）")
    ax2.set_ylabel(r"$|\mathcal{L}-\mathcal{L}_{\mathrm{final}}|$（对数）", color="#2E7D32")
    ax2.tick_params(axis='y', labelcolor="#2E7D32")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{RES}/fig5_pinn_loss.png", dpi=150); plt.close(fig)

    # (6) 位移场 2×2
    WP = w_pin.reshape(ng, ng); WE = w_ex_grid.reshape(ng, ng)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    vmax = float(WE.max())
    im = axes[0, 0].contourf(xg, yg, WE.T, levels=24, cmap="viridis")
    fig.colorbar(im, ax=axes[0, 0]); axes[0, 0].set_title("(a) 解析解 $w^*(x,y)$")
    im = axes[0, 1].contourf(xg, yg, WP.T, levels=24, cmap="viridis")
    fig.colorbar(im, ax=axes[0, 1]); axes[0, 1].set_title("(b) PINN $\\hat w(x,y;q{=}1)$")
    im = axes[1, 0].contourf(xg, yg, np.abs(WP - WE).T, levels=24, cmap="magma")
    fig.colorbar(im, ax=axes[1, 0]); axes[1, 0].set_title("(c) 绝对误差 $|\\hat w-w^*|$")
    for a in axes.ravel():
        a.set_xlabel("$x$"); a.set_ylabel("$y$"); a.set_aspect("equal")
    # (d) 中心线剖面对比
    yc_idx = ng // 2
    axes[1, 1].plot(xg, WE[:, yc_idx], "k-", lw=2, label="解析解")
    axes[1, 1].plot(xg, WP[:, yc_idx], "r--", lw=1.8, label="PINN")
    # FEM 节点值
    d_fem_w = d_fem.reshape(-1, 4)[:, 0]
    Xf = Xnr; Yf = Ynr
    m = np.isclose(Yf, 0.5)
    axes[1, 1].plot(Xf[m], d_fem_w[m], "bs", ms=5, label="FEM 节点 (4×4 BFS)")
    axes[1, 1].plot(Xnr[m], d_pin.reshape(-1, 4)[:, 0][m], "ro", ms=4.5, mfc="none",
                    label="PINN 节点")
    axes[1, 1].set_xlabel("$x$"); axes[1, 1].set_ylabel("$w$")
    axes[1, 1].set_title("(d) 中心线 $y=0.5$ 挠度对比")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)
    fig.suptitle("级 I：位移场精度（$q=1$）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(f"{RES}/fig6_displacement_field.png", dpi=150); plt.close(fig)

    # (7) 单元内场恢复（沿 y=0.5 线）
    XR, YR = Fop["X"].ravel(), Fop["Y"].ravel()
    line = np.isclose(YR, np.round(YR, 6)) & (np.abs(YR - 0.5) < 1e-8)
    if line.sum() < 5:      # 退化处理：取最接近 y=0.5 的一行
        uy = np.unique(np.round(YR, 6))
        y0 = uy[np.argmin(np.abs(uy - 0.5))]
        line = np.isclose(np.round(YR, 6), y0)
    order = np.argsort(XR[line]); XL = XR[line][order]
    w_t = w_analytic(XR, YR, Q_TEST); kx_t, ky_t, kxy_t = curv_analytic(XR, YR, Q_TEST)
    mx_t = D_FLEX * (kx_t + NU * ky_t); my_t = D_FLEX * (ky_t + NU * kx_t)
    mxy_t = D_FLEX * (1 - NU) * kxy_t
    # 物理解码位移 w*：对算子曲率场 Poisson 重构后取中心线 y=0.5 切片
    w_star_g, x_un = wstar_from_field(Fop, npts=15)
    mid_u = int(np.argmin(np.abs(x_un - 0.5)))
    w_star_line = w_star_g[:, mid_u]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    items = [("(a) 位移 $w$", w_t, Fbf["W"].ravel(), Fop["W"].ravel()),
             (r"(b) 弯矩 $M_x$", mx_t,
              D_FLEX * (Fbf["KX"].ravel() + NU * Fbf["KY"].ravel()),
              D_FLEX * (Fop["KX"].ravel() + NU * Fop["KY"].ravel())),
             (r"(c) 弯矩 $M_y$", my_t,
              D_FLEX * (Fbf["KY"].ravel() + NU * Fbf["KX"].ravel()),
              D_FLEX * (Fop["KY"].ravel() + NU * Fop["KX"].ravel())),
             (r"(d) 扭矩 $M_{xy}$", mxy_t,
              D_FLEX * (1 - NU) * Fbf["KXY"].ravel(),
              D_FLEX * (1 - NU) * Fop["KXY"].ravel())]
    for kk, (ax, (tt, ye, yh, yo)) in enumerate(zip(axes.ravel(), items)):
        ax.plot(XL, ye[line][order], "k-", lw=2, label="解析解")
        ax.plot(XL, yh[line][order], "b--", lw=1.5, marker="s", ms=3, markevery=6,
                label="BFS 形函数插值")
        if kk == 0:
            ax.plot(x_un, w_star_line, "g-", lw=2, marker="^", ms=3, markevery=6,
                    label="神经算子（解码 $w^*$）")
            ax.plot(XL, yo[line][order], "r-.", lw=1.2, marker="o", ms=2.5,
                    markevery=8, label="神经算子（位移头）")
        else:
            ax.plot(XL, yo[line][order], "r-.", lw=1.8, marker="o", ms=3,
                    markevery=6, label="神经算子")
        ax.set_xlabel("$x$"); ax.set_ylabel(tt.split(" ", 1)[1]); ax.legend(fontsize=8.5)
        ax.grid(alpha=.3); ax.set_title(tt)
    fig.suptitle("级 II：单元内场恢复对比（$q=1$，沿 $y=0.5$，闭环：PINN 节点自由度驱动）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(f"{RES}/fig7_field_recovery.png", dpi=150); plt.close(fig)

    # (8) 算子损失
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for k, c in [("L1", "#1565C0"), ("L2", "#00838F"), ("L3", "#2E7D32"),
                 ("L4", "#6A1B9A"), ("L5", "#C62828"), ("L6", "#E65100")]:
        ax.semilogy(np.abs(np.array(hist_op[k])) + 1e-18, color=c, lw=1.2,
                    label=k)
    ax.set_xlabel("epoch"); ax.set_ylabel("损失（对数）")
    ax.set_title("级 II：DeepONet 训练损失分量（①位移 ②曲率 ③角点 ④应力 ⑤一致性 ⑥曲率-位移自洽）")
    ax.legend(ncol=3, fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{RES}/fig8_operator_loss.png", dpi=150); plt.close(fig)

    # (9) 泛化（位移用物理解码 w*；exact 组=本轮内存 gen，端到端组=results.json percase_pinn）
    e2e = None
    try:
        with open(f"{RES}/results.json", encoding="utf-8") as _f:
            _g30 = json.load(_f).get("supp_review", {}).get("gen30_pinn_dof", {})
        _pp = _g30.get("percase_pinn")
        if _pp and len(_pp["w_star"]) == len(gen["w_star"]):
            e2e = dict(w_star=np.array(_pp["w_star"]), k=np.array(_pp["k"]),
                       s=np.array(_pp["s"]))
    except Exception:
        e2e = None
    ex = dict(w_star=np.array(gen["w_star"]), k=np.array(gen["k"]), s=np.array(gen["s"]))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    groups = [("解码位移 $w^\\ast$", ex["w_star"], e2e["w_star"] if e2e else None),
              ("曲率 $\\kappa$", ex["k"], e2e["k"] if e2e else None),
              ("应力 $\\sigma$", ex["s"], e2e["s"] if e2e else None)]
    xg = np.arange(3); wbar = 0.34
    c1, c2 = "#1565C0", "#E65100"
    for i, (name, m1, m2) in enumerate(groups):
        ax.bar(xg[i] - wbar / 2, m1.mean(), yerr=m1.std(), width=wbar, color=c1,
               alpha=.9, capsize=4, label="解析节点自由度" if i == 0 else None)
        cv1 = m1.std() / m1.mean() * 100
        ax.text(xg[i] - wbar / 2, (m1.mean() + m1.std()) * 1.5, f"CV {cv1:.0f}%",
                ha="center", fontsize=8)
        if m2 is not None:
            ax.bar(xg[i] + wbar / 2, m2.mean(), yerr=m2.std(), width=wbar, color=c2,
                   alpha=.9, capsize=4, label="端到端（PINN 节点自由度）" if i == 0 else None)
            cv2 = m2.std() / m2.mean() * 100
            ax.text(xg[i] + wbar / 2, (m2.mean() + m2.std()) * 1.5, f"CV {cv2:.0f}%",
                    ha="center", fontsize=8)
    ax.set_xticks(xg); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("$L_2$ 相对误差（对数）"); ax.set_yscale("log")
    ax.set_title("级 II：30 个未见载荷工况泛化误差（均值±标准差）")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{RES}/fig9_generalization.png", dpi=150); plt.close(fig)

    # (10) 应力奇偶图
    sx_t, sy_t, sxy_t = stress_analytic(XR, YR, 1.0, Q_TEST)
    vm_t = mises(sx_t, sy_t, sxy_t)
    vm_op = mises(Fop["SX"].ravel(), Fop["SY"].ravel(), Fop["SXY"].ravel())
    vm_bf = mises(Fbf["SX"].ravel(), Fbf["SY"].ravel(), Fbf["SXY"].ravel())
    fig, ax = plt.subplots(figsize=(6.9, 6.3))
    ax.scatter(vm_t, vm_op, s=8, c="#6A1B9A", alpha=.5, label="神经算子")
    ax.scatter(vm_t, vm_bf, s=8, c="#1565C0", alpha=.5, marker="s", label="BFS 插值")
    lim = [0.0, max(vm_t.max(), vm_op.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1.5, label="理想一致")
    ax.set_xlabel("解析 von Mises 应力 $\\sigma_{vm}^*$")
    ax.set_ylabel("预测 von Mises 应力 $\\hat\\sigma_{vm}$")
    ax.set_title("应力奇偶图（$q=1$，上表面 $\\zeta=+1$）")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{RES}/fig10_stress_parity.png", dpi=150); plt.close(fig)

    # (11) von Mises 云图
    n1 = Fop["W"].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    VM_T = vm_t.reshape(n1, n1); VM_O = vm_op.reshape(n1, n1); VM_B = vm_bf.reshape(n1, n1)
    for ax, Z, tt in [(axes[0], VM_T, "(a) 解析解"),
                      (axes[1], VM_O, "(b) 神经算子（PINN 驱动）"),
                      (axes[2], np.abs(VM_O - VM_T), "(c) 绝对误差")]:
        im = ax.contourf(Fop["X"], Fop["Y"], Z, levels=26, cmap="plasma")
        fig.colorbar(im, ax=ax); ax.set_title(tt)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_aspect("equal")
    fig.suptitle("上表面 von Mises 应力云图（$q=1$，$\\zeta=+1$）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(f"{RES}/fig11_mises_field.png", dpi=150); plt.close(fig)

    # ---------- 汇总 ----------
    report = {
        "geom": {"a": A, "b": B, "D": D_FLEX, "nu": NU, "h": H_TH, "mesh": f"{n}x{n}"},
        "analytic_check": {"w_center": wc, "Mx_center": float(mx_c)},
        "fem_node_max_err": e_fem_node,
        "pinn": {"l2": e_pinn_L2, "max_abs": e_pinn_max,
                 "node_w_max": e_pinn_node_w, "node_wx_max": e_pinn_node_gx,
                 "node_wy_max": e_pinn_node_gy,
                 "w_center": float(w_pin[ng * ng // 2]),
                 "w_center_exact": float(w_ex_grid[ng * ng // 2]),
                 "loss_final": float(np.abs(hist_pinn[-1])),
                 "epochs": int(args.ep_pinn)},
        "operator": {"n_train_cases": int(len(qs_train)),
                     "n_samples": int(len(samples)),
                     "epochs": int(args.ep_op),
                     "loss_final": {k: float(np.abs(v[-1])) for k, v in hist_op.items()}},
        "closed_loop": closed,
        "generalization": {"n_cases": int(len(qs_test)),
                           "w_mean": float(np.mean(gen["w"])), "w_std": float(np.std(gen["w"])),
                           "w_star_mean": float(np.mean(gen["w_star"])),
                           "w_star_std": float(np.std(gen["w_star"])),
                           "k_mean": float(np.mean(gen["k"])), "k_std": float(np.std(gen["k"])),
                           "s_mean": float(np.mean(gen["s"])), "s_std": float(np.std(gen["s"]))},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{RES}/results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(f"{RES}/error_report.txt", "w", encoding="utf-8") as f:
        f.write("四边简支矩形薄板两级神经网络替代体系 —— 数值验证误差报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"解析解自检  中心挠度={wc:.6e} (理论4.062e-3)  中心弯矩Mx={mx_c:.6e} (理论4.789e-2)\n")
        f.write(f"FEM(4x4 BFS) 节点挠度最大误差   : {e_fem_node:.3e}\n")
        f.write(f"PINN 全场位移 L2 相对误差       : {e_pinn_L2:.3e}\n")
        f.write(f"PINN 全场位移 max|Δw|           : {e_pinn_max:.3e}\n")
        f.write(f"PINN 节点 max|Δw|               : {e_pinn_node_w:.3e}\n")
        f.write(f"PINN 节点 max|Δw_x| / |Δw_y|    : {e_pinn_node_gx:.3e} / {e_pinn_node_gy:.3e}\n")
        f.write("-" * 70 + "\n")
        for k, v in closed.items():
            f.write(f"[{k:10s}] w={v['w']:.3e} κxx={v['kxx']:.3e} κyy={v['kyy']:.3e} "
                    f"κxy={v['kxy']:.3e} σxx={v['sxx']:.3e} σyy={v['syy']:.3e} σxy={v['sxy']:.3e}\n")
        f.write("-" * 70 + "\n")
        f.write(f"[泛化 {len(qs_test)}工况] w={np.mean(gen['w']):.2e}±{np.std(gen['w']):.1e}  "
                f"κ={np.mean(gen['k']):.2e}±{np.std(gen['k']):.1e}  "
                f"σ={np.mean(gen['s']):.2e}±{np.std(gen['s']):.1e}\n")
        f.write("=" * 70 + "\n")
    print(f"\n[完成] 耗时 {report['elapsed_s']} s；结果写入 {RES}/")


if __name__ == "__main__":
    main()
