# -*- coding: utf-8 -*-
"""
rev_common.py — 大修实验公共工具（P1-A/B/C/D）
仅依赖 shell_system 提供的基础解/单元/网络；不在核心文件里改任何默认行为。
"""
import os
import json
import time
import numpy as np
import torch
import shell_system as S

RES = S.RES
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
A, B, D_FLEX, NU, H_TH = S.A, S.B, S.D_FLEX, S.NU, S.H_TH


def coarse_dof(q, n=4):
    """n×n BFS FE 解 → 节点 DOF 数组 ((n+1)²,4)，与 _assemble 输入布局一致"""
    d, xs, ys = S.fem_solve(n, q)
    return d.reshape(-1, 4)


def exact_dof_arr(q, n=4):
    xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
    Xr, Yr = np.meshgrid(xs, ys, indexing="ij")
    Xr, Yr = Xr.ravel(), Yr.ravel()
    wq = S.w_analytic(Xr, Yr, q); wxq, wyq = S._w_grad(Xr, Yr, q)
    kxyq = S.curv_analytic(Xr, Yr, q)[2]
    return np.stack([wq, wxq, wyq, -kxyq], axis=1)


def fe_field_grid(nf, d, X, Y, zeta=None):
    """任意 (X,Y)（展平数组）上求细网格 FE 场 (w, kx,ky,kxy, sx,sy,sxy)。
    X,Y 全域坐标 ∈[0,1]；zeta 与 X 同形（默认 +1 顶面）。
    基于 bfs_shape 单点化；自动定位所属细单元并取 16 DOF。"""
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    flat = X.ndim == 1
    if not flat:
        shp = X.shape
        X = X.ravel(); Y = Y.ravel()
    Lex = A / nf; Ley = B / nf
    ii = np.clip(np.floor(X / Lex), 0, nf - 1).astype(np.int64)
    jj = np.clip(np.floor(Y / Ley), 0, nf - 1).astype(np.int64)
    t = np.clip((X - ii * Lex) / Lex, 0.0, 1.0)
    s = np.clip((Y - jj * Ley) / Ley, 0.0, 1.0)
    base = ii * (nf + 1) + jj
    N = X.size
    de = np.zeros((N, 16))
    corner = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for ci, (di, dj) in enumerate(corner):
        nid = base + di * (nf + 1) + dj            # (N,)
        for k in range(4):
            de[:, ci * 4 + k] = d[4 * nid + k]
    Ns, d2x, d2y, dxy = S.bfs_shape(t, s, Lex, Ley)
    w = np.einsum("ij,ij->i", Ns, de)            # (N,)
    kx = -np.einsum("ij,ij->i", d2x, de)
    ky = -np.einsum("ij,ij->i", d2y, de)
    kxy = -np.einsum("ij,ij->i", dxy, de)
    mx = D_FLEX * (kx + NU * ky); my = D_FLEX * (ky + NU * kx)
    mxy = D_FLEX * (1.0 - NU) * kxy
    z = np.ones(N) if zeta is None else np.asarray(zeta, dtype=np.float64)
    f = 6.0 * z / H_TH ** 2
    sx = f * mx; sy = f * my; sxy = f * mxy
    out = (w, kx, ky, kxy, sx, sy, sxy)
    if not flat:
        out = tuple(v.reshape(shp) for v in out)
    return out


def fe_ref_field(nf, q, X, Y, zeta=None):
    d, _, _ = S.fem_solve(nf, q)
    return fe_field_grid(nf, d, X, Y, zeta)


def assemble_bfs(d_all, n=4, npts=15):
    """BFS 插值后处理：与 eval_symavg._assemble 同布局（供 OP vs BFS 同输入对比）"""
    n1 = npts
    W = np.zeros((n * n1, n * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W); WX = np.zeros_like(W); WY = np.zeros_like(W)
    Lex = A / n
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    for ei in range(n):
        for ej in range(n):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d_all.reshape(-1, 4)[ids].ravel()
            w, kxx, kyy, kxy = S.bfs_interp(de, pts[:, 0], pts[:, 1], Lex, Lex)
            mx = D_FLEX * (kxx + NU * kyy); my = D_FLEX * (kyy + NU * kxx)
            mxy = D_FLEX * (1 - NU) * kxy
            f = 6.0 / H_TH ** 2
            sl_i = slice(ei * n1, (ei + 1) * n1); sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = w.reshape(n1, n1)
            KX[sl_i, sl_j] = kxx.reshape(n1, n1); KY[sl_i, sl_j] = kyy.reshape(n1, n1)
            KXY[sl_i, sl_j] = kxy.reshape(n1, n1)
            SX[sl_i, sl_j] = (f * mx).reshape(n1, n1)
            SY[sl_i, sl_j] = (f * my).reshape(n1, n1)
            SXY[sl_i, sl_j] = (f * mxy).reshape(n1, n1)
            xg = xs_g(ei, n, n1, pts[:, 0]); yg = xs_g(ej, n, n1, pts[:, 1])
            WX[sl_i, sl_j] = xg.reshape(n1, n1); WY[sl_i, sl_j] = yg.reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)


def xs_g(ei, n, n1, xi):
    Lex = A / n
    return (ei * Lex) + 0.5 * Lex * (xi + 1.0)


def l2rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def field_metrics_truth(F, truth, npts=15, decode=True):
    """truth: dict(X,Y 同 F 坐标的 κ/σ 真值及 'WFE' 唯一网格 w) 或 'analytic'
    返回 dict(w_star, k, s, w_direct)"""
    from eval_poisson import _squeeze
    G, xq, _ = _squeeze(F, npts)
    Xr, Yr = F["X"].ravel(), F["Y"].ravel()
    if truth == "analytic":
        raise ValueError("use analytic_metrics instead")
    kx_t = truth["kx"](Xr, Yr); ky_t = truth["ky"](Xr, Yr); kxy_t = truth["kxy"](Xr, Yr)
    sx_t = truth["sx"](Xr, Yr); sy_t = truth["sy"](Xr, Yr); sxy_t = truth["sxy"](Xr, Yr)
    k = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
              np.concatenate([kx_t, ky_t, kxy_t]))
    s = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
              np.concatenate([sx_t, sy_t, sxy_t]))
    out = {"k": k, "s": s}
    if decode:
        w_star, x_u = S.wstar_from_field(F, npts=npts)
        XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
        wt = truth["w"](XU.ravel(), YU.ravel()).reshape(XU.shape)
        out["w_star"] = l2rel(w_star.ravel(), wt.ravel())
    return out


def analytic_metrics(F, q, npts=15, decode=True):
    Xr, Yr = F["X"].ravel(), F["Y"].ravel()
    kx_t, ky_t, kxy_t = S.curv_analytic(Xr, Yr, q)
    sx_t, sy_t, sxy_t = S.stress_analytic(Xr, Yr, 1.0, q)
    out = {"k": l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
                      np.concatenate([kx_t, ky_t, kxy_t])),
           "s": l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
                      np.concatenate([sx_t, sy_t, sxy_t]))}
    if decode:
        w_star, x_u = S.wstar_from_field(F, npts=npts)
        XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
        out["w_star"] = l2rel(w_star.ravel(), S.w_analytic(XU.ravel(), YU.ravel(), q))
    return out


def mxy_center_max(F):
    KXY = F["KXY"]
    return float(np.abs(KXY[KXY.shape[0] // 2 - 2: KXY.shape[0] // 2 + 3, :]).max())


def interface_jump(F, npts=15):
    """全场沿内部单元边界的最大跳变：相邻单元在同一边界坐标上的输出差最大值"""
    keys = [k for k in ("KX", "KXY", "SX", "SXY") if k in F]
    out = {}
    for key in keys:
        M = F[key]
        ntot = M.shape[0]
        n_el = ntot // npts
        assert n_el * npts == ntot and n_el > 1, (ntot, npts)
        jmax = 0.0
        for e in range(1, n_el):            # 内部 x 边界（行）
            ra = e * npts - 1; rb = e * npts
            jmax = max(jmax, float(np.abs(M[ra, :] - M[rb, :]).max()))
        for e in range(1, n_el):            # 内部 y 边界（列）
            ca = e * npts - 1; cb = e * npts
            jmax = max(jmax, float(np.abs(M[:, ca] - M[:, cb]).max()))
        out[key] = jmax
    return out


def load_json():
    rj = os.path.join(RES, "results.json")
    return json.load(open(rj, encoding="utf-8"))


def save_json(R):
    rj = os.path.join(RES, "results.json")
    json.dump(R, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
