# -*- coding: utf-8 -*-
"""
fe_extra.py — 通用 BFS 板 FE 求解（扩展 fem_solve：载荷分布可选、边界可选）
仅供评审补充实验（patch/clamped），不改 shell_system 默认行为。
"""
import numpy as np
import shell_system as S

A, B, D_FLEX, NU = S.A, S.B, S.D_FLEX, S.NU
_GP, _GW = np.polynomial.legendre.leggauss(4)


def load_uniform(x, y, q=1.0):
    return np.full(np.shape(x), q)


def load_center_patch(x, y, q=1.0, half=0.25):
    """中心 [0.5-half, 0.5+half]^2 内的均布 patch，其余 0"""
    m = (np.abs(x - 0.5) <= half) & (np.abs(y - 0.5) <= half)
    return np.where(m, q, 0.0)


def fem_solve2(n, qfun, bc="ss", q=1.0):
    """n×n BFS 单元；载荷 qfun(x,y)(展平) 或 'uniform'/'patch' 简写；bc: 'ss'|'clamped'
    返回 节点自由度 d(4(n+1)²,)，节点坐标 xs, ys"""
    Lex, Ley = A / n, B / n
    nnode = (n + 1) * (n + 1)
    ndof = 4 * nnode
    if callable(qfun):
        pass
    elif qfun == "uniform":
        qfun = lambda x, y: load_uniform(x, y, q)
    elif qfun == "patch":
        qfun = lambda x, y: load_center_patch(x, y, q)
    else:
        raise ValueError(qfun)

    Dm = D_FLEX * np.array([[1.0, NU, 0.0], [NU, 1.0, 0.0],
                            [0.0, 0.0, (1.0 - NU) / 2.0]])
    K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
    for ei in range(n):
        for ej in range(n):
            Ke = np.zeros((16, 16)); fe = np.zeros(16)
            for gp, gw in zip(_GP, _GW):
                for gq2, gv in zip(_GP, _GW):
                    x = (ei + 0.5 * (gp + 1.0)) * Lex
                    y = (ej + 0.5 * (gq2 + 1.0)) * Ley
                    qv = qfun(np.array([x]), np.array([y]))[0]
                    wJ = 0.25 * gw * gv * Lex * Ley
                    t = 0.5 * (gp + 1.0); s = 0.5 * (gq2 + 1.0)
                    N, d2x, d2y, dxy = S.bfs_shape(np.array([t]), np.array([s]),
                                                   Lex, Ley)
                    Bm = np.vstack([d2x, d2y, 2.0 * dxy])
                    Ke += wJ * (Bm.T @ Dm @ Bm)
                    fe += wJ * qv * N[0]
            ids = np.array([4 * ((ei + di) * (n + 1) + (ej + dj)) + k
                            for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)]
                            for k in range(4)])
            K[np.ix_(ids, ids)] += Ke
            F[ids] += fe

    fixed = set()
    for i in range(n + 1):
        for j in range(n + 1):
            nid = i * (n + 1) + j
            on_x = (i == 0 or i == n)          # x=0/a 边
            on_y = (j == 0 or j == n)          # y=0/b 边
            if on_x or on_y:
                fixed.add(4 * nid + 0)         # w = 0
            if bc == "ss":
                if on_x:
                    fixed.add(4 * nid + 2)     # wy = 0（w≡0 → 切向导数 0）
                if on_y:
                    fixed.add(4 * nid + 1)     # wx = 0
            elif bc == "clamped":
                if on_x or on_y:
                    fixed.add(4 * nid + 1)
                    fixed.add(4 * nid + 2)
            else:
                raise ValueError(bc)
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)
    d = np.zeros(ndof)
    d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    xs = np.linspace(0, A, n + 1); ys = np.linspace(0, B, n + 1)
    return d, xs, ys


def coarse_nodes_of_fine(n_c, n_f):
    """细网格节点序号 (i*(n_f+1)+j) 的子集 = 粗节点 (i0,j0) ∈0..n_c"""
    st = n_f // n_c
    assert n_f % n_c == 0
    out = []
    for i0 in range(n_c + 1):
        for j0 in range(n_c + 1):
            out.append(i0 * st * (n_f + 1) + j0 * st)
    return np.array(out)
