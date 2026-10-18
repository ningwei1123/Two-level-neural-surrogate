# -*- coding: utf-8 -*-
"""
rev_scale2.py — R3 #4 计时复核：16×16 FE 计时非单调疑点的严格重测
==================================================================
口径（同机、每工况、warm-up 后 7 次中位数）：
  fe_dense_fresh   : Python 稠密 BFS 组装+求解（当前正文基准；每次全新）
  fe_dense_fact    : 稠密 LU 分解一次/网格，之后逐载荷回代（许多工况场景）
  fe_sparse_fact   : scipy SuperLU 分解一次/网格，之后逐载荷回代（生产型口径）
  surrogate        : 级I 节点 DOF 求值 + 级II 批量单元前向(npts=11) +
                     预分解 Poisson 回代（全推理链）
模型：pinn_state.pt（级I）；op_state.pt / op_state_g8 / op_state_g16（级II按网格重训）
落盘 supp_review.scaling2。
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "..", "results")

import numpy as np
import torch
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import shell_system as S
import rev_b
from rev_common import load_json, save_json

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)


def fem_dense_fresh(n, q):
    """与正文基准一致的逐载荷稠密组装+求解"""
    return S.fem_solve(n, q)


def build_system(n, q):
    """返回 (Kcsr(free,free), Ffree, free, ndof, nnode)"""
    Lex, Ley = S.A / n, S.B / n
    nnode = (n + 1) * (n + 1); ndof = 4 * nnode
    K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
    Ke, fe = S.bfs_element(Lex, Ley, q)
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
            if i == 0 or i == n:
                fixed.update([4 * nid + 0, 4 * nid + 2])
            if j == 0 or j == n:
                fixed.update([4 * nid + 0, 4 * nid + 1])
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)
    Kf = sp.csr_matrix(K[np.ix_(free, free)])
    return Kf, F[free], free, ndof


def poisson_factor(M):
    """五点 ∇²w=-(κxx+κyy)，Dirichlet 0，返回 (splu, M)"""
    h = 1.0 / (M - 1)
    N = (M - 2) ** 2
    rows, cols, vals = [], [], []
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
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return spla.splu(A.tocsc()), h


def poisson_solve(factor, h, kxx, kyy):
    M = kxx.shape[0]
    s = -(kxx + kyy)
    rhs = (s[1:M - 1, 1:M - 1] * h * h).ravel()
    W = np.zeros((M, M))
    W[1:M - 1, 1:M - 1] = factor.solve(rhs).reshape(M - 2, M - 2)
    return W


def time_med(fn, repeat=7, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    pinn.eval()

    grids = {"4": "op_state.pt", "8": "op_state_g8.pt", "16": "op_state_g16.pt"}
    out = {"note": ("Same-machine median wall-clock per load case (warm-up then 7 "
                    "repeats). FE dense-fresh = Python BFS assembly+dense solve each "
                    "load (previous baseline). dense-fact = dense LU factorised once "
                    "per grid then per-load back-substitution. sparse-fact = scipy "
                    "SuperLU factorised once per grid then per-load solve (production-"
                    "type sparse). surrogate = Level-I nodal DOF evaluation + batched "
                    "element operator forward (npts=11) + pre-factorised Poisson "
                    "back-substitution.")}
    for n_s, fname in grids.items():
        n = int(n_s)
        if not os.path.exists(os.path.join(RES, fname)):
            out[n_s] = {"missing": fname}
            print(f"[{n_s}] skip: {fname} missing")
            continue
        op = S.PlateOperator(p=64, hidden=64)
        op.load_state_dict(torch.load(os.path.join(RES, fname),
                                      weights_only=False))
        op.eval()

        Kf, Fr, free, ndof = build_system(n, 1.0)
        # dense factor once
        Kd = Kf.toarray()
        lu = sla.lu_factor(Kd)
        # sparse factor once
        lu_s = spla.splu(Kf.tocsc())

        # Poisson 预分解（解码用，唯一坐标网格）
        keep = rev_b.unique_idx(n, 11)
        M = keep.size
        pfac, ph = poisson_factor(M)

        # ---- 计时 ----
        t_fresh = time_med(lambda: fem_dense_fresh(n, 1.0))
        t_df = time_med(lambda: sla.lu_solve(lu, Fr))
        t_sf = time_med(lambda: lu_s.solve(Fr))
        dp = rev_b.pinn_dof_grid(pinn, 1.0, n)

        def surr():
            Ff = rev_b.assemble_op_batch(dp, op, n, npts=11)
            kx = Ff["KX"][np.ix_(keep, keep)]; ky = Ff["KY"][np.ix_(keep, keep)]
            poisson_solve(pfac, ph, kx, ky)

        t_sur = time_med(surr)
        row = {
            "free_dof": int(len(free)), "n_elem": int(n * n),
            "fe_dense_fresh_ms": round(t_fresh * 1e3, 3),
            "fe_dense_fact_ms": round(t_df * 1e3, 3),
            "fe_sparse_fact_ms": round(t_sf * 1e3, 3),
            "surrogate_ms": round(t_sur * 1e3, 3),
            "speedup_vs_dense_fresh": round(t_fresh / max(t_sur, 1e-12), 2),
            "speedup_vs_dense_fact": round(t_df / max(t_sur, 1e-12), 2),
            "speedup_vs_sparse_fact": round(t_sf / max(t_sur, 1e-12), 2),
        }
        out[n_s] = row
        print(f"[{n_s}] free_dof={row['free_dof']} "
              f"FE fresh={row['fe_dense_fresh_ms']}ms "
              f"dense-fact={row['fe_dense_fact_ms']}ms "
              f"sparse-fact={row['fe_sparse_fact_ms']}ms "
              f"surrogate={row['surrogate_ms']}ms "
              f"(x{row['speedup_vs_sparse_fact']} vs sparse)")
    R = load_json()
    R.setdefault("supp_review", {})["scaling2"] = {
        "note": out["note"],
        "grids": {k: v for k, v in out.items() if k != "note"}}
    save_json(R)
    print("saved supp_review.scaling2")


if __name__ == "__main__":
    main()
