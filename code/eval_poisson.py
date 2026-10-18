# -*- coding: utf-8 -*-
"""
eval_poisson.py — 动作 B 原型：位移由曲率经 Poisson 重构 (w* = ∇⁻²(κ_xx+κ_yy))
===============================================================================
Kirchhoff–Love 板：κ_xx = -w_xx, κ_yy = -w_yy → ∇²w = -(κ_xx + κ_yy)。
简支边界 w|∂Ω = 0 → 五点离散 Poisson 一次稀疏求解。

步骤：
  1) 自检：用解析 κ 重构 w，验证离散+BC 实现（应接近解析 w）
  2) 闭环：算子(PINN DOF 驱动)输出 κ → 重构 w*，与 BFS/原算子 w 头对比
  3) 泛化 30 工况：w* 的均值/CV vs 原 w 头（检验 CV 63% 是否下降）

依赖 eval_symavg._assemble 输出的规则场（每单元 npts 点，60×60）。
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import shell_system as S

RES = S.RES
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Q = S.Q_TEST


def poisson_reconstruct(kxx, kyy, h):
    """五点离散 ∇²w = -(κ_xx+κ_yy)，w|∂=0。
    kxx,kyy: (M,M) 规则网格（行沿 x、列沿 y 的 F 场布局）。
    返回 w* 同形状。M 内部点 (M-2)² 未知量。"""
    M = kxx.shape[0]
    s = -(kxx + kyy)                      # 右端 = -∇²w 的源
    N = (M - 2) ** 2
    # 内部点行主编号 r = (i-1)*(M-2)+(j-1)
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


def l2rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def _grid(F):
    """从场 F 恢复 (60,60) 规则网格坐标（行沿 x 增、列沿 y 增）"""
    return F["X"][:, 0], F["Y"][0, :]


def _squeeze(F, npts=15):
    """assemble 输出的场每单元含共享边界点（4×15=60 行），
    去除每单元末尾重复点 → 57×57 均匀网格（间距 1/56）。
    返回 (压缩场 dict, x 坐标 (57,), y 坐标 (57,))"""
    keep = []
    for ei in range(n):
        keep += list(range(ei * npts, ei * npts + npts - 1))
    keep += [n * npts - 1]                        # 末尾 x=1 保留
    keep = np.asarray(keep)
    G = {k: v[np.ix_(keep, keep)] if k not in ("X", "Y") else v
         for k, v in F.items()}
    x = np.linspace(0, S.A, 1 + n * (npts - 1))
    return G, x, x


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--dof", default="pinn", choices=["pinn", "exact"])
    args = ap.parse_args()

    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, f"op_state{args.tag}.pt"),
                                  weights_only=False))

    from eval_symavg import _assemble, pinn_field_all
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij"); Xnr, Ynr = Xnr.ravel(), Ynr.ravel()
    if args.dof == "pinn":
        pinn = _load_pinn()
        w0, wx0, wy0, wxy0 = pinn_field_all(pinn, Xnr, Ynr)
        d = np.stack([w0, wx0, wy0, wxy0], axis=1)
    else:
        wq = S.w_analytic(Xnr, Ynr, Q); wxq, wyq = S._w_grad(Xnr, Ynr, Q)
        kxyq = S.curv_analytic(Xnr, Ynr, Q)[2]
        d = np.stack([wq, wxq, wyq, -kxyq], axis=1)

    npts = 15
    Fraw = _assemble(d, op, scale=None, npts=npts)
    F, Xr, Yr = _squeeze(Fraw, npts)
    Xg, Yg = np.meshgrid(Xr, Yr, indexing="ij")
    h = Xr[1] - Xr[0]

    # ---- 0) 自检：解析 κ → Poisson → w* ----
    kx_t = S.curv_analytic(Xg.ravel(), Yg.ravel(), Q)[0].reshape(Xg.shape)
    ky_t = S.curv_analytic(Xg.ravel(), Yg.ravel(), Q)[1].reshape(Xg.shape)
    w_t = S.w_analytic(Xg.ravel(), Yg.ravel(), Q).reshape(Xg.shape)
    w_self = poisson_reconstruct(kx_t, ky_t, h)
    e_self = l2rel(w_self.ravel(), w_t.ravel())
    print(f"[自检] 解析 κ → Poisson 重构 w  L2 = {e_self:.3e}  (应 ~1e-4 量级，验证离散+BC)")

    # ---- 1) 闭环重构 ----
    w_star = poisson_reconstruct(F["KX"], F["KY"], h)
    w_op_head = F["W"]                       # 算子原 w 头
    e_star = l2rel(w_star.ravel(), w_t.ravel())
    e_head = l2rel(w_op_head.ravel(), w_t.ravel())
    print(f"\n[闭环 q=1, {args.dof} DOF]")
    print(f"  算子 w 头  : {e_head:.4e}")
    print(f"  Poisson 重构: {e_star:.4e}   （BFS exact 参考 1.77e-3）")

    # ---- 2) 泛化 30 工况 ----
    rng = np.random.default_rng(11)
    qs = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    gw = {"head": [], "star": []}
    for qv in qs:
        wq = S.w_analytic(Xnr, Ynr, qv); wxq, wyq = S._w_grad(Xnr, Ynr, qv)
        kxyq = S.curv_analytic(Xnr, Ynr, qv)[2]
        dq = np.stack([wq, wxq, wyq, -kxyq], axis=1)
        Fraw_q = _assemble(dq, op, scale=None, npts=11)
        Fq, Xrq, Yrq = _squeeze(Fraw_q, 11)
        Xgq, Ygq = np.meshgrid(Xrq, Yrq, indexing="ij")
        hq = Xrq[1] - Xrq[0]
        w_tq = S.w_analytic(Xgq.ravel(), Ygq.ravel(), qv).reshape(Xgq.shape)
        w_sq = poisson_reconstruct(Fq["KX"], Fq["KY"], hq)
        gw["head"].append(l2rel(Fq["W"].ravel(), w_tq.ravel()))
        gw["star"].append(l2rel(w_sq.ravel(), w_tq.ravel()))
    print(f"\n[泛化 30] w 头: {np.mean(gw['head']):.2e}±{np.std(gw['head']):.1e} "
          f"(CV {np.std(gw['head'])/np.mean(gw['head'])*100:.0f}%)")
    print(f"         重构 : {np.mean(gw['star']):.2e}±{np.std(gw['star']):.1e} "
          f"(CV {np.std(gw['star'])/np.mean(gw['star'])*100:.0f}%)")


def _load_pinn():
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    return pinn


if __name__ == "__main__":
    main()
