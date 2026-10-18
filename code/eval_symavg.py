# -*- coding: utf-8 -*-
"""
eval_symavg.py — 推理期板级对称平均 (test-time symmetry averaging)
=====================================================================
动作 A 的推理侧验证（research_plan_tier1.md S2，不改网络、不重训）：
对 PINN 节点自由度全场 d 施加板级对称变换 {e, Rx, Ry, R180}，
各自 assemble 后按场量变号规则搬回原坐标系并平均，检查：
  (1) Mxy/σxy 对称线伪影是否被对消（M1 打击点）
  (2) w/κ/σ 闭环 L2 是否优于单次推理
  (3) 泛化 30 工况 CV 是否下降

场量在板级反射下的对称性（基于 G3 已验证的 DOF 符号规律）：
  w, κ_xx, κ_yy, σ_xx, σ_yy : Rx/Ry/R180 均为偶（搬回不变号）
  κ_xy, σ_xy                : Rx、Ry 为奇（搬回变号）；R180 为偶
坐标搬移：assemble 输出矩阵行沿 x、列沿 y → Rx 行反向、Ry 列反向、R180 行列反向。
"""
import os
import json
import numpy as np
import torch
import shell_system as S
from sym_aug import _PX, _PY  # 复用块置换以保持约定一致

RES = S.RES
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH
Lex = S.A / n
Q = S.Q_TEST


# ---------------------------------------------------------------------------
# 板级 DOF 场变换：d_all((n+1)²,4)，节点索引 i*(n+1)+j 对应 (xs[i], ys[j])
# ---------------------------------------------------------------------------
_DOF_SIGN_RX = np.array([1.0, -1.0, 1.0, -1.0])   # w 偶, wx 奇, wy 偶, wxy 奇
_DOF_SIGN_RY = np.array([1.0, 1.0, -1.0, -1.0])
_DOF_SIGN_R180 = np.array([1.0, -1.0, -1.0, 1.0])


def dof_reflect_x(d_all):
    m = d_all.reshape(n + 1, n + 1, 4)
    return (m[::-1, :, :] * _DOF_SIGN_RX).reshape(-1, 4)


def dof_reflect_y(d_all):
    m = d_all.reshape(n + 1, n + 1, 4)
    return (m[:, ::-1, :] * _DOF_SIGN_RY).reshape(-1, 4)


def dof_rot180(d_all):
    m = d_all.reshape(n + 1, n + 1, 4)
    return (m[::-1, ::-1, :] * _DOF_SIGN_R180).reshape(-1, 4)


# ---------------------------------------------------------------------------
# 场量搬回符号：键序 (W,KX,KY,KXY,SX,SY,SXY)
# ---------------------------------------------------------------------------
_FIELD_SIGN = {
    "Rx": np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]),
    "Ry": np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]),
    "R180": np.ones(7),
}


def _sym_avg_field(d_all, op, scale=None, npts=15):
    """对 d_all 做 4 向板级变换 → assemble → 按符号搬回 → 平均。
    返回 (dict 场均, dict 单次场 e)。"""
    F0 = _assemble(d_all, op, scale=scale, npts=npts)
    cands = [("Rx", dof_reflect_x(d_all)), ("Ry", dof_reflect_y(d_all)),
             ("R180", dof_rot180(d_all))]
    fields = {"e": F0}
    for name, d2 in cands:
        F2 = _assemble(d2, op, scale=scale, npts=npts)
        # 坐标搬移
        if name == "Rx":
            F2 = {k: (v[::-1, :] if k != "X" and k != "Y" else v)
                  for k, v in F2.items()}
        elif name == "Ry":
            F2 = {k: (v[:, ::-1] if k != "X" and k != "Y" else v)
                  for k, v in F2.items()}
        else:
            F2 = {k: (v[::-1, ::-1] if k != "X" and k != "Y" else v)
                  for k, v in F2.items()}
        # 场量变号（X/Y 坐标矩阵不变号）
        sgn = _FIELD_SIGN[name]
        for j, k in enumerate(["W", "KX", "KY", "KXY", "SX", "SY", "SXY"]):
            F2[k] = F2[k] * sgn[j]
        fields[name] = F2

    avg = {}
    for k in ["W", "KX", "KY", "KXY", "SX", "SY", "SXY"]:
        avg[k] = (fields["e"][k] + fields["Rx"][k]
                  + fields["Ry"][k] + fields["R180"][k]) / 4.0
    avg["X"], avg["Y"] = F0["X"], F0["Y"]
    return avg, F0


def _assemble(d_all, op, scale=None, npts=15):
    """独立复刻 main() 中 assemble（仅 'op' 源），不依赖 shell_system.main 闭包"""
    n1 = npts
    W = np.zeros((n * n1, n * n1)); KX = np.zeros_like(W); KY = np.zeros_like(W)
    KXY = np.zeros_like(W); SX = np.zeros_like(W); SY = np.zeros_like(W)
    SXY = np.zeros_like(W); WX = np.zeros_like(W); WY = np.zeros_like(W)
    scl = torch.tensor(scale if scale is not None else np.ones(16),
                       dtype=torch.float64)
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    g = np.linspace(-1, 1, n1)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    for ei in range(n):
        for ej in range(n):
            ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                   [(0, 0), (1, 0), (1, 1), (0, 1)]]
            de = d_all.reshape(-1, 4)[ids].ravel()
            d_t = (torch.tensor(de) * scl).reshape(1, 16)
            with torch.no_grad():
                wk = op.wk_at(d_t, torch.tensor(pts)).numpy()
                sp = op.s_at(d_t, torch.stack(
                    [torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1]),
                     torch.ones(pts.shape[0])], dim=1)).numpy()
            sl_i = slice(ei * n1, (ei + 1) * n1)
            sl_j = slice(ej * n1, (ej + 1) * n1)
            W[sl_i, sl_j] = wk[:, 0].reshape(n1, n1)
            KX[sl_i, sl_j] = wk[:, 1].reshape(n1, n1)
            KY[sl_i, sl_j] = wk[:, 2].reshape(n1, n1)
            KXY[sl_i, sl_j] = wk[:, 3].reshape(n1, n1)
            SX[sl_i, sl_j] = sp[:, 0].reshape(n1, n1)
            SY[sl_i, sl_j] = sp[:, 1].reshape(n1, n1)
            SXY[sl_i, sl_j] = sp[:, 2].reshape(n1, n1)
            xg = xs[ei] + 0.5 * Lex * (pts[:, 0] + 1)
            yg = ys[ej] + 0.5 * Lex * (pts[:, 1] + 1)
            WX[sl_i, sl_j] = xg.reshape(n1, n1); WY[sl_i, sl_j] = yg.reshape(n1, n1)
    return dict(W=W, KX=KX, KY=KY, KXY=KXY, SX=SX, SY=SY, SXY=SXY, X=WX, Y=WY)


def l2rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def _metrics(F):
    XR, YR = F["X"].ravel(), F["Y"].ravel()
    w_t = S.w_analytic(XR, YR, Q)
    kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, Q)
    sx_t, sy_t, sxy_t = S.stress_analytic(XR, YR, 1.0, Q)
    return dict(
        w=l2rel(F["W"].ravel(), w_t),
        kxx=l2rel(F["KX"].ravel(), kx_t), kyy=l2rel(F["KY"].ravel(), ky_t),
        kxy=l2rel(F["KXY"].ravel(), kxy_t),
        sxx=l2rel(F["SX"].ravel(), sx_t), syy=l2rel(F["SY"].ravel(), sy_t),
        sxy=l2rel(F["SXY"].ravel(), sxy_t),
        # Mxy 对称线伪影：y=0.5 附近 κxy 的真值应≈0，取其 max|·|
        mxy_center_max=float(np.abs(F["KXY"][
            F["KXY"].shape[0] // 2 - 2: F["KXY"].shape[0] // 2 + 3,
            :]).max()) if "KXY" in F else 0.0,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--dof", default="pinn", choices=["pinn", "exact"])
    ap.add_argument("--scale", action="store_true")
    args = ap.parse_args()

    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    op = S.PlateOperator(p=64, hidden=64)
    op_path = os.path.join(RES, f"op_state{args.tag}.pt")
    op.load_state_dict(torch.load(op_path, weights_only=False))
    print(f"[载入] pinn_state.pt + {os.path.basename(op_path)}")

    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Xnr, Ynr = np.meshgrid(xs, ys, indexing="ij")
    Xnr, Ynr = Xnr.ravel(), Ynr.ravel()

    if args.dof == "pinn":
        w0, wx0, wy0, wxy0 = pinn_field_all(pinn, Xnr, Ynr)
        d = np.stack([w0, wx0, wy0, wxy0], axis=1)   # (N,4)，与 shell_system 一致
    else:
        wq = S.w_analytic(Xnr, Ynr, Q); wxq, wyq = S._w_grad(Xnr, Ynr, Q)
        kxyq = S.curv_analytic(Xnr, Ynr, Q)[2]
        d = np.stack([wq, wxq, wyq, -kxyq], axis=1)

    scale = np.ones(16)
    if args.scale:
        # 简单情形：与 shell_system 的 compute_feature_scale 一致的成本过高，跳过
        pass

    avg, single = _sym_avg_field(d, op, scale=None)
    m_s, m_a = _metrics(single), _metrics(avg)
    print("\n======== 闭环 (q=1, %s DOF) 单次 vs 板级对称平均 ========" % args.dof)
    print(f"{'场量':8s} {'单次':>12s} {'对称平均':>12s} {'改善':>8s}")
    for k in ["w", "kxx", "kyy", "kxy", "sxx", "syy", "sxy"]:
        r = m_s[k] / max(m_a[k], 1e-300)
        print(f"{k:8s} {m_s[k]:.4e} {m_a[k]:.4e} {r:7.2f}x")
    print(f"\nMxy 对称线 max|κxy| (应≈0):  单次 {m_s['mxy_center_max']:.3e}  "
          f"对称平均 {m_a['mxy_center_max']:.3e}")

    # ---- 泛化 30 工况（对称平均） ----
    rng = np.random.default_rng(11)
    qs = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    print("\n===== 跨工况泛化 30（对称平均）=====")
    gen_s, gen_a = {"w": [], "k": [], "s": []}, {"w": [], "k": [], "s": []}
    for qv in qs:
        wq = S.w_analytic(Xnr, Ynr, qv); wxq, wyq = S._w_grad(Xnr, Ynr, qv)
        kxyq = S.curv_analytic(Xnr, Ynr, qv)[2]
        dq = np.stack([wq, wxq, wyq, -kxyq], axis=1)
        av, sg = _sym_avg_field(dq, op, scale=None, npts=11)
        for pool, F in ((gen_s, sg), (gen_a, av)):
            XR, YR = F["X"].ravel(), F["Y"].ravel()
            kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, qv)
            sx_t, sy_t, sxy_t = S.stress_analytic(XR, YR, 1.0, qv)
            pool["w"].append(l2rel(F["W"].ravel(), S.w_analytic(XR, YR, qv)))
            pool["k"].append(l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(),
                                                   F["KXY"].ravel()]),
                                   np.concatenate([kx_t, ky_t, kxy_t])))
            pool["s"].append(l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(),
                                                   F["SXY"].ravel()]),
                                   np.concatenate([sx_t, sy_t, sxy_t])))
    for nm, g in (("单次", gen_s), ("对称平均", gen_a)):
        print(f"[{nm}] w={np.mean(g['w']):.2e}±{np.std(g['w']):.1e} "
              f"(CV {np.std(g['w'])/max(np.mean(g['w']),1e-300)*100:.0f}%)  "
              f"κ={np.mean(g['k']):.2e} (CV {np.std(g['k'])/max(np.mean(g['k']),1e-300)*100:.0f}%)  "
              f"σ={np.mean(g['s']):.2e} (CV {np.std(g['s'])/max(np.mean(g['s']),1e-300)*100:.0f}%)")


def pinn_field_all(pinn, Xr, Yr):
    """与 shell_system.pinn_field 等价：返回 (w, wx, wy, wxy) 各为 array"""
    xy = torch.tensor(np.stack([Xr, Yr], axis=1), requires_grad=True)
    u = pinn(xy).reshape(-1)
    g, H = S._grad_hess(u, xy)
    uxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    return (Q * u.detach().numpy(), Q * g[:, 0].detach().numpy(),
            Q * g[:, 1].detach().numpy(), Q * uxy.detach().numpy())


if __name__ == "__main__":
    main()
