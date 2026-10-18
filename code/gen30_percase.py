# -*- coding: utf-8 -*-
"""
gen30_percase.py — 30 未见载荷工况的逐工况误差数据（供 Fig.11/表3 双口径图与端到端行使用）
口径与 shell_system.main 泛化段逐位同构：
  · q 序列: rng(11) 同序
  · w*   : wstar_from_field(Fq, npts=11) 唯一网格 Poisson 重构 vs 解析
  · κ/σ  : 分片场拼接三分量 L2（非三分量单独平均）
  · 端到端: Level-I PINN 在 q 处前向得到节点 DOF（载荷分离，一次前向/工况）
落盘（读改写保留其它键）:
  generalization.percase_exact = {q, w_star, k, s}
  supp_review.gen30_pinn_dof   = 原聚合键 + k_std/k_cv_pct/s_std/s_cv_pct + percase_pinn
并打印 exact 聚合 vs generalization 聚合的核验偏差。
"""
import os
import json
import numpy as np
import torch
import shell_system as S
from eval_symavg import _assemble, l2rel
import eval_poisson as EP
from supp_experiments import exact_dof, pinn_dof

RES = S.RES
rj = os.path.join(RES, "results.json")
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = S.N_MESH


def metrics(F, qv, npts=11):
    """与 shell_system.main 泛化同构：返回 (w_star, k, s) 三标量"""
    XR, YR = F["X"].ravel(), F["Y"].ravel()
    kx_t, ky_t, kxy_t = S.curv_analytic(XR, YR, qv)
    sx_t, sy_t, sxy_t = S.stress_analytic(XR, YR, 1.0, qv)
    w_star, x_u = S.wstar_from_field(F, npts=npts)
    XU, YU = np.meshgrid(x_u, x_u, indexing="ij")
    wv = l2rel(w_star.ravel(), S.w_analytic(XU.ravel(), YU.ravel(), qv))
    kv = l2rel(np.concatenate([F["KX"].ravel(), F["KY"].ravel(), F["KXY"].ravel()]),
               np.concatenate([kx_t, ky_t, kxy_t]))
    sv = l2rel(np.concatenate([F["SX"].ravel(), F["SY"].ravel(), F["SXY"].ravel()]),
               np.concatenate([sx_t, sy_t, sxy_t]))
    return wv, kv, sv


def main():
    op = S.PlateOperator(p=64, hidden=64)
    op.load_state_dict(torch.load(os.path.join(RES, "op_state.pt"), weights_only=False))
    op.eval()
    pinn = EP._load_pinn()

    rng = np.random.default_rng(11)
    qs = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])

    pe = {"q": [], "w_star": [], "k": [], "s": []}     # exact DOF
    pp = {"q": [], "w_star": [], "k": [], "s": []}     # end-to-end PINN DOF
    for qv in qs:
        dp = pinn_dof(pinn, qv)                 # PINN 二阶导需要 autograd，置于 no_grad 外
        with torch.no_grad():
            Fe = _assemble(exact_dof(qv), op, scale=None, npts=11)
            Fp = _assemble(dp, op, scale=None, npts=11)
        we, ke, se = metrics(Fe, qv)
        wp, kp, sp = metrics(Fp, qv)
        pe["q"].append(float(qv)); pe["w_star"].append(we); pe["k"].append(ke); pe["s"].append(se)
        pp["q"].append(float(qv)); pp["w_star"].append(wp); pp["k"].append(kp); pp["s"].append(sp)

    R = json.load(open(rj, encoding="utf-8"))
    R["generalization"]["percase_exact"] = pe
    sp_ = R.setdefault("supp_review", {})
    g30 = sp_.setdefault("gen30_pinn_dof", {})
    g30.update({
        "w_star_mean": float(np.mean(pp["w_star"])),
        "w_star_std": float(np.std(pp["w_star"])),
        "w_star_cv_pct": float(np.std(pp["w_star"]) / np.mean(pp["w_star"]) * 100),
        "k_mean": float(np.mean(pp["k"])), "k_std": float(np.std(pp["k"])),
        "k_cv_pct": float(np.std(pp["k"]) / np.mean(pp["k"]) * 100),
        "s_mean": float(np.mean(pp["s"])), "s_std": float(np.std(pp["s"])),
        "s_cv_pct": float(np.std(pp["s"]) / np.mean(pp["s"]) * 100),
        "percase_pinn": pp,
    })
    json.dump(R, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 核验：exact percase 聚合 vs main generalization 聚合 ----
    g = R["generalization"]
    for key in ["w_star", "k", "s"]:
        agg_new = float(np.mean(pe[key]))
        agg_old = float(g[f"{key}_mean"]) if f"{key}_mean" in g else float("nan")
        print(f"exact {key}: percase-mean={agg_new:.5e}  main-agg={agg_old:.5e}  "
              f"rel-diff={abs(agg_new-agg_old)/max(abs(agg_old),1e-300):.2e}")
    print(f"pinn  w*: {g30['w_star_mean']:.4e}±{g30['w_star_std']:.1e} CV{g30['w_star_cv_pct']:.0f}%")
    print(f"pinn  κ : {g30['k_mean']:.4e}±{g30['k_std']:.1e} CV{g30['k_cv_pct']:.0f}%")
    print(f"pinn  σ : {g30['s_mean']:.4e}±{g30['s_std']:.1e} CV{g30['s_cv_pct']:.0f}%")


if __name__ == "__main__":
    main()
