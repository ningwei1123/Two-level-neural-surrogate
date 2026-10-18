# -*- coding: utf-8 -*-
"""算子配置对照实验：只训练 OP + 轻量评估（不绘图、不写 results.json，避免并行冲突）
用法: python eval_op.py <tag> <lw> <epochs> <holdout(0/1)>
"""
import os
import sys
import time
import numpy as np
import torch
import shell_system as S

RES = os.path.join(S.RES)


def evaluate(op, n=4, npts=11, qs=(1.0,)):
    """在 qs 各工况（解析节点自由度）评估算子全局场 L2 误差（w/κ/σ 平均）"""
    xs = np.linspace(0, S.A, n + 1); ys = np.linspace(0, S.B, n + 1)
    Lex = S.A / n
    Xn, Yn = np.meshgrid(xs, ys, indexing="ij")
    Xr, Yr = Xn.ravel(), Yn.ravel()
    g = np.linspace(-1, 1, npts)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    PX, PY = [], []
    for ei in range(n):
        for ej in range(n):
            PX.append(xs[ei] + 0.5 * Lex * (pts[:, 0] + 1))
            PY.append(ys[ej] + 0.5 * Lex * (pts[:, 1] + 1))
    PX = np.concatenate(PX); PY = np.concatenate(PY)
    out = {}
    for q in qs:
        w0 = S.w_analytic(Xr, Yr, q)
        wx0, wy0 = S._w_grad(Xr, Yr, q)
        kxy0 = S.curv_analytic(Xr, Yr, q)[2]
        dnode = np.stack([w0, wx0, wy0, -kxy0], axis=1)
        W = []; K = []; Sg = []
        for ei in range(n):
            for ej in range(n):
                ids = [(ei + di) * (n + 1) + (ej + dj) for (di, dj) in
                       [(0, 0), (1, 0), (1, 1), (0, 1)]]
                de = dnode[ids].ravel()
                d_t = torch.tensor(de).reshape(1, 16)
                with torch.no_grad():
                    wk = op.wk_at(d_t, torch.tensor(pts)).numpy()
                    sp = op.s_at(d_t, torch.stack(
                        [torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1]),
                         torch.full((pts.shape[0],), 1.0)], dim=1)).numpy()
                W.append(wk[:, 0]); K.append(wk[:, 1:4]); Sg.append(sp)
        W = np.concatenate(W); K = np.concatenate(K); Sg = np.concatenate(Sg)
        kx_t, ky_t, kxy_t = S.curv_analytic(PX, PY, q)
        sx_t, sy_t, sxy_t = S.stress_analytic(PX, PY, 1.0, q)
        # 逐分量相对 L2，再平均（避免点主序/分量主序拼接错位）
        out[q] = dict(
            w=S.l2rel(W, S.w_analytic(PX, PY, q)),
            k=float(np.mean([S.l2rel(K[:, 0], kx_t), S.l2rel(K[:, 1], ky_t),
                             S.l2rel(K[:, 2], kxy_t)])),
            s=float(np.mean([S.l2rel(Sg[:, 0], sx_t), S.l2rel(Sg[:, 1], sy_t),
                             S.l2rel(Sg[:, 2], sxy_t)])),
        )
    return out


def main():
    tag = sys.argv[1]
    lw = float(sys.argv[2])
    n_ep = int(sys.argv[3])
    holdout = int(sys.argv[4]) == 1
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    torch.manual_seed(S.SEED); np.random.seed(S.SEED)

    qs_all = np.random.default_rng(3).uniform(S.Q_LO, S.Q_HI, 200)
    if not holdout:
        qs_train = np.concatenate([qs_all[:137], [1.0, 0.98, 1.02]])
    else:
        qs_train = qs_all[(qs_all < 0.9) | (qs_all > 1.1)][:120]
    samples = S.make_operator_dataset(qs_train, n=4, n_pts=10, seed=1)
    op = S.PlateOperator(p=64, hidden=64)
    print(f"[{tag}] 训练工况 {len(qs_train)}，lw={lw}，epochs={n_ep}，holdout={holdout}")
    t0 = time.time()
    S.train_operator(op, samples, n_epochs=n_ep, batch=24, Le=0.25, lam_w=lw,
                     log_every=250)
    torch.save(op.state_dict(), os.path.join(RES, f"op_{tag}.pt"))
    print(f"[{tag}] 训练耗时 {time.time()-t0:.0f}s，开始评估...")
    q_test = [0.6, 0.9, 1.0, 1.1, 1.5]
    res = evaluate(op, qs=tuple(q_test))
    for q, r in res.items():
        print(f"[{tag}] q={q}: w={r['w']:.3e}  κ={r['k']:.3e}  σ={r['s']:.3e}")
    # 泛化统计（30 个，避开 [0.9,1.1] 若 holdout）
    rng = np.random.default_rng(11)
    if holdout:
        qs_g = np.concatenate([rng.uniform(0.5, 0.9, 15), rng.uniform(1.1, 2.0, 15)])
    else:
        qs_g = np.concatenate([rng.uniform(0.5, 0.98, 15), rng.uniform(1.02, 2.0, 15)])
    gg = evaluate(op, qs=tuple(qs_g))
    arr = {k: [gg[q][k] for q in gg] for k in ("w", "k", "s")}
    print(f"[{tag}] 泛化({len(gg)}工况): w={np.mean(arr['w']):.2e}±{np.std(arr['w']):.1e}  "
          f"κ={np.mean(arr['k']):.2e}±{np.std(arr['k']):.1e}  "
          f"σ={np.mean(arr['s']):.2e}±{np.std(arr['s']):.1e}")


if __name__ == "__main__":
    main()
