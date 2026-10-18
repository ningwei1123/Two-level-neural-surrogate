# -*- coding: utf-8 -*-
"""
sym_aug.py — 方板板级对称变换的纯函数实现 + 物理一致性单测
================================================================
目标（research_plan_tier1.md 动作 A / S1）：
对四边简支方板（a=b=1, 4×4 BFS 网格, 均布载荷）单元节点自由度向量 de(16)
实现 4 元对称子群作用：e / x 反射 / y 反射 / 180° 旋转，
用于：(1) 推理期对称平均（对消 Mxy 反对称伪影，M1）；
      (2) 训练期等变正则 / 数据增广（后续 S2）。

DOF 约定（与 shell_system.py 严格一致）
-----------------------------------------
de = dnode[ids].ravel()，节点顺序 ids=[(0,0),(1,0),(1,1),(0,1)]（相对单元左下角），
每个节点 4 分量 {w, w_x, w_y, w_xy}（w_xy 存储 = -κ_xy，见 shell_system dnode 组装）。
局部坐标 (ξ,η)∈[-1,1]² 与节点序对应：
  block0 (ξ=-1,η=-1)   block1 (ξ=+1,η=-1)
  block2 (ξ=+1,η=+1)   block3 (ξ=-1,η=+1)

板级变换（对称轴取板中线 x=0.5 / y=0.5，180° 绕板心）：
  Rx: (x,y)->(1-x,y)     局部 (ξ,η)->(-ξ,η)     w_x,w_xy 变号
  Ry: (x,y)->(x,1-y)     局部 (ξ,η)->(ξ,-η)     w_y,w_xy 变号
  R180 = Rx∘Ry           w_x,w_y 变号，w_xy 不变

单元 (ei,ej) 经 Rx 映射到镜像单元 (n-1-ei, ej)。本模块只作用于 de 向量与局部坐标，
镜像单元的全局采样坐标由调用方按 ei' = n-1-ei 重算（见 augment / 推理平均评估器）。
"""
import numpy as np

# 节点块置换表（按新局部序 [-1,-1],[+1,-1],[+1,+1],[-1,+1] 排列，取值=旧块索引）
_PX = np.array([1, 0, 3, 2], dtype=np.int64)   # x 反射：ξ 反向
_PY = np.array([3, 2, 1, 0], dtype=np.int64)   # y 反射：η 反向
# 每节点 4 分量符号（w, w_x, w_y, w_xy）
_SX = np.array([1.0, -1.0, 1.0, -1.0])
_SY = np.array([1.0, 1.0, -1.0, -1.0])
_S180 = np.array([1.0, -1.0, -1.0, 1.0])       # Rx∘Ry 的符号复合（校验于下方单测）
_P180 = np.array([2, 3, 0, 1], dtype=np.int64)  # = PX[PY[k]]（校验）

_LOCAL_XI = np.array([[-1.0, -1.0], [1.0, -1.0],
                      [1.0, 1.0], [-1.0, 1.0]])  # 节点局部坐标（与 CORNER_XI 一致）


def _apply(d16, perm, sign):
    """d16: (16,) 或 (N,16)；按块置换 perm 并乘每块符号 sign"""
    d = np.asarray(d16, dtype=np.float64)
    d = np.atleast_2d(d).reshape(-1, 4, 4)        # (N,4节点,4分量)
    out = d[:, perm, :] * sign[None, None, :]
    return out.reshape(-1, 16) if d16 is not None and np.ndim(d16) == 2 else out.reshape(16)


def flip_x_dof(d16):
    """x 反射（x→1-x）：ξ 反向；w_x, w_xy 变号"""
    return _apply(d16, _PX, _SX)


def flip_y_dof(d16):
    """y 反射（y→1-y）：η 反向；w_y, w_xy 变号"""
    return _apply(d16, _PY, _SY)


def rot180_dof(d16):
    """绕板心 180°：(x,y)->(1-x,1-y)；w_x,w_y 变号"""
    return _apply(d16, _P180, _S180)


def flip_x_xi(xi):
    """局部坐标 x 反射：(ξ,η)->(-ξ,η)"""
    xi = np.asarray(xi, dtype=np.float64)
    out = np.array(xi, copy=True)
    out[..., 0] = -xi[..., 0]
    return out


def flip_y_xi(xi):
    """局部坐标 y 反射：(ξ,η)->(ξ,-η)"""
    xi = np.asarray(xi, dtype=np.float64)
    out = np.array(xi, copy=True)
    out[..., 1] = -xi[..., 1]
    return out


def rot180_xi(xi):
    """局部坐标 180°：(ξ,η)->(-ξ,-η)"""
    return -np.asarray(xi, dtype=np.float64)


# ---------------------------------------------------------------------------
# 单测（python sym_aug.py 直接运行）
# ---------------------------------------------------------------------------
def _unit_de(n, ei, ej):
    """在单元 (ei,ej) 四个角用解析场构造真实 de(16)（含 w_x,w_y,-κ_xy）"""
    import shell_system as S
    Lex = S.A / n
    xs = np.linspace(0, S.A, n + 1)
    ys = np.linspace(0, S.B, n + 1)
    # 四个角全局坐标（局部 -1/+1）
    X = np.array([xs[ei], xs[ei + 1], xs[ei + 1], xs[ei]])
    Y = np.array([ys[ej], ys[ej], ys[ej + 1], ys[ej + 1]])
    w = S.w_analytic(X, Y, S.Q_TEST)
    wx, wy = S._w_grad(X, Y, S.Q_TEST)
    kxy = S.curv_analytic(X, Y, S.Q_TEST)[2]
    return np.stack([w, wx, wy, -kxy], axis=1).ravel()


def _mirror_de(n, ei, ej, kind):
    """镜像单元上直接解析取值构造的 de（作为物理真值）"""
    import shell_system as S
    if kind == "x":
        ei2, ej2 = n - 1 - ei, ej
    elif kind == "y":
        ei2, ej2 = ei, n - 1 - ej
    elif kind == "r180":
        ei2, ej2 = n - 1 - ei, n - 1 - ej
    else:
        raise ValueError(kind)
    return _unit_de(n, ei2, ej2)


def run_tests():
    n, Q = 4, 1.0
    ok = []
    def check(name, cond):
        ok.append((name, bool(np.all(cond))))
        if not np.all(cond):
            print(f"  [FAIL] {name}")

    rng = np.random.default_rng(0)
    de_ref = rng.uniform(-1, 1, 16)          # 任意 16 向量，验证群代数

    # --- G1: 逆元与复合（纯代数） ---
    check("G1a flip_x 对合", np.allclose(flip_x_dof(flip_x_dof(de_ref)), de_ref))
    check("G1b flip_y 对合", np.allclose(flip_y_dof(flip_y_dof(de_ref)), de_ref))
    check("G1c rot180 对合", np.allclose(rot180_dof(rot180_dof(de_ref)), de_ref))
    check("G1d rot180 = Rx∘Ry", np.allclose(
        rot180_dof(de_ref), flip_x_dof(flip_y_dof(de_ref))))
    check("G1e rot180 = Ry∘Rx (交换)", np.allclose(
        rot180_dof(de_ref), flip_y_dof(flip_x_dof(de_ref))))

    # --- G2: 局部坐标作用 ---
    xi = rng.uniform(-1, 1, (7, 2))
    check("G2a Rx 坐标对合", np.allclose(flip_x_xi(flip_x_xi(xi)), xi))
    check("G2b R180 坐标", np.allclose(rot180_xi(xi), -xi))
    check("G2c 坐标复合", np.allclose(rot180_xi(xi), flip_x_xi(flip_y_xi(xi))))

    # --- G3: 物理一致性（核心）：变换 de == 镜像单元直接解析 de ---
    for ei in range(n):
        for ej in range(n):
            de = _unit_de(n, ei, ej)
            check(f"G3 x@{ei},{ej}", np.allclose(
                flip_x_dof(de), _mirror_de(n, ei, ej, "x"), rtol=1e-9, atol=1e-11))
            check(f"G3 y@{ei},{ej}", np.allclose(
                flip_y_dof(de), _mirror_de(n, ei, ej, "y"), rtol=1e-9, atol=1e-11))
            check(f"G3 r@{ei},{ej}", np.allclose(
                rot180_dof(de), _mirror_de(n, ei, ej, "r180"), rtol=1e-9, atol=1e-11))
    # 全部单元 + 抽样随机单元（组合同样成立）
    rng2 = np.random.default_rng(1)
    for _ in range(20):
        ei = rng2.integers(0, n); ej = rng2.integers(0, n)
        de = _unit_de(n, ei, ej)
        check("G3 comb RxRy", np.allclose(
            flip_x_dof(flip_y_dof(de)),
            _mirror_de(n, ei, ej, "r180"), rtol=1e-9, atol=1e-11))

    passed = sum(1 for _, c in ok if c)
    print(f"\n[sym_aug] 单测通过 {passed}/{len(ok)}")
    if passed < len(ok):
        raise SystemExit(1)
    print("[sym_aug] 全部通过：对称变换与镜像单元解析取值严格一致")


if __name__ == "__main__":
    run_tests()
