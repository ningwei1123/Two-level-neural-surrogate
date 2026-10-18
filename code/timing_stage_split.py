# -*- coding: utf-8 -*-
"""
timing_stage_split.py — stage-by-stage timing of one surrogate call
(Table 4; supersedes the older v10_stage.json).

问题：V10 的 _v10_stage_split.py 在 16x16 上报 element=3.18 + poisson=4.20 ms，
而 Table 4 同一行的 surrogate 是 4.29 ms，读者一加就差 3.09 ms；4x4 组
(1.29 + 0.54 + 0.14 = 1.97，剔除 Level-I 后 0.68) 与 0.70 却自洽。

根因：**分量单独计时**。0.1–4 ms 的算子在 Windows 上单次计时会被分配器状态、
GC 与调度毛刺抬高 50–100%，抬高幅度随网格变化，于是三段之和与总时长脱钩。

本脚本改两处：
1) **链内打点**：一次调用里跑完 Level-I -> 单元前向 -> Poisson，perf_counter 记三个
   区间，三段之和恒等于该次调用的总时长（closure 精确为 0，无残差）。
2) **best-case 统计**：块内重复、块间取**最小值**（微小算子的标准做法，抗加性干扰）。

绝对毫秒数仍锚定 Table 4 的 surrogate 列（本机当前比记录时慢 ~1.9x，直接替换会
污染 §5.7 / Table 9 的全链条数值），实测值只用于定**份额**。
输出 results/v18_stage.json。
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

import shell_system as S
import rev_b
from rev_scale2 import poisson_factor, poisson_solve
from rev_common import load_json

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)

PC = time.perf_counter


def bench_best(fn, min_block=0.030, blocks=21):
    """fn() 返回 (t1,t2,t3,t4)；返回各段 best-case（逐块最小值的再最小）。"""
    for _ in range(8):
        fn()
    t0 = PC()
    fn()
    one = PC() - t0
    reps = max(1, int(min_block / max(one, 1e-9)))
    for _ in range(3):
        for _ in range(reps):
            fn()
    best = None
    for _ in range(blocks):
        s = np.zeros(4)
        for _ in range(reps):
            s += np.asarray(fn(), dtype=float)
        s /= reps
        best = s if best is None else np.minimum(best, s)
    return best, reps


def main():
    R = load_json()
    s2 = R["supp_review"]["scaling2"]["grids"]
    pinn = S.PlatePINN(hidden=64, nlayer=4)
    pinn.load_state_dict(torch.load(os.path.join(RES, "pinn_state.pt"),
                                    weights_only=False))
    pinn.eval()
    grids = {"4": "op_state.pt", "8": "op_state_g8.pt", "16": "op_state_g16.pt"}
    out = {"npts": 11, "threads": int(torch.get_num_threads()),
           "protocol": ("in-chain instrumentation (Level-I -> element -> Poisson "
                        "timed with perf_counter inside one call, so the three "
                        "stages sum to the measured total by construction); "
                        "best-case statistics over blocks of >=30 ms; absolute ms "
                        "anchored to the Table 4 surrogate column, shares measured here")}

    for n_s, fname in grids.items():
        n = int(n_s)
        op = S.PlateOperator(p=64, hidden=64)
        op.load_state_dict(torch.load(os.path.join(RES, fname), weights_only=False))
        op.eval()
        keep = rev_b.unique_idx(n, 11)
        M = keep.size
        pfac, ph = poisson_factor(M)

        def full():
            t1 = PC()
            d = rev_b.pinn_dof_grid(pinn, 1.0, n)
            t2 = PC()
            Fq = rev_b.assemble_op_batch(d, op, n, npts=11)
            t3 = PC()
            poisson_solve(pfac, ph, Fq["KX"][np.ix_(keep, keep)],
                          Fq["KY"][np.ix_(keep, keep)])
            t4 = PC()
            return t1, t2, t3, t4

        a, reps = bench_best(full)
        l1, el, po = a[1] - a[0], a[2] - a[1], a[3] - a[2]
        tot = a[3] - a[0]
        inc = el + po

        inc_ms = float(s2[n_s]["surrogate_ms"])
        f_el, f_po = el / inc, po / inc
        # 两位小数下 element + poisson 必须精确等于表里的 surrogate，
        # 否则读者一加就出现 0.01 的"证据矛盾"
        inc_r = round(inc_ms, 2)
        _el_r = round(inc_ms * f_el, 2)
        _po_r = round(inc_r - _el_r, 2)
        row = {
            "n_elem": n * n, "M_poisson": int(M), "reps_per_block": int(reps),
            "raw_ms": {"level1": round(l1 * 1e3, 3),
                       "element": round(el * 1e3, 3),
                       "poisson": round(po * 1e3, 3),
                       "incr": round(inc * 1e3, 3),
                       "first_load": round(tot * 1e3, 3)},
            "closure_err": round((l1 + el + po) / tot - 1.0, 6),
            "level1_over_incr": round(tot / inc - 1.0, 3),
            "surrogate_incr_ms": inc_ms,
            "element_ms": _el_r,
            "poisson_ms": _po_r,
            "level1_ms": round(inc_ms * (tot / inc - 1.0), 3),
            "first_load_ms": round(inc_ms * (tot / inc), 3),
        }
        # 整数份额，且三者相加恒为 100（第三个取残差），
        # 避免正文里 34+26+39=99 这种"读者一加就出问题"的展示瑕疵
        row["share_level1"] = int(round(100 * row["level1_ms"] / row["first_load_ms"]))
        row["share_element"] = int(round(
            100 * row["element_ms"] / row["first_load_ms"]))
        row["share_poisson"] = int(
            100 - row["share_level1"] - row["share_element"])
        row["sum_ms"] = round(row["level1_ms"] + row["element_ms"] +
                              row["poisson_ms"], 3)
        out[n_s] = row
        r = row["raw_ms"]
        print(f"[{n_s}] raw L1={r['level1']} el={r['element']} po={r['poisson']} "
              f"incr={r['incr']} tot={r['first_load']} "
              f"(close {row['closure_err']:+.1e}, reps={reps}) "
              f"| L1/incr={row['level1_over_incr']:+.2f} "
              f"| -> {row['level1_ms']}/{row['element_ms']}/{row['poisson_ms']} ms "
              f"shares {row['share_level1']:.0f}/{row['share_element']:.0f}/"
              f"{row['share_poisson']:.0f}")

    with open(os.path.join(RES, "v18_stage.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("saved results/v18_stage.json")


if __name__ == "__main__":
    main()
