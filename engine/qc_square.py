#!/usr/bin/env python3
"""方波损坏质检（对标浪音AISing 的 astats 质检思路，本地泛化实现）。

背景：SEED-VC / BigVGAN 在极端情况下会输出满幅方波——所有样本钉死在
±1.0（逐样本交替）或直流平台（恒 +1.0 / -1.0），听感为刺耳啸叫或死寂。
竞品用 ffmpeg astats（Bit_depth<=1 ∧ Zero_crossings<=1 ∧ Entropy<0.15，
连续 2 帧）判定；但实测 float 域下 astats 的 Bit_depth 是 popcount 口径，
只能抓"恰好 ±1.0"的极端损坏。本模块改用信号特征自判，口径更宽：

  帧级前置条件（帧内够响才值得报警）：peak > -1dBFS 且 RMS > -6dBFS
  坏帧判定（满足其一）：
    A 卡死平台：过零数 zc<=2 ∧ 量化电平数 levels<=4（整帧钉在常数/极少电平，
      含满幅直流与 0.99 等非满幅卡死；饱和占比不参与判定，避免漏报非满幅平台）
    B 逐样本交替：sat>=0.95 ∧ levels<=2 ∧ (zc<=2 ∨ zc>=0.8×帧样本数)
      —— 过零约束用于区分“损坏”与合法低频方波音乐：220Hz 方波 zc≈440，
         远低于 0.8×44100=35280，不误报；逐样本 ±1.0 交替 zc≈44100，命中。

  连续 min_consecutive 个坏帧 → 判定损坏，返回起始秒。

用法：
  CLI   .venv/bin/python engine/qc_square.py <wav> [<wav> ...]
  导入  from engine.qc_square import detect_square_wave

返回：正常返回 None；损坏返回 {"start_sec": float}。
单帧 stats 可通过 frame_stats() 供调试。
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# 阈值（帧级，1 秒帧，44.1kHz 或 48kHz 均适用）
PEAK_MIN = 10 ** (-1 / 20)   # -1 dBFS：满幅损坏才报警
RMS_MIN = 10 ** (-6 / 20)    # -6 dBFS：同竞品口径
ZC_A, LEVELS_A = 2, 4                    # 条件 A：卡死平台（过零数/电平数上限）
SAT_B, LEVELS_B, ZC_RATIO_B = 0.95, 2, 0.8  # 条件 B：逐样本交替
# 条件 A 无饱和占比要求：非满幅（如 0.99）的卡死平台同样是损坏


def frame_stats(x: np.ndarray) -> dict:
    """单帧信号特征（x 为 float64 单声道片段）。"""
    n = len(x)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(np.square(x))))
    sat = float(np.mean(np.abs(x) >= 0.999))
    sign = np.signbit(x)  # 负数 True；+0.0 False
    zc = int(np.count_nonzero(sign[1:] != sign[:-1]))
    levels = int(np.unique(np.round(x, 3)).size)
    return {"n": n, "peak": peak, "rms": rms, "sat": sat, "zc": zc, "levels": levels}


def _is_bad(st: dict) -> bool:
    if not (st["peak"] > PEAK_MIN and st["rms"] > RMS_MIN):
        return False
    if st["zc"] <= ZC_A and st["levels"] <= LEVELS_A:
        return True   # A 卡死平台：整帧钉在常数/极少电平且够响
    if (st["sat"] >= SAT_B and st["levels"] <= LEVELS_B
            and (st["zc"] <= ZC_A or st["zc"] >= ZC_RATIO_B * st["n"])):
        return True   # B 逐样本 ±1.0 交替（含过零约束，防误伤低频方波音乐）
    return False


def detect_square_wave(path, frame_sec: float = 1.0, min_consecutive: int = 2):
    """检测满幅方波损坏。返回 None（正常）或 {"start_sec": 起始秒}。"""
    data, sr = sf.read(str(path))
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 2:                      # 立体声下混（损坏通常两声道同源）
        x = x.mean(axis=1)
    frame = int(round(frame_sec * sr))
    if len(x) < frame:
        return None
    run = 0
    start_idx = None
    for i in range(0, len(x) - frame + 1, frame):
        if _is_bad(frame_stats(x[i:i + frame])):
            if run == 0:
                start_idx = i
            run += 1
            if run >= min_consecutive:
                return {"start_sec": round(start_idx / sr, 1)}
        else:
            run = 0
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python engine/qc_square.py <wav> [<wav> ...]")
        sys.exit(2)
    bad = 0
    for p in sys.argv[1:]:
        try:
            r = detect_square_wave(p)
        except Exception as exc:
            print(f"{Path(p).name}: READ-ERROR {exc}")
            continue
        if r:
            bad += 1
            print(f"{Path(p).name}: BAD  方波损坏 @ {r['start_sec']}s")
        else:
            print(f"{Path(p).name}: OK")
    sys.exit(1 if bad else 0)
