"""从窗口扫描表（windows_scan.json）里重放选段。

存在的理由：改 X（min_events_per_window）不需要重跑检测、也不需要重跑聚类
——每个候选窗口起点的角色数是与 X 无关的量。把它落成一张表，Web 端拖 X 滑块
就退化成一次纯 Python 比较（几十毫秒），否则改一个数字要等几分钟。

本模块**不做 I/O、不 import cv2/onnxruntime**，后端可以直接引用；
:func:`main.select_segments` 也走同一条实现，保证两边不会走偏。
"""

import math
from typing import Dict, List, Optional, Tuple


def nearest_cut(t: float, cuts: List[float], max_shift: float) -> Optional[float]:
    """距 t 最近的切镜时刻；没有或超出 max_shift 时返回 None（max_shift=0 即关闭）。"""
    if not cuts or max_shift <= 0:
        return None
    import bisect

    i = bisect.bisect_left(cuts, t)
    candidates = cuts[max(0, i - 1):i + 1]
    best = min(candidates, key=lambda c: abs(c - t))
    return best if abs(best - t) <= max_shift else None


def select_from_scan(
    scan: Dict,
    min_events: int,
    snap_max_shift: float,
) -> Tuple[List[Dict], int]:
    """在窗口扫描表上贪心选段，逻辑与流水线内的选段完全一致。

    命中 → 起点吸附到 snap_max_shift 内最近的切镜点 → 吸附后复核仍达标才采用
    → 下一个候选窗口跳到 start + window 之后（保证片段不重叠）。

    参数：
        scan: :func:`main.scan_windows` 的产物（windows_scan.json 的内容）。
        min_events: 需求里的 X。
        snap_max_shift: 起点吸附的最大位移秒数（config.clip_snap_max_shift）。

    返回：
        (片段列表, 合格窗口数)。片段只有 start / end / character_count——
        character_ids 要另外从 tracks.json 按时间区间取（见 main.select_segments）。
    """
    step = scan["frame_interval"]
    window = scan["window_seconds"]
    duration = scan["duration"]
    counts = [w["n"] for w in scan["windows"]]
    cuts = scan["cuts"]
    # 吸附目标一定是某个切镜时刻，而切镜时刻不落在 k×step 的网格上，
    # 所以复核用的角色数必须单独存一份（见 main.scan_windows）。
    cut_counts = {round(w["t"], 3): w["n"] for w in scan.get("cut_windows", [])}

    segments: List[Dict] = []
    num_qualified = 0
    k = 0
    while k < len(counts):
        t = k * step
        if t + window > duration + 1e-6:
            break
        count = counts[k]
        if count >= min_events:
            num_qualified += 1
            start = t
            snapped = nearest_cut(t, cuts, snap_max_shift)
            if snapped is not None and snapped + window <= duration + 1e-6:
                snapped_count = cut_counts.get(round(snapped, 3))
                if snapped_count is not None and snapped_count >= min_events:
                    start, count = snapped, snapped_count
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(start + window, 3),
                    "character_count": count,
                    # 吸附前的网格起点。UI 要显示「向前吸附了多少秒到切镜点」，
                    # 而吸附后就再也算不出来了。
                    "raw_start": round(t, 3),
                }
            )
            k = math.ceil((start + window) / step - 1e-9)
        else:
            k += 1
    return segments, num_qualified
