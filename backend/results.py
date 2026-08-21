"""读一次分析的产物，按 X 现算片段。

改 X 不重跑模型：`windows_scan.json` 里每个候选窗口起点的角色数与 X 无关，
拖滑块只是在这张表上重放一次选段（几十毫秒）。选段实现直接用流水线自己的
`segments.select_from_scan`，不另写一套，否则两边数字会走偏。
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from pipeline import pipeline_config, select_from_scan

# (路径, mtime) -> 解析好的 JSON。滑块一拖就是几十次请求，重复解析没必要。
_CACHE: Dict[Tuple[str, float], object] = {}


def _load_json(path: str):
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        with open(path, encoding="utf-8") as f:
            _CACHE.clear()          # 只留最近一批，避免长跑之后无限涨
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def load_scan(out_dir: str) -> Optional[Dict]:
    path = os.path.join(out_dir, "windows_scan.json")
    return _load_json(path) if os.path.isfile(path) else None


def load_tracks(out_dir: str) -> List[Dict]:
    path = os.path.join(out_dir, "tracks.json")
    return _load_json(path) if os.path.isfile(path) else []


def crop_by_character(out_dir: str) -> Dict[int, str]:
    """character_id → 代表裁剪图的相对路径（每个角色取第一条有图的轨迹）。"""
    mapping: Dict[int, str] = {}
    for track in load_tracks(out_dir):
        cid = track.get("character_id")
        crops = track.get("representative_crops") or []
        if cid is None or not crops:
            continue
        mapping.setdefault(int(cid), crops[0])
    return mapping


def _characters_in(tracks: List[Dict], start: float, end: float) -> List[int]:
    """与 [start, end) 相交的轨迹里出现过的不同 character_id（口径同 main）。"""
    ids = {
        int(t["character_id"])
        for t in tracks
        if t.get("character_id") is not None
        and t["start_time"] < end and t["end_time"] >= start
    }
    return sorted(ids)


def segments_for(out_dir: str, x: int) -> List[Dict]:
    """当前 X 下的片段。start 已按切镜点吸附，snap_shift 记录吸附了多少。"""
    scan = load_scan(out_dir)
    if not scan:
        return []
    config = pipeline_config()
    picks, _qualified = select_from_scan(scan, x, config.clip_snap_max_shift)
    tracks = load_tracks(out_dir)
    out = []
    for pick in picks:
        out.append({
            "start": pick["start"],
            "end": pick["end"],
            "count": pick["character_count"],
            "snap_shift": round(pick["raw_start"] - pick["start"], 3),
            "characters": _characters_in(tracks, pick["start"], pick["end"]),
        })
    return out


def curve(scan: Dict, points: int = 200) -> List[float]:
    """把窗口角色数降采样成一条曲线，桶内取最大值。

    取 max 而不是均值：这条曲线是给「哪一段能命中门槛」看的，峰值被均值抹掉
    就与下面的命中块对不上了。
    """
    values = [w["n"] for w in scan.get("windows", [])]
    if not values:
        return []
    if len(values) <= points:
        return [float(v) for v in values]
    step = len(values) / points
    return [
        float(max(values[int(i * step):max(int((i + 1) * step), int(i * step) + 1)]))
        for i in range(points)
    ]


def sensitivity(scans: List[Dict], lo: int = 2, hi: int = 10) -> List[Dict]:
    """门槛敏感度：每个 X 下有多大比例的候选窗口合格。

    需求方还没给 X，这条曲线就是给他看的（README 第六之九节的那张表）。
    """
    values = [w["n"] for scan in scans for w in scan.get("windows", [])]
    total = len(values) or 1
    return [
        {"x": k, "ratio": round(sum(1 for v in values if v >= k) / total, 4)}
        for k in range(lo, hi + 1)
    ]
