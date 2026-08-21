"""汇总一批 `windows.json`：一次批量运行的产出概览。

干什么：扫 `<root>/*/windows.json`，每段素材一行，打出轨迹数、角色数、选出的
片段数，以及**片段数 / 不重叠窗口上限**——完整片源上最该盯的就是这一列，
它直接回答"这套门槛到底筛掉了多少"。没有人工标注时这是唯一的整体指标。

怎么跑（项目根目录）：

    $py backend/core/main.py data/原始数据/*.mp4 --no-clip --output-dir output_raw
    $py backend/core/summarize.py output_raw

配合 `backend/core/montage.py` 用：这里看数量对不对，那里看身份分对没分对。
"""

import argparse
import glob
import json
import os
import unicodedata
from typing import List, Optional

_COLUMNS = [("片源", 20), ("风格", 6), ("时长s", 8), ("轨迹", 7), ("角色", 6),
            ("片段", 6), ("窗口上限", 10), ("窗内中位", 10), ("窗内最大", 10)]


def _width(text: str) -> int:
    """字符串在等宽终端里占几列：CJK 全角算 2 列（中文片名不换算，表格必错位）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(value, width: int) -> str:
    text = str(value)
    while _width(text) > width:
        text = text[1:]  # 超宽从左侧截断，保留信息量更大的尾部
    return " " * (width - _width(text)) + text


def collect(root: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "windows.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        counts = sorted(s["character_count"] for s in data["segments"])
        rows.append({
            "name": os.path.basename(os.path.dirname(path)),
            "style": data["style"],
            "duration": data["duration"],
            "tracks": data["num_tracks"],
            "characters": data["num_characters"],
            "segments": len(counts),
            # 不重叠窗口上限：整段视频最多能切出几个互不重叠的 window_seconds 片段。
            # 分母取它，才能把"选出 40 个片段"读成"71% 的片长被判为合格"。
            "max_windows": int(data["duration"] // data["params"]["window_seconds"]),
            "median": counts[len(counts) // 2] if counts else 0,
            "max": counts[-1] if counts else 0,
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="汇总一批运行结果的 windows.json。")
    parser.add_argument("root", help="输出根目录（含多个 <stem>/windows.json）。")
    args = parser.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print(f"{args.root} 下没有任何 windows.json。")
        return 1

    header = "".join(_pad(name, w) for name, w in _COLUMNS)
    print(header)
    print("-" * _width(header))
    for row in rows:
        print(_pad(row["name"], 20) + _pad(row["style"], 6)
              + _pad(f"{row['duration']:.0f}", 8) + _pad(row["tracks"], 7)
              + _pad(row["characters"], 6) + _pad(row["segments"], 6)
              + _pad(row["max_windows"], 10) + _pad(row["median"], 10) + _pad(row["max"], 10))

    segments = sum(r["segments"] for r in rows)
    windows = sum(r["max_windows"] for r in rows)
    print(f"\n合计 {len(rows)} 段：轨迹 {sum(r['tracks'] for r in rows)}，"
          f"角色 {sum(r['characters'] for r in rows)}，"
          f"片段 {segments} / 窗口上限 {windows} = "
          f"{100 * segments / max(1, windows):.0f}% 的片长被判为合格片段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
