"""片长尺度的聚类范围对照：全片聚类 vs 窗口内聚类。

## 为什么需要这个脚本

`evaluate.py` 用的 9 段素材**每段正好 30 秒**，等于"一个窗口"，因此
`cluster_scope` 在它上面**恒等无效**——它测不出这次改动的任何东西。
而 README 第六之五节记录的过拆（`魔女之旅` 整集 54 个角色、主角一人占 5 簇）
只在完整片源上出现，那里又没有真值。

这个脚本把两边接上：9 段切片都是从 `data/原始数据/` 的完整片源里**按整秒
原位截出**的（已逐帧比对确认 shift=0），所以文件名里的
「跨镜头身份去重后期望」同时也是**完整片源上 [offset, offset+30) 这个窗口的真值**。
于是可以在完整片源上跑一遍，只看那几个有真值的窗口，直接量出：

    全片聚类口径 vs 窗口聚类口径，谁在片长尺度上更接近真值。

参照列 `clip30` 是同一段素材单独作为 30 秒文件跑出来的结果——它是这条流水线
在"没有片长干扰"时的上限。窗口口径若能贴近它，就说明片长带来的过拆确实被
聚类范围这一步消掉了。

画风**强制用真值**（`groundtruth.STYLE_BY_TITLE`），不走自动路由：README 已经
记录路由在完整片源上会把 `爱情神话` 判成 3d，那是另一个独立缺陷，混进来会污染
本次对照。

## 怎么跑（项目根目录）

    $py src/evaluate_long.py                     # 全部能配上对的片源
    $py src/evaluate_long.py --titles 魔女之旅    # 只跑一部，快速验证
    $py src/evaluate_long.py --no-clip-baseline  # 跳过 30 秒切片参照列

产出：控制台表格 + <output-dir>/long_scope.csv 与 long_films.csv。
代价：每部片源解码一遍、身份特征提一次；两种口径共用同一份差异矩阵，
所以第二种口径几乎免费。
"""

import argparse
import csv
import dataclasses
import glob
import os
import time
import unicodedata
from typing import Dict, List, Optional

from config import Config
from evaluate import count_f1, recall
from groundtruth import GroundTruth, parse_ground_truth
from main import (
    IdentityIndex,
    _characters_in_window,
    _cluster_by_difference,
    compute_differences,
    force_utf8_stdout,
    get_detector,
    resolve_identity_threshold,
    scan_video,
    select_segments,
)
from style import apply_style

SCOPES = ("video", "window")


def find_pairs(clip_dir: str, source_dir: str) -> Dict[str, List[GroundTruth]]:
    """把带标注的 30 秒切片按标题归到各自的完整片源上。

    切片文件名的标题段（首个下划线之前）就是片源文件名，两边对不上的直接跳过。
    """
    pairs: Dict[str, List[GroundTruth]] = {}
    for clip in sorted(glob.glob(os.path.join(clip_dir, "*.mp4"))):
        gt = parse_ground_truth(clip)
        if gt is None:
            continue
        source = os.path.join(source_dir, f"{gt.title}.mp4")
        if not os.path.isfile(source):
            print(f"[skip] 找不到片源 {source}")
            continue
        pairs.setdefault(source, []).append(gt)
    return pairs


def scan_once(config: Config, video_path: str, out_dir: str, style: str,
              limit_seconds: Optional[float] = None):
    """扫一遍视频，返回 (轨迹, 时长, 切镜表, 差异索引, 路由后的 config, 全片角色数)。

    画风由调用方指定（真值），不做自动路由——见模块文档。
    全片聚类照常写进 ``track.character_id``（全片口径那一列要读它），窗口口径
    则走返回的索引，两者共用同一份差异矩阵。
    """
    config = apply_style(config, style)
    os.makedirs(out_dir, exist_ok=True)
    detector = get_detector(config.detector, config)
    tracks, detection_records, duration, _frames, cuts = scan_video(
        config, video_path, out_dir, detector, limit_seconds
    )
    # 检测明细本脚本用不到，聚类前先释放：完整片源上它是几万条字典，
    # 而下一步的 CCIP/ArcFace 批推理正是整个进程的内存峰值。
    del detection_records
    candidates, diff = compute_differences(tracks, out_dir, config)
    if diff is None:
        return tracks, duration, cuts, None, config, 0
    threshold = resolve_identity_threshold(config)
    labels = _cluster_by_difference(diff, threshold)
    for track, label in zip(candidates, labels):
        track.character_id = label
    index = IdentityIndex(candidates, diff, threshold)
    return tracks, duration, cuts, index, config, len(set(labels))


def count_in_window(tracks, index, start: float, window: float) -> Dict[str, int]:
    """同一个窗口在两种聚类口径下各数出几个角色。"""
    ordered = sorted(tracks, key=lambda tr: tr.start_time)
    starts = [tr.start_time for tr in ordered]
    ids, overlapping = _characters_in_window(ordered, starts, start, window)
    if index is None:
        return {"video": 0, "window": 0, "tracks": len(overlapping)}
    # 全片口径 = 这些轨迹在全片聚类里分属几个簇（character_id 由调用方预先写好）。
    return {"video": len(ids), "window": index.count(overlapping),
            "tracks": len(overlapping)}


# === 报表 ===

def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(value, width: int) -> str:
    text = str(value)
    while _w(text) > width:
        text = text[1:]
    return " " * (width - _w(text)) + text


_COLS = [("片源_起点", 18), ("风格", 6), ("真值", 6), ("窗内轨迹", 10),
         ("全片口径", 10), ("窗口口径", 10), ("clip30", 8),
         ("F1全片", 8), ("F1窗口", 8), ("F1clip", 8)]


def print_windows(rows: List[Dict]) -> None:
    header = "".join(_pad(n, w) for n, w in _COLS)
    print("\n=== 有真值的窗口：完整片源上 [offset, offset+30) 的角色数 ===")
    print(header)
    print("-" * _w(header))
    for r in rows:
        print(_pad(r["window_id"], 18) + _pad(r["style"], 6) + _pad(r["gt_chars"], 6)
              + _pad(r["window_tracks"], 10) + _pad(r["pred_video"], 10)
              + _pad(r["pred_window"], 10)
              + _pad("-" if r["pred_clip30"] is None else r["pred_clip30"], 8)
              + _pad(f"{r['f1_video']:.2f}", 8) + _pad(f"{r['f1_window']:.2f}", 8)
              + _pad("-" if r["f1_clip30"] is None else f"{r['f1_clip30']:.2f}", 8))

    n = len(rows)
    clip = [r["f1_clip30"] for r in rows if r["f1_clip30"] is not None]
    tail = f"   clip30 参照 {sum(clip) / len(clip):.3f}" if clip else ""
    print(f"\nmacro F1   全片口径 {sum(r['f1_video'] for r in rows) / n:.3f}"
          f"   窗口口径 {sum(r['f1_window'] for r in rows) / n:.3f}{tail}")
    print(f"macro 召回 全片口径 {sum(r['rec_video'] for r in rows) / n:.3f}"
          f"   窗口口径 {sum(r['rec_window'] for r in rows) / n:.3f}")
    print(f"平均 ratio（检出/真值，>1 = 过拆）"
          f" 全片口径 {sum(r['ratio_video'] for r in rows) / n:.2f}"
          f"   窗口口径 {sum(r['ratio_window'] for r in rows) / n:.2f}")


_FCOLS = [("片源", 22), ("风格", 6), ("时长s", 8), ("轨迹", 7), ("全片角色", 10),
          ("窗口上限", 10), ("片段@全片", 12), ("片段@窗口", 12), ("耗时s", 8)]


def print_films(rows: List[Dict]) -> None:
    header = "".join(_pad(n, w) for n, w in _FCOLS)
    print("\n=== 整片口径：这套门槛到底筛掉了多少 ===")
    print(header)
    print("-" * _w(header))
    for r in rows:
        cells = [_pad(r["title"], 22), _pad(r["style"], 6),
                 _pad(f"{r['duration']:.0f}", 8), _pad(r["tracks"], 7),
                 _pad(r["characters"], 10), _pad(r["max_windows"], 10)]
        for scope in SCOPES:
            got = r[f"segments_{scope}"]
            cells.append(_pad(f"{got} ({100 * got / max(1, r['max_windows']):.0f}%)", 12))
        cells.append(_pad(f"{r['seconds']:.0f}", 8))
        print("".join(cells))
    total = sum(r["max_windows"] for r in rows)
    for scope in SCOPES:
        got = sum(r[f"segments_{scope}"] for r in rows)
        print(f"合计 片段@{scope:<6s} {got} / 窗口上限 {total} = "
              f"{100 * got / max(1, total):.0f}% 的片长被判为合格片段")


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(description="片长尺度上对比两种聚类范围。")
    parser.add_argument("--clip-dir", default="data", help="带标注的 30 秒切片目录。")
    parser.add_argument("--source-dir", default="data/原始数据", help="完整片源目录。")
    parser.add_argument("--output-dir", default="output_long", help="输出根目录。")
    parser.add_argument("--titles", nargs="*", help="只跑这些标题（默认全部）。")
    parser.add_argument("--limit-seconds", type=float, help="每部片源只跑前 N 秒（冒烟用）。")
    parser.add_argument("--no-clip-baseline", action="store_true",
                        help="跳过 30 秒切片参照列（省 9 次小扫描）。")
    parser.add_argument("--min-events", type=int, help="覆盖 min_events_per_window。")
    args = parser.parse_args(argv)

    base = Config()
    base.style_routing = False  # 画风用真值，理由见模块文档
    if args.min_events is not None:
        base.min_events_per_window = args.min_events

    pairs = find_pairs(args.clip_dir, args.source_dir)
    if args.titles:
        pairs = {src: gts for src, gts in pairs.items()
                 if os.path.splitext(os.path.basename(src))[0] in args.titles}
    if not pairs:
        print("没有任何切片能配上片源。")
        return 1

    window_rows: List[Dict] = []
    film_rows: List[Dict] = []

    for source, gts in sorted(pairs.items()):
        title = os.path.splitext(os.path.basename(source))[0]
        style = gts[0].style
        print(f"\n########## {title}（风格真值 {style}，{len(gts)} 个有真值的窗口）")
        started = time.time()
        tracks, duration, cuts, index, config, num_chars = scan_once(
            base, source, os.path.join(args.output_dir, title), style, args.limit_seconds
        )

        segments = {}
        for scope in SCOPES:
            probe = dataclasses.replace(config, cluster_scope=scope)
            segs, _qualified = select_segments(tracks, duration, probe, cuts, index)
            segments[scope] = len(segs)
        seconds = time.time() - started
        print(f"[{title}] {len(tracks)} 轨迹 / {num_chars} 全片角色 / "
              f"片段 全片口径 {segments['video']} vs 窗口口径 {segments['window']}")

        film_rows.append({
            "title": title, "style": style, "duration": round(duration, 1),
            "tracks": len(tracks), "characters": num_chars,
            "max_windows": int(duration // config.window_seconds),
            "segments_video": segments["video"], "segments_window": segments["window"],
            "seconds": round(seconds, 1),
        })

        for gt in sorted(gts, key=lambda g: g.source_offset):
            counts = count_in_window(tracks, index, float(gt.source_offset),
                                     config.window_seconds)
            row = {
                "window_id": f"{title}_{gt.source_offset}s",
                "style": style,
                "gt_chars": gt.num_characters,
                "window_tracks": counts["tracks"],
                "pred_clip30": None,
                "f1_clip30": None,
            }
            for scope in SCOPES:
                pred = counts[scope]
                row[f"pred_{scope}"] = pred
                row[f"f1_{scope}"] = round(count_f1(pred, gt.num_characters), 3)
                row[f"rec_{scope}"] = round(recall(pred, gt.num_characters), 3)
                row[f"ratio_{scope}"] = round(pred / gt.num_characters, 2)

            if not args.no_clip_baseline:
                clip_path = os.path.join(args.clip_dir, f"{gt.stem}.mp4")
                *_rest, clip_chars = scan_once(
                    base, clip_path,
                    os.path.join(args.output_dir, "clip30", gt.stem), style
                )
                row["pred_clip30"] = clip_chars
                row["f1_clip30"] = round(count_f1(clip_chars, gt.num_characters), 3)
            window_rows.append(row)

    print_windows(window_rows)
    print_films(film_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, table in (("long_scope.csv", window_rows), ("long_films.csv", film_rows)):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
