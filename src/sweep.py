"""阈值敏感性扫描：同一批视频在不同参数下的输出波动。

干什么：把一个（或多个）视频扫描一遍，然后在 identity_threshold ×
min_events_per_window 的网格上反复重跑角色聚类与选段，打出一张
"同一视频在不同配置下结果差多少"的表。

为什么要有它：本流水线的两个关键参数会互相补偿——identity_threshold 调严，
同一角色被拆成多个身份，角色数虚高，于是更容易越过 min_events_per_window
门槛。两个参数一起动的时候，最终片段数可以在几倍范围内摆动，而这与"画面里
到底有几个角色"没有必然关系。在没有逐窗口人工真值的情况下，单次运行的数字
不构成测量，只是这组阈值的函数。跑一遍本脚本，看波动范围有多宽，再决定
参数值是否站得住。

怎么跑（Windows PowerShell，项目根目录下）：
    $py = "D:\\Programs\\DevEnvironments\\Anaconda\\anaconda3\\envs\\myenv\\python.exe"
    & $py src/sweep.py data/*.mp4
    & $py src/sweep.py <video> --limit-seconds 120
    & $py src/sweep.py data/*.mp4 --threshold 0.05,0.1,0.178 --min-events 4,6,8

每段素材实际用的那个阈值（画风路由定的）一定会出现在它自己那张表里，
所以不同画风的素材行数可能不同——阈值那一列跨画风不可比。

需要什么：与 main.py 相同的依赖（imgutils / opencv / scipy）和 ffprobe。

产出：控制台表格 + <output-dir>/sweep.csv（每次重跑覆盖，幂等）。

代价：视频只解码一次，身份特征只提取一次；网格上的每个格子都只是重跑一次
层次聚类和滑窗计数，几乎免费。所以格子多不影响耗时。
"""

import argparse
import csv
import dataclasses
import os
from typing import List, Optional

from config import Config
from main import (
    IdentityIndex,
    _cluster_by_difference,
    compute_differences,
    force_utf8_stdout,
    get_detector,
    resolve_identity_threshold,
    scan_video,
    select_segments,
)
from style import apply_style, classify_style

# === 默认扫描网格（用 --threshold / --min-events 覆盖）===
# 身份合并阈值。0.178 是 CCIP 自带的默认阈值，作为基准锚点保留。
# 注意画风路由打开时 2D 走 CCIP、非 2D 走 ArcFace，两者尺度不同，
# 同一张网格上的数字跨画风不可比。
DEFAULT_THRESHOLD = [0.05, 0.10, 0.178, 0.25]
# 窗口内所需的不同角色数。9 段素材的真值落在 4~11。
DEFAULT_MIN_EVENTS = [4, 6, 8]


def _parse_floats(text: str) -> List[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def sweep_video(
    config: Config,
    video_path: str,
    output_root: str,
    threshold_values: List[float],
    min_events_values: List[int],
    limit_seconds: Optional[float] = None,
) -> List[dict]:
    """扫描一个视频，返回网格上每个格子的一行结果。

    视频只解码一次（scan_video），身份特征只提取一次（compute_differences），
    之后每个阈值重跑一次聚类、每个 min_events 重跑一次选段。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    if config.style_routing:
        style, _votes = classify_style(video_path, config)
        config = apply_style(config, style)
        print(f"[{stem}] style={style} -> {config.detector} + {config.embedder}")

    tracks, _detections, duration, num_frames, cuts = scan_video(
        config, video_path, out_dir, get_detector(config.detector, config), limit_seconds
    )
    candidates, diff_matrix = compute_differences(tracks, out_dir, config)
    if diff_matrix is None:
        print(f"[{stem}] 没有任何代表裁剪图，跳过。")
        return []
    print(f"[{stem}] {num_frames} frames, {len(candidates)} 条轨迹参与聚类")

    # 保证表里一定有"这段素材现在实际用的那个阈值"这一行。它按画风走，
    # 因此只能在这里逐视频补，不能在 main() 里对全局网格补一次。
    routed = resolve_identity_threshold(config)
    threshold_values = sorted(set(threshold_values) | {routed})

    rows = []
    for threshold in threshold_values:
        labels = _cluster_by_difference(diff_matrix, threshold)
        for track, label in zip(candidates, labels):
            track.character_id = label
        num_characters = len(set(labels))
        # 窗口口径的聚类也要跟着阈值走，所以索引在阈值循环里重建（只是包一层，
        # 差异矩阵本身仍然只算了一次）。
        identity = IdentityIndex(candidates, diff_matrix, threshold, config)
        for min_events in min_events_values:
            probe = dataclasses.replace(config, min_events_per_window=min_events)
            segments, num_qualified = select_segments(
                tracks, duration, probe, cuts, identity
            )
            rows.append(
                {
                    "video": stem,
                    "duration": round(duration, 2),
                    "tracks": len(tracks),
                    "identity_threshold": threshold,
                    "characters": num_characters,
                    "min_events": min_events,
                    "qualified_windows": num_qualified,
                    "segments": len(segments),
                }
            )
    return rows


def print_table(rows: List[dict], min_events_values: List[int]) -> None:
    """按视频分组打印：行=身份阈值，列=min_events，格=片段数。

    最后一行给出该视频在整个网格上的片段数区间——这就是"换一组阈值，
    结论差多少"的直接读数。
    """
    videos = []
    for row in rows:
        if row["video"] not in videos:
            videos.append(row["video"])

    for video in videos:
        subset = [r for r in rows if r["video"] == video]
        duration = subset[0]["duration"]
        tracks = subset[0]["tracks"]
        print(f"\n=== {video}  ({duration}s, {tracks} tracks) ===")
        header = f"{'thresh':>7} {'chars':>7}" + "".join(
            f"{'ME=' + str(m):>8}" for m in min_events_values
        )
        print(header)
        print("-" * len(header))
        for threshold in sorted({r["identity_threshold"] for r in subset}):
            at_threshold = [r for r in subset if r["identity_threshold"] == threshold]
            chars = at_threshold[0]["characters"]
            cells = []
            for min_events in min_events_values:
                match = [r for r in at_threshold if r["min_events"] == min_events]
                cells.append(f"{match[0]['segments']:>8}" if match else f"{'-':>8}")
            print(f"{threshold:>7.3f} {chars:>7}" + "".join(cells))

        counts = [r["segments"] for r in subset]
        char_counts = [r["characters"] for r in subset]
        print(
            f"波动范围：片段数 {min(counts)}~{max(counts)}，"
            f"角色数 {min(char_counts)}~{max(char_counts)}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="扫描 identity_threshold × min_events_per_window，量化输出对阈值的敏感度。"
    )
    parser.add_argument("videos", nargs="+", help="输入视频路径。")
    parser.add_argument("--output-dir", default="output", help="输出根目录。")
    parser.add_argument("--threshold", help=f"逗号分隔的身份阈值。默认 {DEFAULT_THRESHOLD}")
    parser.add_argument("--min-events", help=f"逗号分隔的窗口角色数门槛。默认 {DEFAULT_MIN_EVENTS}")
    parser.add_argument("--frame-interval", type=float, help="覆盖抽帧间隔。")
    parser.add_argument("--limit-seconds", type=float, help="只处理前 N 秒。")
    args = parser.parse_args(argv)

    config = Config()
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval

    threshold_values = _parse_floats(args.threshold) if args.threshold else list(DEFAULT_THRESHOLD)
    min_events_values = _parse_ints(args.min_events) if args.min_events else list(DEFAULT_MIN_EVENTS)
    # 阈值那一行由 sweep_video 逐视频补（它按画风走）；min_events 是全局的，在这里补。
    if config.min_events_per_window not in min_events_values:
        min_events_values.append(config.min_events_per_window)
    min_events_values.sort()

    rows = []
    for video_path in args.videos:
        rows.extend(
            sweep_video(
                config, video_path, args.output_dir,
                threshold_values, min_events_values, args.limit_seconds,
            )
        )

    if not rows:
        print("没有可用结果。")
        return 1

    print_table(rows, min_events_values)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "sweep.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n写入 {csv_path}")
    print(
        "读法：同一视频不同阈值下片段数的跨度越大，说明当前参数越是在"
        "支配结论本身，而不是在测量画面里的角色数。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
