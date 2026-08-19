"""召回率评估：用文件名里的人工标注当真值，量出流水线漏了多少角色。

干什么：对每段素材跑一遍流水线到轨迹层，聚类出角色，与文件名标注的
「跨镜头身份去重后期望」和「逐镜头人脸数期望」对账，打出召回率表。

## 召回率在这里是什么意思

真值只给**数量**（"这 30 秒里有 9 个不同角色"），不给身份，也没有框坐标。
所以做不了逐身份配对的召回，只能算**数量召回**：

    recall = min(检出角色数, 真值角色数) / 真值角色数

这是召回率的**乐观上界**：它默认检出的每个身份都能对上一个不同的真值身份。
实际可能检出 9 个身份、其中 3 个是同一个角色被拆开的（那真实召回更低）。
反过来，检出数超过真值时 recall 被截到 1.0，此时要看 ratio 列——
它是没截断的 检出/真值，>1 说明在过检（同一角色被拆成多个身份）。
所以两列要一起读，单看 recall 会把"把一个角色拆成十个"读成满分。

逐镜头人脸数用同样的口径，只是把每个镜头内的不同角色数加起来再比。

真值本身有噪声（标注时存在大量模棱两可的脸），结论按数量级读。

## 怎么跑（PowerShell，项目根目录）

    $py = "D:\\Programs\\DevEnvironments\\Anaconda\\anaconda3\\envs\\myenv\\python.exe"
    & $py src/evaluate.py data/*.mp4                    # 用 config 当前参数（含画风路由）
    & $py src/evaluate.py data/*.mp4 --threshold 0.05,0.1,0.178,0.25   # 按阈值网格扫召回
    & $py src/evaluate.py data/*.mp4 --eyes 2           # 打开正脸过滤做对照（默认关）

产出：控制台表格 + <output-dir>/recall.csv（每次重跑覆盖，幂等）。

代价：视频只解码一次、CCIP 特征只提一次，阈值网格上每个格子只重跑一次聚类，
所以格子多不影响耗时。
"""

import argparse
import bisect
import csv
import os
from typing import Dict, List, Optional

from config import Config, set_frontal_weight
from groundtruth import GroundTruth, parse_ground_truth
from main import (
    _cluster_by_difference,
    count_characters,
    compute_differences,
    force_utf8_stdout,
    get_detector,
    resolve_identity_threshold,
    scan_video,
)
from style import apply_style, classify_style


def _parse_floats(text: str) -> List[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def recall(pred: int, gt: int) -> float:
    """数量召回：min(pred, gt) / gt。gt 为 0 时约定为 1.0（没什么可漏的）。"""
    if gt <= 0:
        return 1.0
    return min(pred, gt) / gt


def count_f1(pred: int, gt: int) -> float:
    """数量 F1：2·min(pred,gt) / (pred+gt)。

    单看召回没法定参数——把一个角色拆成十个身份，召回是满分 1.0。
    数量精确率 min(pred,gt)/pred 惩罚这种拆分，两者的调和平均刚好化简成
    这个式子。选阈值看这一列，不要看召回列。
    """
    if pred + gt == 0:
        return 1.0
    return 2 * min(pred, gt) / (pred + gt)


def characters_per_shot(tracks, cuts: List[float]) -> List[int]:
    """按 scdet 切镜把轨迹分到各镜头，返回每个镜头内的不同角色数。

    轨迹按设计不会跨切镜（FaceTracker 遇切镜强制断轨），所以用 start_time
    定位镜头即可。character_id 为 None 的轨迹（无代表裁剪图）不计入，
    与选段阶段的口径保持一致。

    返回：
        长度为 len(cuts)+1 的列表，与真值里 `人脸数期望` 的分段一一对应。
    """
    per_shot: List[set] = [set() for _ in range(len(cuts) + 1)]
    for track in tracks:
        if track.character_id is None:
            continue
        per_shot[bisect.bisect_right(cuts, track.start_time)].add(track.character_id)
    return [len(s) for s in per_shot]


def evaluate_one(
    gt: GroundTruth,
    tracks,
    cuts: List[float],
    identity_threshold: float,
    config: Config,
) -> Dict:
    """给定一组已聚类的轨迹，算出这一格的全部指标。

    角色数走 ``count_characters``，因此 ``config.min_character_seconds``（按角色
    算的累计出镜时长门槛）在这里生效。素材本身就是 30 秒 = 一个窗口，所以这里
    的口径与 select_segments 里的窗口口径一致。
    """
    known = [t for t in tracks if t.character_id is not None]
    pred_characters = count_characters(known, [t.character_id for t in known], config)
    shot_faces = characters_per_shot(tracks, cuts)
    pred_faces = sum(shot_faces)

    row = {
        # 用「标题_源片位置」当短名，完整文件名（带标注）在表里太长，读不了。
        "video": f"{gt.title}_{gt.source_offset}s",
        "style_gt": gt.style,
        "threshold": identity_threshold,
        "tracks": len(tracks),
        # 主指标：跨镜头去重后的角色数
        "gt_chars": gt.num_characters,
        "pred_chars": pred_characters,
        "recall_chars": round(recall(pred_characters, gt.num_characters), 3),
        "ratio_chars": round(pred_characters / gt.num_characters, 2) if gt.num_characters else None,
        "f1_chars": round(count_f1(pred_characters, gt.num_characters), 3),
        # 辅助指标：镜头切分与逐镜头主体数
        "gt_shots": gt.num_shots,
        "pred_shots": len(cuts) + 1,
        "gt_faces": gt.total_faces,
        "pred_faces": pred_faces,
        "recall_faces": round(recall(pred_faces, gt.total_faces), 3),
    }
    # 只有镜头数正好对上时，逐镜头对齐才有意义；否则一次错位会污染后面全部镜头。
    if len(shot_faces) == gt.num_shots and gt.total_faces > 0:
        matched = sum(min(p, g) for p, g in zip(shot_faces, gt.shot_faces))
        row["recall_shotwise"] = round(matched / gt.total_faces, 3)
    else:
        row["recall_shotwise"] = None
    return row


def evaluate_video(
    config: Config,
    video_path: str,
    output_root: str,
    threshold_values: Optional[List[float]],
) -> List[Dict]:
    """扫描一段素材，在给定的身份阈值上各算一行召回。

    ``threshold_values`` 为 None 时只跑一格：由画风路由（或 config 默认值）定出的阈值。

    注意阈值网格**跨 embedder 不可比**：2D 走 CCIP（同人差异 ≈0.1~0.2），
    非 2D 走 ArcFace（≈0.4~0.8）。扫网格时看分风格那几列，别看全局那行。
    """
    gt = parse_ground_truth(video_path)
    if gt is None:
        print(f"[{os.path.basename(video_path)}] 文件名里没有标注，跳过评估。")
        return []

    out_dir = os.path.join(output_root, gt.stem)
    os.makedirs(out_dir, exist_ok=True)

    style_pred, votes = "-", {}
    if config.force_style:
        style_pred, votes = config.force_style, {"forced": True}
        config = apply_style(config, style_pred)
        print(f"[{gt.stem[:20]}] 画风人工指定：{style_pred}（真值 {gt.style}）"
              f" -> {config.detector} + {config.embedder}")
    elif config.style_routing:
        style_pred, votes = classify_style(video_path, config)
        config = apply_style(config, style_pred)
        print(f"[{gt.stem[:20]}] 画风判别：{style_pred}（真值 {gt.style}，票数 {votes}）"
              f" -> {config.detector} + {config.embedder}")

    detector = get_detector(config.detector, config)
    tracks, _detections, _duration, num_frames, cuts = scan_video(
        config, video_path, out_dir, detector
    )
    candidates, diff_matrix = compute_differences(tracks, out_dir, config)
    if diff_matrix is None:
        print(f"[{gt.stem[:20]}] 没有任何代表裁剪图，全部角色都漏了。")
        return [
            {
                **evaluate_one(gt, tracks, cuts, threshold or 0.0, config),
                "style_pred": style_pred,
            }
            for threshold in (threshold_values or [None])
        ]
    print(f"[{gt.stem[:20]}] {num_frames} 帧，{len(tracks)} 条轨迹，{len(candidates)} 条参与聚类")

    grid = threshold_values or [resolve_identity_threshold(config)]

    rows = []
    for threshold in grid:
        labels = _cluster_by_difference(diff_matrix, threshold)
        for track, label in zip(candidates, labels):
            track.character_id = label
        row = evaluate_one(gt, tracks, cuts, threshold, config)
        row["style_pred"] = style_pred
        rows.append(row)
    return rows


# === 报表 ===

_COLUMNS = [
    ("video", 20, "s"), ("style_gt", 9, "s"), ("style_pred", 11, "s"),
    ("thresh", 7, ".3f"), ("gt_chars", 9, "d"), ("pred_chars", 11, "d"),
    ("recall", 8, ".2f"), ("ratio", 7, ".2f"), ("f1", 6, ".2f"),
    ("gt_shots", 9, "d"), ("pred_shots", 11, "d"),
    ("gt_faces", 9, "d"), ("pred_faces", 11, "d"), ("rec_faces", 10, ".2f"),
]
# 表头短名 -> 行里的字段名（表头要短才排得下，字段名要长才自解释）。
_HEADER_TO_FIELD = {"recall": "recall_chars", "ratio": "ratio_chars",
                    "f1": "f1_chars", "rec_faces": "recall_faces",
                    "thresh": "threshold"}


def _width(text: str) -> int:
    """字符串在等宽终端里占几列：CJK 全角字符算 2 列。

    中文片名不做这个换算，表格所有列都会错位。
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度右对齐；超宽则从左侧截断（保留信息量更大的尾部）。"""
    while _width(text) > width:
        text = text[1:]
    return " " * (width - _width(text)) + text


def _fmt(value, width: int, spec: str) -> str:
    return _pad("-" if value is None else format(value, spec), width)


def print_table(rows: List[Dict]) -> None:
    header = "".join(_pad(name, w) for name, w, _ in _COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print("".join(
            _fmt(row.get(_HEADER_TO_FIELD.get(n, n)), w, s) for n, w, s in _COLUMNS
        ))


def print_summary(rows: List[Dict], threshold_values: Optional[List[float]]) -> None:
    """按身份阈值 × 画风汇总召回。

    micro = 全部素材的 min(pred,gt) 之和 / 真值之和（大素材权重更大）；
    macro = 各素材召回的算术平均（每段素材等权，9 段样本下这个更有代表性）。

    开着画风路由扫网格时，2D 与非 2D 用的是不同的 embedder，阈值不在同一
    尺度上——只有分风格那几列有意义，全局 macro 那几列没有。
    """
    thresholds = sorted({r["threshold"] for r in rows})
    styles = ["2d", "3d", "real"]

    print("\n=== 角色召回率汇总（按 identity_threshold）===")
    print("列：macro/micro 召回，过检段数，F1（选阈值看这个），以及分风格的召回 | F1")
    cols = [("thresh", 7), ("macro", 8), ("micro", 8), ("过检段数", 10),
            ("macroF1", 9)] + [(s, 14) for s in styles]
    header = "".join(_pad(name, w) for name, w in cols)
    print(header)
    print("-" * _width(header))
    best = None
    for threshold in thresholds:
        subset = [r for r in rows if r["threshold"] == threshold]
        macro = sum(r["recall_chars"] for r in subset) / len(subset)
        micro = sum(min(r["pred_chars"], r["gt_chars"]) for r in subset) / sum(
            r["gt_chars"] for r in subset
        )
        macro_f1 = sum(r["f1_chars"] for r in subset) / len(subset)
        # 过检段数：检出角色数超过真值的素材数。macro 召回被截断在 1.0 看不出
        # 这件事，必须单列一栏，否则"把一个角色拆成十个"会显示成满分。
        over = sum(1 for r in subset if r["pred_chars"] > r["gt_chars"])
        if best is None or macro_f1 > best[1]:
            best = (threshold, macro_f1)
        cells = [_pad(f"{threshold:.3f}", 7), _pad(f"{macro:.2f}", 8),
                 _pad(f"{micro:.2f}", 8), _pad(str(over), 10),
                 _pad(f"{macro_f1:.2f}", 9)]
        for style in styles:
            group = [r for r in subset if r["style_gt"] == style]
            cells.append(_pad(
                f"{sum(r['recall_chars'] for r in group) / len(group):.2f}"
                f" | {sum(r['f1_chars'] for r in group) / len(group):.2f}"
                if group else "-", 14))
        print("".join(cells))

    if len(thresholds) > 1:
        print(f"\n全局最佳（按 macroF1）：threshold={best[0]:.3f}，macroF1={best[1]:.2f}")
        # 分风格各自挑最佳阈值——这正是画风路由存在的理由，两者不同才说明路由有用。
        for style in styles:
            per_threshold = []
            for threshold in thresholds:
                group = [r for r in rows if r["threshold"] == threshold and r["style_gt"] == style]
                if group:
                    per_threshold.append((sum(r["f1_chars"] for r in group) / len(group), threshold))
            if per_threshold:
                f1, threshold = max(per_threshold)
                print(f"  {style:>5} 最佳 threshold={threshold:.3f}（F1={f1:.2f}）")

    if threshold_values is None:
        print("\n（以上只有一行：用的是 config 当前参数 + 画风路由后的阈值，不是网格扫描。）")

    routed = [r for r in rows if r.get("style_pred") not in (None, "-")]
    if routed:
        correct = sum(1 for r in routed if r["style_pred"] == r["style_gt"])
        print(f"\n画风路由准确率：{correct}/{len(routed)}（2d / 3d / real 三分类）")


def main(argv: Optional[List[str]] = None) -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="用文件名标注当真值，测量各素材的角色召回率。"
    )
    parser.add_argument("videos", nargs="+", help="输入视频路径。")
    parser.add_argument("--output-dir", default="output", help="输出根目录。")
    parser.add_argument(
        "--threshold",
        help="逗号分隔的身份阈值网格。不填则只跑一格（config 参数 + 画风路由）。"
             "跨 embedder 不可比：CCIP ≈0.1~0.2，ArcFace ≈0.4~0.8。",
    )
    parser.add_argument("--frame-interval", type=float, help="覆盖抽帧间隔。")
    # 下面这几个覆盖项配合 --no-style-routing 使用：关掉路由后，就能把同一套
    # 检测器/特征/门槛压到全部 9 段素材上跑对照，再看分风格那几列。
    # 路由开着时它们会被 profile 覆盖掉（profile 是成套的，见 config.StyleProfile）。
    parser.add_argument("--crop-margin", type=float, help="覆盖代表裁剪图外扩比例。")
    parser.add_argument("--detector", help="覆盖检测器名。")
    parser.add_argument("--embedder", help="覆盖身份特征名。")
    parser.add_argument("--min-face-height", type=float, help="覆盖人脸高度占比下限。")
    parser.add_argument(
        "--frontal-weight", type=float, metavar="W",
        help="正脸分在代表图擂台里的权重（0=关）。与画风路由兼容：会同时压到每个 profile 上。",
    )
    parser.add_argument(
        "--min-frontal", type=float, metavar="S",
        help="正脸硬门槛：分数低于 S 的人脸框直接丢弃（0=关）。",
    )
    parser.add_argument(
        "--min-track-seconds", type=float, metavar="S",
        help="最短出镜时长（秒）：短于 S 的轨迹整条丢弃（0=关）。按轨迹算，已实测有害。",
    )
    parser.add_argument(
        "--min-character-seconds", type=float, metavar="S",
        help="按**角色**算的最短累计出镜时长（秒），聚类之后生效（0=关）。对应标注口径「出镜 >1s」。",
    )
    parser.add_argument(
        "--eyes", type=int, metavar="N",
        help="覆盖正脸过滤门槛（人脸上至少 N 只眼）。默认 0=关闭；传 2 可复现对照实验。",
    )
    parser.add_argument(
        "--no-style-routing", action="store_true", help="关掉画风路由，所有素材共用一个阈值。"
    )
    parser.add_argument("--style", choices=["2d", "3d", "real"],
                        help="人工指定画风 profile，跳过自动判别。")
    args = parser.parse_args(argv)

    config = Config()
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    if args.crop_margin is not None:
        config.crop_margin = args.crop_margin
    if args.detector is not None:
        config.detector = args.detector
    if args.embedder is not None:
        config.embedder = args.embedder
    if args.min_face_height is not None:
        config.min_face_height_ratio = args.min_face_height
    if args.eyes is not None:
        config.require_eyes = args.eyes
    if args.frontal_weight is not None:
        set_frontal_weight(config, args.frontal_weight)
    if args.min_frontal is not None:
        config.min_frontal_score = args.min_frontal
    if args.min_track_seconds is not None:
        config.min_track_seconds = args.min_track_seconds
    if args.min_character_seconds is not None:
        config.min_character_seconds = args.min_character_seconds
    if args.no_style_routing:
        config.style_routing = False
    if args.style is not None:
        config.force_style = args.style

    threshold_values = _parse_floats(args.threshold) if args.threshold else None

    rows: List[Dict] = []
    for video_path in args.videos:
        rows.extend(
            evaluate_video(config, video_path, args.output_dir, threshold_values)
        )

    if not rows:
        print("没有任何带标注的素材，无法评估。")
        return 1

    print_table(rows)
    print_summary(rows, threshold_values)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "recall.csv")
    fieldnames = list(rows[0])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n写入 {csv_path}")
    print(
        "读法：recall_chars 被截断在 1.0，必须和 ratio_chars 一起看——"
        "ratio>1 表示同一角色被拆成了多个身份，此时的满分召回是假的。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
