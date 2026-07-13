"""动漫脸剪辑器的端到端流程。

从动漫视频中找出并截取所有“合格”的 15 秒片段。当一个片段窗口内出现过
至少 min_events_per_window 个不同角色时，该片段视为合格。一条轨迹由相邻
帧中 IoU 重叠的人脸框串联而成（镜头切换会强制断开轨迹）；再用 CCIP 特征
对各轨迹的代表裁剪图聚类，得到轨迹级角色身份（character_id），同一角色的
多条轨迹在窗口内只计一次。

从项目根目录运行：

    python src/main.py                      # 处理 data/1.mp4 -> output/1/
    python src/main.py data/1.mp4 --viz 8   # 同时导出带标注的示例帧

阶段（按下方分节注释组织）：
    抽帧 -> 检测 -> 过滤 -> 跟踪 -> 角色识别 -> 选段 -> 截取
"""

import argparse
import bisect
import dataclasses
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from config import Config
from detectors import Detection, Detector, get_detector

# 重新导出，方便调用方使用 from src.main import Detection。
# 该类定义在 detectors.py 中，以避免检测器模块出现循环依赖。
__all__ = ["Detection", "Track", "process_video", "run_pipeline", "main"]


@dataclasses.dataclass
class Track:
    """单次人脸出现事件：按时间串联的一组检测结果。

    属性：
        track_id: 单个视频内的唯一 id。
        label: 从检测结果继承的类别。
        start_time: 首次检测的时间戳（用于窗口计数）。
        end_time: 最后一次检测的时间戳。
        detections: 按时间顺序排列的成员检测结果。
        representative_frame: 最清晰检测结果所在的帧索引。
        representative_time: 该检测结果的时间戳。
        representative_bbox: 该检测结果的框（用于从视频中重新定位）。
        representative_crop: 保存的裁剪图路径（相对于输出目录）。
        character_id: CCIP 聚类得到的角色簇编号；None 表示身份未知
            （无代表裁剪图或读取失败），不参与窗口内的角色计数。
    """

    track_id: int
    label: str
    start_time: float
    end_time: float
    detections: List[Detection]
    representative_frame: int = -1
    representative_time: float = 0.0
    representative_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    representative_crop: str = ""
    character_id: Optional[int] = None


# 模块级保护，确保只报告一次当前使用的 ONNX providers。
_providers_reported = False


# === 1. 抽帧 ===

def extract_frames(config: Config, video_path: str, frames_dir: str) -> List[Tuple[int, float, str]]:
    """使用 ffmpeg 按固定间隔采样帧。

    参数：
        config: 流程配置。
        video_path: 源视频。
        frames_dir: 已存在的目录，JPEG 会写入其中。

    返回：
        按时间顺序排列的 (frame_index, time_seconds, frame_path) 列表。
    """
    pattern = os.path.join(frames_dir, "%06d.jpg")
    cmd = [
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"fps=1/{config.frame_interval}",
        "-q:v", "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")
    )
    frames = []
    for index, name in enumerate(files):
        time = index * config.frame_interval
        frames.append((index, time, os.path.join(frames_dir, name)))
    return frames


def compute_hsv_hist(image_bgr):
    """计算归一化 HSV（H、S）直方图，把每帧压成一个颜色直方图，用于镜头切换比较。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# === 3. 过滤 ===

def laplacian_variance(image_bgr, bbox: Tuple[int, int, int, int]) -> float:
    """边界框裁剪图的拉普拉斯方差（聚焦/模糊度量）。

    对空裁剪或退化裁剪返回 0.0。
    """
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1) # 把框裁回画面边界内
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: # 空框/退化框
        return 0.0
    crop = image_bgr[y1:y2, x1:x2] # 只取人脸框那块，因为我们只关心脸糊不糊，而不是整帧。
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # 格式转换。把彩色(3 通道 BGR)变成单通道灰度图。因为拉普拉斯算子是作用在单通道亮度上的,彩色三通道没必要分别算。
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) # 用拉普拉斯方差计算并返回清晰度。越清晰值越大


def passes_quality(detection: Detection, frame_height: int, config: Config) -> bool:
    """应用三道质量门槛：置信度、人脸大小、清晰度。"""
    if detection.confidence < config.conf_threshold:
        return False
    face_height = detection.bbox[3] - detection.bbox[1]
    if face_height < config.min_face_height_ratio * frame_height:
        return False
    if (detection.blur_var or 0.0) < config.blur_var_threshold:
        return False
    return True


# === 4. 跟踪 ===

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """两个框 (x1, y1, x2, y2) 的交并比。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cut_between(is_cut: List[bool], last_index: int, current_index: int) -> bool:
    """如果在帧区间 (last_index, current_index] 内发生镜头切换，则返回 True。"""
    return any(is_cut[last_index + 1:current_index + 1])


def track_faces(
    frame_detections: List[List[Detection]],
    is_cut: List[bool],
    config: Config,
) -> List[Track]:
    """使用 IoU 和镜头切换断轨，将逐帧检测结果连接为轨迹。

    相邻检测结果在标签相同且 IoU >= iou_threshold 时会加入同一条轨迹。
    最多允许丢失 track_gap_tolerance 帧。两帧之间如果发生镜头切换，
    即使 IoU 很高也禁止跨越切换连接（带丢帧容忍的重连也不能跨越切换）。

    参数：
        frame_detections: 每帧中通过质量过滤的检测结果列表。
        is_cut: 每帧标记；is_cut[i] 表示第 i-1 帧和第 i 帧之间有切换。
        config: 流程配置。

    返回：
        所有轨迹，按开始时间排序。
    """
    active: List[Dict] = []  # 当前"还活着"、可能继续延伸的轨迹。每项：{id, label, last_index, dets:[Detection]}
    finalized: List[Track] = [] # 已经封存、不再延伸的轨迹
    next_id = 1 # 轨迹 id 自增计数器

    def _finalize(track: Dict) -> None:
        dets = track["dets"]
        finalized.append(
            Track(
                track_id=track["id"],
                label=track["label"],
                start_time=dets[0].time,
                end_time=dets[-1].time,
                detections=dets,
            )
        )

    for i, dets in enumerate(frame_detections):
        # 丢弃无法再恢复的轨迹：丢帧超过容忍值，或其最后检测到当前帧之间已有切换。
        still_active = []
        for tr in active:
            gap = i - tr["last_index"] - 1
            if gap > config.track_gap_tolerance or _cut_between(is_cut, tr["last_index"], i):
                _finalize(tr)
            else:
                still_active.append(tr)
        active = still_active

        # 贪心 IoU 匹配：最佳配对优先，每条轨迹和每个检测结果只使用一次。
        pairs = []
        for ti, tr in enumerate(active):
            last_box = tr["dets"][-1].bbox
            for di, det in enumerate(dets):
                if det.label != tr["label"]:
                    continue
                score = iou(last_box, det.bbox) # 计算两个框 (x1,y1,x2,y2) 的交并比
                if score >= config.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            active[ti]["dets"].append(dets[di])
            active[ti]["last_index"] = i

        # 未匹配任何轨迹的框，说明是一张新出现的脸，开一条新轨迹，分配next_id。
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            active.append({"id": next_id, "label": det.label, "last_index": i, "dets": [det]})
            next_id += 1

    for tr in active:
        _finalize(tr)

    finalized.sort(key=lambda t: t.start_time)
    return finalized


def assign_representatives(tracks: List[Track], frame_paths: Dict[int, str], crops_dir: str) -> None:
    """为每条轨迹选择最清晰的检测结果，并保存其裁剪图。

    代表检测结果是在该轨迹内使 blur_var * confidence 最大的检测结果。
    裁剪图会从对应采样帧中读回并写入 crops_dir；轨迹会记录路径和源位置。
    """
    os.makedirs(crops_dir, exist_ok=True)
    for track in tracks:
        best = max(
            track.detections,
            key=lambda d: (d.blur_var or 0.0) * d.confidence,
        )
        track.representative_frame = best.frame_index
        track.representative_time = best.time
        track.representative_bbox = best.bbox

        frame_path = frame_paths.get(best.frame_index)
        if not frame_path:
            continue
        image = cv2.imread(frame_path)
        if image is None:
            continue
        x1, y1, x2, y2 = best.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_name = f"track_{track.track_id}.jpg"
        cv2.imwrite(os.path.join(crops_dir, crop_name), image[y1:y2, x1:x2])
        track.representative_crop = os.path.join("crops", crop_name)


# === 4.5 角色识别 ===

def _cluster_by_difference(diff_matrix, threshold: float) -> List[int]:
    """按 complete-linkage（全连接）层次聚类分簇。

    合并规则：簇内任意两元素差异都 < threshold 时才允许处于同一簇。
    不做传递合并：a~b 且 b~c 但 a、c 差异超阈值时，a、c 不会同簇
    （旧实现用并查集传递合并，差异链会把所有元素塌缩进一个簇）。

    参数：
        diff_matrix: N×N 对称差异矩阵（列表或 numpy 数组均可）。
        threshold: 严格小于该值的差异才允许合并。

    返回：
        长度为 N 的簇编号列表（编号从 0 开始，按首次出现顺序分配）。
    """
    n = len(diff_matrix)
    if n == 0:  # scipy 对 n<2 会抛错，前置短路
        return []
    if n == 1:
        return [0]

    matrix = np.array(diff_matrix, dtype=float)  # 拷贝，避免 fill_diagonal 改到调用方的矩阵
    np.fill_diagonal(matrix, 0.0)
    condensed = squareform(matrix, checks=False)
    tree = linkage(condensed, method="complete")
    # fcluster 的合并条件是 <= t，而这里的语义是严格 < threshold，
    # 因此把切割点取到 threshold 之下最近的浮点数。
    cut = math.nextafter(threshold, 0.0)
    raw = fcluster(tree, t=cut, criterion="distance")

    # fcluster 的簇编号从 1 开始且顺序任意，按首次出现顺序重映射为从 0 开始。
    remap: Dict[int, int] = {}
    labels = []
    for cluster in raw:
        if cluster not in remap:
            remap[cluster] = len(remap)
        labels.append(remap[cluster])
    return labels


def assign_characters(tracks: List[Track], out_dir: str, config: Config) -> int:
    """用 CCIP 对轨迹代表裁剪图聚类，给每条轨迹写入 character_id。

    对每条有代表裁剪图的轨迹批量提取 CCIP 特征，做 complete-linkage
    层次聚类（簇内任意两张裁剪图差异都 < 阈值），同一簇视为同一角色。
    无裁剪图（或文件缺失）的轨迹保持 character_id=None，不参与后续
    窗口内的角色计数。

    阈值取 config.ccip_threshold；为 None 时使用 imgutils 的
    ccip_default_threshold()（约 0.178）。

    返回：
        识别出的不同角色总数。
    """
    # 延迟导入：CCIP 模型较重且首次使用需从 HuggingFace 下载，
    # 与 detectors.py 的延迟导入风格一致。
    from imgutils.metrics import (
        ccip_batch_differences,
        ccip_batch_extract_features,
        ccip_default_threshold,
    )

    candidates: List[Track] = []
    crop_paths: List[str] = []
    for track in tracks:
        if not track.representative_crop:
            continue
        crop_path = os.path.join(out_dir, track.representative_crop)
        if not os.path.isfile(crop_path):
            continue
        candidates.append(track)
        crop_paths.append(crop_path)

    if not candidates:
        return 0

    threshold = config.ccip_threshold
    if threshold is None:
        threshold = ccip_default_threshold()

    # 分批提取：一次性送入几百张图会让 ONNX 推理内存分配失败（bad allocation）。
    batch_size = 32
    feature_batches = [
        ccip_batch_extract_features(crop_paths[i:i + batch_size])
        for i in range(0, len(crop_paths), batch_size)
    ]
    features = np.concatenate(feature_batches)
    diff_matrix = ccip_batch_differences(features)
    cluster_ids = _cluster_by_difference(diff_matrix, threshold)
    for track, cluster_id in zip(candidates, cluster_ids):
        track.character_id = cluster_id
    return len(set(cluster_ids))


# === 5. 选段 ===

def select_segments(
    tracks: List[Track],
    duration: float,
    config: Config,
) -> Tuple[List[Dict], int]:
    """滑动窗口统计出现过的不同角色数，并贪心选择片段。

    候选窗口起点按 frame_interval（抽帧间隔）步进。窗口 [t, t+W) 内
    “出现过”的轨迹指时间区间与窗口相交的轨迹（start_time < t+W 且
    end_time >= t，包括窗口开始前就在画面中的角色）。这些轨迹中不同
    character_id 的数量达到 min_events_per_window 时窗口合格；
    character_id 为 None（身份未知）的轨迹不参与角色计数。遇到合格
    窗口时输出对应片段，下一个候选窗口跳到 >= t+W，从而保证片段不重叠。

    返回：
        元组 (segments, num_qualified_windows)，其中每个片段都是包含
        start、end、character_count、character_ids 和 track_ids 的字典。
    """
    ordered = sorted(tracks, key=lambda tr: tr.start_time)
    starts = [tr.start_time for tr in ordered]
    window = config.window_seconds
    step = config.frame_interval

    segments: List[Dict] = []
    num_qualified = 0
    k = 0
    while True:
        t = k * step
        if t + window > duration + 1e-6:
            break
        hi = bisect.bisect_left(starts, t + window) # 二分取前缀：下标 [0, hi) 的轨迹满足 start_time < t+window（窗口右端开区间）
        overlapping = [tr for tr in ordered[:hi] if tr.end_time >= t] # 再线性筛掉窗口开始前就已结束的轨迹，剩下的即与窗口相交
        character_ids = sorted(
            {tr.character_id for tr in overlapping if tr.character_id is not None}
        )
        if len(character_ids) >= config.min_events_per_window:
            num_qualified += 1
            segments.append(
                {
                    "start": round(t, 3),
                    "end": round(t + window, 3),
                    "character_count": len(character_ids),
                    "character_ids": character_ids,
                    "track_ids": [tr.track_id for tr in overlapping],
                }
            )
            k = math.ceil((t + window) / step - 1e-9) # math.ceil((t + window) / step)取最小的满足条件的整数，后面的 - 1e-9 是为了浮点防抖，防浮点误差
        else:
            k += 1
    return segments, num_qualified


# === 6. 截取 ===

def _encode_clip(config: Config, video_path: str, start: float, out_path: str, encoder: str) -> bool:
    """使用指定编码器截取一个重新编码且帧精确的片段。"""
    cmd = [
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{config.window_seconds:.3f}",
        "-c:v", encoder,
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
    return result.returncode == 0


def clip_segments(config: Config, video_path: str, segments: List[Dict], clips_dir: str) -> List[str]:
    """截取所有选中的片段，优先使用 GPU 编码器，失败时回退到 CPU。"""
    os.makedirs(clips_dir, exist_ok=True)
    encoder = config.encoder
    out_paths = []
    for idx, segment in enumerate(segments, start=1):
        out_path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
        ok = _encode_clip(config, video_path, segment["start"], out_path, encoder)
        if not ok and encoder != config.encoder_fallback:
            print(f"  encoder {encoder!r} failed, falling back to {config.encoder_fallback!r}")
            encoder = config.encoder_fallback
            ok = _encode_clip(config, video_path, segment["start"], out_path, encoder)
        if ok:
            out_paths.append(out_path)
        else:
            print(f"  failed to cut segment {idx} at {segment['start']}s")
    return out_paths


# === 工具 ===

def probe_duration(config: Config, video_path: str) -> float:
    """通过 ffprobe 返回视频时长（秒）。"""
    cmd = [
        config.ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _detection_record(det: Detection, kept: bool) -> Dict:
    return {
        "frame_index": det.frame_index,
        "time": round(det.time, 3),
        "bbox": list(det.bbox),
        "confidence": round(det.confidence, 4),
        "label": det.label,
        "blur_var": round(det.blur_var, 2) if det.blur_var is not None else None,
        "kept": kept,
    }


def _track_record(track: Track) -> Dict:
    return {
        "track_id": track.track_id,
        "label": track.label,
        "start_time": round(track.start_time, 3),
        "end_time": round(track.end_time, 3),
        "num_detections": len(track.detections),
        "representative_frame": track.representative_frame,
        "representative_time": round(track.representative_time, 3),
        "representative_bbox": list(track.representative_bbox),
        "representative_crop": track.representative_crop,
        "character_id": track.character_id,
    }


def _report_providers(detector: Detector) -> None:
    """只打印一次 ONNX providers，用于确认 GPU 使用情况。"""
    global _providers_reported
    if _providers_reported:
        return
    _providers_reported = True
    try:
        import onnxruntime

        print(f"  onnxruntime available providers: {onnxruntime.get_available_providers()}")
    except Exception:
        pass
    active = detector.actual_providers()
    if active:
        print(f"  active session providers: {active}")


def _save_visualization(viz_dir: str, frame_path: str, detections: List[Detection]) -> None:
    """在帧上绘制框和分数并保存。"""
    image = cv2.imread(frame_path)
    if image is None:
        return
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"{det.confidence:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    os.makedirs(viz_dir, exist_ok=True)
    cv2.imwrite(os.path.join(viz_dir, os.path.basename(frame_path)), image)


# === 主流程 ===

def process_video(
    config: Config,
    video_path: str,
    output_root: str,
    detector: Optional[Detector] = None,
    limit_seconds: Optional[float] = None,
    viz_count: int = 0,
    keep_frames: bool = False,
) -> Dict:
    """对单个视频运行完整流程。

    参数：
        config: 流程配置。
        video_path: 源视频路径。
        output_root: 基础输出目录；结果会写入 <root>/<stem>/。
        detector: 共享检测器实例（为 None 时创建）。传入该实例可让批处理复用
            同一个已加载模型。
        limit_seconds: 如果设置，只处理该时间戳之前的帧（用于快速校准）。
        viz_count: 随机导出的标注示例帧数量。
        keep_frames: 保留临时抽取帧，而不是删除。

    返回：
        摘要字典（也会持久化到 JSON 文件中）。
    """
    if detector is None:
        detector = get_detector(config.detector, config)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = tempfile.mkdtemp(prefix=f"afc_{stem}_")

    try:
        print(f"[{stem}] extracting frames -> {frames_dir}")
        frames = extract_frames(config, video_path, frames_dir)
        if limit_seconds is not None:
            frames = [f for f in frames if f[1] < limit_seconds]
        print(f"[{stem}] {len(frames)} frames; detecting + filtering...")

        frame_paths: Dict[int, str] = {}
        frame_detections: List[List[Detection]] = []
        is_cut: List[bool] = []
        detection_records: List[Dict] = []
        prev_hist = None
        viz_candidates: List[Tuple[str, List[Detection]]] = []

        # 2. 检测 + 镜头切换标记
        for index, time, path in frames: # 阶段1的extract_frames函数产出的数据结构是List[(frame_index, time_seconds, frame_path)]，和这里的index, time, path对应
            frame_paths[index] = path
            image = cv2.imread(path)
            if image is None:
                is_cut.append(False)
                frame_detections.append([])
                continue
            frame_h = image.shape[0]

            # 与上一采样帧比较得到镜头切换标记。
            hist = compute_hsv_hist(image) # 把每帧压成一个颜色直方图(一个 numpy 数组)
            if prev_hist is None:
                is_cut.append(False)
            else:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                is_cut.append(corr < config.scene_cut_threshold)
            prev_hist = hist # prev_hist 只保留最近一帧的直方图,下一轮被新的覆盖。所以任意时刻内存里最多只有 2 个直方图

            # 先检测，再应用三道质量门槛。
            raw = detector.detect(path, index, time) # 阶段2拿到原始检测框
            _report_providers(detector)
            kept = []
            for det in raw:
                det.blur_var = laplacian_variance(image, det.bbox) # 计算清晰度
                ok = passes_quality(det, frame_h, config)
                detection_records.append(_detection_record(det, ok))
                if ok:
                    kept.append(det)
            frame_detections.append(kept)
            if kept:
                viz_candidates.append((path, kept))

        # 带镜头切换断轨的跟踪。
        print(f"[{stem}] tracking...")
        tracks = track_faces(frame_detections, is_cut, config)
        crops_dir = os.path.join(out_dir, "crops")
        assign_representatives(tracks, frame_paths, crops_dir)

        # 角色识别：CCIP 聚类给轨迹分配 character_id。
        print(f"[{stem}] identifying characters...")
        num_characters = assign_characters(tracks, out_dir, config)
        print(f"[{stem}] {num_characters} distinct characters identified")

        # 片段选择。
        duration = probe_duration(config, video_path)
        if limit_seconds is not None:
            duration = min(duration, limit_seconds)
        segments, num_qualified = select_segments(tracks, duration, config)
        print(
            f"[{stem}] {len(tracks)} tracks, {num_qualified} qualified windows, "
            f"{len(segments)} segments selected"
        )

        # 截取片段。
        clips_dir = os.path.join(out_dir, "clips")
        clip_paths = clip_segments(config, video_path, segments, clips_dir)

        # 可选的检测可视化。
        if viz_count > 0 and viz_candidates:
            viz_dir = os.path.join(out_dir, "viz")
            sample = random.sample(viz_candidates, min(viz_count, len(viz_candidates)))
            for path, dets in sample:
                _save_visualization(viz_dir, path, dets)

        # 持久化输出。
        _write_json(os.path.join(out_dir, "detections.json"), detection_records)
        _write_json(
            os.path.join(out_dir, "tracks.json"),
            [_track_record(t) for t in tracks],
        )
        _write_json(
            os.path.join(out_dir, "windows.json"),
            {
                "video": video_path,
                "duration": round(duration, 3),
                "num_tracks": len(tracks),
                "num_qualified_windows": num_qualified,
                "params": dataclasses.asdict(config),
                "segments": segments,
                "clips": [os.path.relpath(p, out_dir) for p in clip_paths],
            },
        )

        summary = {
            "video": video_path,
            "frames": len(frames),
            "tracks": len(tracks),
            "qualified_windows": num_qualified,
            "segments": len(segments),
            "clips": len(clip_paths),
            "output_dir": out_dir,
        }
        print(f"[{stem}] done: {summary}")
        return summary
    finally:
        if keep_frames:
            print(f"[{stem}] frames kept at {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)


def run_pipeline(
    config: Config,
    video_paths: List[str],
    output_root: str,
    **kwargs,
) -> List[Dict]:
    """处理一个或多个视频，并复用单个已加载的检测器。

    v1 中的批处理支持有意保持最小化：这里保留循环和共享检测器，
    CLI 只传入单个视频。
    """
    detector = get_detector(config.detector, config)
    summaries = []
    for video_path in video_paths:
        summaries.append(
            process_video(config, video_path, output_root, detector=detector, **kwargs)
        )
    return summaries


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime face clipper.")
    # 位置参数：输入视频，可传多个；不填则处理 data/1.mp4。例：python src/main.py a.mp4 b.mp4
    parser.add_argument(
        "videos", nargs="*", default=["data/1.mp4"],
        help="Input video path(s). Default: data/1.mp4",
    )
    # 输出根目录，默认 output。
    parser.add_argument("--output-dir", default="output", help="Output base directory.")
    # 便于校准的覆盖参数：不填则用 config.py 中的默认值（见 main() 应用逻辑）。
    parser.add_argument("--conf", type=float, help="Override conf_threshold.") # 置信度阈值（默认 0.5）。调高更严格：误检少、漏检多。
    parser.add_argument("--blur-var", type=float, help="Override blur_var_threshold.") # 模糊过滤的拉普拉斯方差下限（默认 50.0）。调高丢弃更多模糊/拖影脸。
    parser.add_argument("--scene-cut", type=float, help="Override scene_cut_threshold.") # 镜头切换阈值（默认 0.6）。调低则检测到的切换更少。
    parser.add_argument("--min-events", type=int, help="Override min_events_per_window.") # 窗口内所需不同角色数（默认 13）。调低则更多片段合格、出片更多。
    parser.add_argument("--ccip-threshold", type=float, help="Override ccip_threshold.") # CCIP 角色合并阈值（默认用模型自带阈值 ≈0.178）。调高更容易把不同轨迹合并为同一角色。
    parser.add_argument("--frame-interval", type=float, help="Override frame_interval.") # 抽帧间隔秒数（默认 0.3）。调小则采样更密、更慢更准。
    parser.add_argument("--encoder", help="Override video encoder (e.g. libx264).") # 视频编码器（默认 h264_nvenc）。失败会自动回退到 libx264；无 GPU 时显式传 libx264。
    # 运行 / 调试参数。
    parser.add_argument("--limit-seconds", type=float, help="Only process first N seconds.") # 只处理前 N 秒；调参时先跑短片段很有用。
    parser.add_argument("--viz", type=int, default=0, help="Dump N annotated sample frames.") # 导出 N 张带标注的样本帧，用于肉眼检查检测/过滤效果（默认 0，不导出）。
    parser.add_argument("--keep-frames", action="store_true", help="Keep temp frames.") # 保留临时抽帧目录（默认清理），便于排查。
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口。"""
    args = _build_arg_parser().parse_args(argv)

    config = Config()
    if args.conf is not None:
        config.conf_threshold = args.conf
    if args.blur_var is not None:
        config.blur_var_threshold = args.blur_var
    if args.scene_cut is not None:
        config.scene_cut_threshold = args.scene_cut
    if args.min_events is not None:
        config.min_events_per_window = args.min_events
    if args.ccip_threshold is not None:
        config.ccip_threshold = args.ccip_threshold
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    if args.encoder is not None:
        config.encoder = args.encoder

    run_pipeline(
        config,
        args.videos,
        args.output_dir,
        limit_seconds=args.limit_seconds,
        viz_count=args.viz,
        keep_frames=args.keep_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
