"""动漫脸剪辑器的端到端流程。

从动漫视频中找出并截取所有“合格”的 15 秒片段。当一个片段窗口内出现过
至少 min_events_per_window 个不同角色时，该片段视为合格。一条轨迹由相邻
帧中 IoU 重叠的人脸框串联而成（镜头切换会强制断开轨迹）；再用 CCIP 特征
对各轨迹的代表裁剪图聚类，得到轨迹级角色身份（character_id），同一角色的
多条轨迹在窗口内只计一次。

流式处理：帧由 cv2.VideoCapture 顺序解码，用完即弃，全程不落盘。整条流水线
在内存中同时持有的图像只有“当前帧 + 每条活跃轨迹一张人脸裁剪图”，与视频
长度无关。代表裁剪图必须在检测的当帧就地留下——流式下没有第二次机会回头
读取任意一帧。

从项目根目录运行：

    python src/main.py                      # 处理 data/1.mp4 -> output/1/
    python src/main.py data/1.mp4 --viz 8   # 同时导出带标注的示例帧

阶段（按下方分节注释组织）：
    解码抽帧 -> 检测 -> 过滤 -> 跟踪 -> 角色识别 -> 选段 -> 截取

阈值敏感性：本流水线的输出对 ccip_threshold 与 min_events_per_window 高度
敏感，两者会互相补偿（阈值调严 -> 同一角色被拆成多个 -> 角色数虚高 ->
更容易越过 min_events 门槛）。在无逐窗口真值标注的情况下，单次运行的数字
不构成对“真实角色数”的测量。定参数前请先用 sweep.py 扫一遍敏感性曲线。
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
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
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
        representative_bbox: 该检测结果的框（仅作溯源记录）。
        representative_image: 跟踪时在线留存的 BGR 裁剪图；写盘后置 None。
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
    representative_image: Optional[np.ndarray] = None
    representative_crop: str = ""
    character_id: Optional[int] = None


# 模块级保护，确保只报告一次当前使用的 ONNX providers。
_providers_reported = False


# === 1. 解码抽帧 ===

def iter_frames(
    config: Config,
    video_path: str,
    limit_seconds: Optional[float] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """顺序解码视频，按 frame_interval 采样，逐帧产出 BGR 图像。

    用 grab()/retrieve() 而非 seek：长 GOP 的 H.264 上 seek 会被吸附到关键帧，
    既不准也不快。grab() 跳过不需要的帧的色彩转换与内存拷贝，retrieve() 只在
    命中采样点时才真正取出图像。

    时间戳取 raw_index / fps 而非“采样序号 * 间隔”——后者在 fps 非整除时会
    随视频长度线性漂移，直接污染窗口边界与最终切片位置。

    参数：
        config: 流程配置。
        video_path: 源视频。
        limit_seconds: 若给定，到达该时间戳即停止解码（不是解码完再过滤）。

    产出：
        (sample_index, time_seconds, frame_bgr)，sample_index 从 0 连续递增。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            raise RuntimeError(f"视频未报告有效帧率（fps={fps!r}）：{video_path}")

        sample_index = 0
        raw_index = 0
        next_sample_time = 0.0
        while True:
            if not cap.grab():
                break
            time = raw_index / fps
            raw_index += 1
            if time + 1e-9 < next_sample_time:
                continue
            if limit_seconds is not None and time >= limit_seconds:
                break
            ok, frame = cap.retrieve()
            if not ok:
                continue
            next_sample_time += config.frame_interval
            yield sample_index, time, frame
            sample_index += 1
    finally:
        cap.release()


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    """OpenCV 的 BGR 数组 -> PIL RGB 图像（imgutils 模型的输入类型）。"""
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def imwrite_unicode(path: str, image_bgr: np.ndarray) -> bool:
    """写图片，兼容非 ASCII 路径。

    cv2.imwrite 在 Windows 上把路径按本地 ANSI 代码页处理，遇到中文目录名会
    静默返回 False 而不抛异常——中文片名的视频会导致所有裁剪图丢失、
    所有轨迹的 character_id 变成 None、最终一个片段都选不出来。
    绕开方式是自己编码后用 numpy.tofile 写（走 Python 的宽字符文件 API）。
    """
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    buf.tofile(path)
    return True


def compute_hsv_hist(image_bgr):
    """计算归一化 HSV（H、S）直方图，把每帧压成一个颜色直方图，用于镜头切换比较。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# === 3. 过滤 ===

def crop_bbox(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """按框裁出子图（拷贝一份，使其不再持有整帧的内存）。

    框会先裁回画面边界内；空框/退化框返回 None。
    """
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1) # 把框裁回画面边界内
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: # 空框/退化框
        return None
    # copy() 是关键：numpy 切片是原帧的视图，留着它等于留着整帧不放，
    # 流式处理的内存优势会被这一个引用全部抵消。
    return image_bgr[y1:y2, x1:x2].copy()


def laplacian_variance(crop_bgr: Optional[np.ndarray]) -> float:
    """裁剪图的拉普拉斯方差（聚焦/模糊度量）。

    只取人脸框那块，因为我们只关心脸糊不糊，而不是整帧。空裁剪返回 0.0。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) # 格式转换。把彩色(3 通道 BGR)变成单通道灰度图。因为拉普拉斯算子是作用在单通道亮度上的,彩色三通道没必要分别算。
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


class FaceTracker:
    """在线 IoU 跟踪器：逐帧喂入检测结果，封存轨迹时一并交出代表裁剪图。

    相邻检测结果在标签相同且 IoU >= iou_threshold 时会加入同一条轨迹。
    最多允许丢失 track_gap_tolerance 帧。两帧之间如果发生镜头切换，
    即使 IoU 很高也禁止跨越切换连接（带丢帧容忍的重连也不能跨越切换）。

    之所以做成在线的：代表裁剪图要挑“该轨迹内 blur_var * confidence 最大”
    的那一张，而这在轨迹结束前无从判定。批处理版可以等全部跑完再回头读那一帧，
    流式下帧已经丢了。于是改为边跟踪边擂台——每条活跃轨迹只留当前最优的一张
    裁剪图，来了更好的就换掉，内存占用与视频长度无关。
    """

    def __init__(self, config: Config):
        self._config = config
        self._active: List[Dict] = []  # 当前"还活着"、可能继续延伸的轨迹
        self._finalized: List[Track] = []  # 已经封存、不再延伸的轨迹
        self._next_id = 1  # 轨迹 id 自增计数器

    def _open(self, index: int, det: Detection) -> None:
        """一张没能接上任何轨迹的脸，说明是新出现的，开一条新轨迹。"""
        track = {
            "id": self._next_id,
            "label": det.label,
            "last_index": index,
            "dets": [det],
            "blocked": False,
            "best_score": -1.0,
            "best_det": None,
            "best_crop": None,
        }
        self._next_id += 1
        self._active.append(track)
        self._offer(track, det)

    def _offer(self, track: Dict, det: Detection) -> None:
        """让 det 参与该轨迹的代表擂台；无论胜负都释放 det 持有的裁剪图。"""
        score = (det.blur_var or 0.0) * det.confidence
        if score > track["best_score"]:
            track["best_score"] = score
            track["best_det"] = det
            track["best_crop"] = det.crop
        det.crop = None  # 落选的裁剪图到此为止，不随 dets 列表一路累积

    def _finalize(self, track: Dict) -> None:
        dets = track["dets"]
        best = track["best_det"]
        self._finalized.append(
            Track(
                track_id=track["id"],
                label=track["label"],
                start_time=dets[0].time,
                end_time=dets[-1].time,
                detections=dets,
                representative_frame=best.frame_index if best else -1,
                representative_time=best.time if best else 0.0,
                representative_bbox=best.bbox if best else (0, 0, 0, 0),
                representative_image=track["best_crop"],
            )
        )

    def update(self, index: int, cut: bool, detections: List[Detection]) -> None:
        """处理第 index 帧。

        参数：
            index: 采样帧的从零开始索引。
            cut: 第 index-1 帧与第 index 帧之间是否发生镜头切换。
            detections: 该帧中通过质量过滤的检测结果。
        """
        # 切换发生在本帧之前，所有活跃轨迹（last_index 必 < index）都被它隔开。
        if cut:
            for tr in self._active:
                tr["blocked"] = True

        # 丢弃无法再恢复的轨迹：丢帧超过容忍值，或其最后检测到当前帧之间已有切换。
        still_active = []
        for tr in self._active:
            gap = index - tr["last_index"] - 1
            if gap > self._config.track_gap_tolerance or tr["blocked"]:
                self._finalize(tr)
            else:
                still_active.append(tr)
        self._active = still_active

        # 贪心 IoU 匹配：最佳配对优先，每条轨迹和每个检测结果只使用一次。
        pairs = []
        for ti, tr in enumerate(self._active):
            last_box = tr["dets"][-1].bbox
            for di, det in enumerate(detections):
                if det.label != tr["label"]:
                    continue
                score = iou(last_box, det.bbox) # 计算两个框 (x1,y1,x2,y2) 的交并比
                if score >= self._config.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            track = self._active[ti]
            track["dets"].append(detections[di])
            track["last_index"] = index
            self._offer(track, detections[di])

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self._open(index, det)

    def finish(self) -> List[Track]:
        """封存所有剩余轨迹，返回按开始时间排序的全部轨迹。"""
        for tr in self._active:
            self._finalize(tr)
        self._active = []
        self._finalized.sort(key=lambda t: t.start_time)
        return self._finalized


def track_faces(
    frame_detections: List[List[Detection]],
    is_cut: List[bool],
    config: Config,
) -> List[Track]:
    """把逐帧检测结果一次性喂给 :class:`FaceTracker`（批处理入口）。

    参数：
        frame_detections: 每帧中通过质量过滤的检测结果列表。
        is_cut: 每帧标记；is_cut[i] 表示第 i-1 帧和第 i 帧之间有切换。
        config: 流程配置。

    返回：
        所有轨迹，按开始时间排序。
    """
    tracker = FaceTracker(config)
    for i, dets in enumerate(frame_detections):
        tracker.update(i, is_cut[i], dets)
    return tracker.finish()


def save_representatives(tracks: List[Track], crops_dir: str) -> None:
    """把跟踪阶段在线留存的代表裁剪图写盘，并释放其内存。

    裁剪图在跟踪时就已选定（见 :class:`FaceTracker`），这里只负责落盘：
    写入 crops_dir，在轨迹上记下相对路径，然后丢掉数组引用。

    写入前先清空目录：换一组参数重跑往往产生更少的轨迹，留着上一轮的
    track_N.jpg 会让目录内容与 tracks.json 对不上。
    """
    if os.path.isdir(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir, exist_ok=True)
    for track in tracks:
        image = track.representative_image
        track.representative_image = None
        if image is None:
            continue
        crop_name = f"track_{track.track_id}.jpg"
        if imwrite_unicode(os.path.join(crops_dir, crop_name), image):
            track.representative_crop = os.path.join("crops", crop_name)


# === 5. 角色识别 ===

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


def compute_ccip_differences(
    tracks: List[Track],
    out_dir: str,
) -> Tuple[List[Track], Optional[np.ndarray]]:
    """对有代表裁剪图的轨迹批量提取 CCIP 特征，返回两两差异矩阵。

    这是整个角色识别里唯一昂贵的一步（要跑 ONNX 推理），且结果与合并阈值
    无关，因此单独拆出来——阈值扫描可以只算一次、复用到所有阈值上。

    返回：
        (候选轨迹, N×N 差异矩阵)；没有候选轨迹时矩阵为 None。
    """
    # 延迟导入：CCIP 模型较重且首次使用需从 HuggingFace 下载，
    from imgutils.metrics import ccip_batch_differences, ccip_batch_extract_features

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
        return [], None

    # 分批提取：一次性送入几百张图会让 ONNX 推理内存分配失败（bad allocation）。
    batch_size = 32
    feature_batches = [
        ccip_batch_extract_features(crop_paths[i:i + batch_size])
        for i in range(0, len(crop_paths), batch_size)
    ]
    features = np.concatenate(feature_batches)
    return candidates, ccip_batch_differences(features)


def resolve_ccip_threshold(config: Config) -> float:
    """取 config.ccip_threshold；为 None 时用 imgutils 的默认阈值（约 0.178）。"""
    if config.ccip_threshold is not None:
        return config.ccip_threshold
    from imgutils.metrics import ccip_default_threshold

    return ccip_default_threshold()


def assign_characters(tracks: List[Track], out_dir: str, config: Config) -> int:
    """用 CCIP 对轨迹代表裁剪图聚类，给每条轨迹写入 character_id。

    对候选轨迹的差异矩阵做 complete-linkage 层次聚类（簇内任意两张裁剪图
    差异都 < 阈值），同一簇视为同一角色。无裁剪图（或文件缺失）的轨迹保持
    character_id=None，不参与后续窗口内的角色计数。

    返回：
        识别出的不同角色总数。
    """
    candidates, diff_matrix = compute_ccip_differences(tracks, out_dir)
    if diff_matrix is None:
        return 0
    cluster_ids = _cluster_by_difference(diff_matrix, resolve_ccip_threshold(config))
    for track, cluster_id in zip(candidates, cluster_ids):
        track.character_id = cluster_id
    return len(set(cluster_ids))


# === 6. 选段 ===

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


# === 7. 截取 ===

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


def force_utf8_stdout() -> None:
    """把 stdout/stderr 切到 UTF-8。

    Windows 控制台默认按本地 ANSI 代码页（简中为 936/GBK）编码输出，
    中文片名和中文日志到了 UTF-8 终端就是一片乱码。放在 CLI 入口调用。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


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


def _annotate(image_bgr: np.ndarray, detections: List[Detection]) -> bytes:
    """在帧的副本上绘制框和分数，编码成 JPEG 字节返回。

    流式下无法预先知道总帧数，可视化样本要靠蓄水池采样从整段视频里均匀抽取；
    直接留整帧太占内存（1080p ≈ 6MB/张），因此当场压成 JPEG（≈200KB/张）
    再进池子。
    """
    canvas = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas, f"{det.confidence:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return cv2.imencode(".jpg", canvas)[1].tobytes()


# === 主流程 ===

def scan_video(
    config: Config,
    video_path: str,
    out_dir: str,
    detector: Detector,
    limit_seconds: Optional[float] = None,
    viz_count: int = 0,
) -> Tuple[List[Track], List[Dict], float, int]:
    """流式扫描一遍视频，产出轨迹和落盘的代表裁剪图。

    这是流水线里唯一需要读视频的部分，也是唯一昂贵的部分。解码、镜头切换判定、
    检测、质量过滤、跟踪全部在同一趟循环里完成，每帧用完即弃。停在轨迹这一层
    （不做角色聚类、不选段），因为后面的阶段只依赖轨迹和裁剪图，与阈值无关，
    可以在不重扫视频的前提下反复重跑——sweep.py 正是这么用的。

    参数：
        config: 流程配置。
        video_path: 源视频路径。
        out_dir: 该视频的输出目录；裁剪图写入 <out_dir>/crops/。
        detector: 已加载的检测器实例。
        limit_seconds: 如果设置，只处理该时间戳之前的帧（用于快速校准）。
        viz_count: 随机导出的标注示例帧数量。

    返回：
        (轨迹列表, 检测记录列表, 有效时长秒数, 采样帧数)。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    duration = probe_duration(config, video_path)
    if limit_seconds is not None:
        duration = min(duration, limit_seconds)

    tracker = FaceTracker(config)
    detection_records: List[Dict] = []
    prev_hist = None
    num_frames = 0
    # 蓄水池采样：流式下不知道总帧数，靠它从整段视频均匀抽 viz_count 张样本帧。
    viz_pool: List[bytes] = []
    viz_seen = 0

    print(f"[{stem}] streaming decode + detect + track...")
    for index, time, image in iter_frames(config, video_path, limit_seconds):
        num_frames += 1
        frame_h = image.shape[0]

        # 与上一采样帧比较得到镜头切换标记。
        hist = compute_hsv_hist(image) # 把每帧压成一个颜色直方图(一个 numpy 数组)
        if prev_hist is None:
            cut = False
        else:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            cut = corr < config.scene_cut_threshold
        prev_hist = hist # prev_hist 只保留最近一帧的直方图,下一轮被新的覆盖。所以任意时刻内存里最多只有 2 个直方图

        # 先检测，再应用三道质量门槛。
        raw = detector.detect(to_pil(image), index, time) # 阶段2拿到原始检测框
        kept = []
        for det in raw:
            det.crop = crop_bbox(image, det.bbox) # 当帧就地裁下来，帧一丢就没有第二次机会
            det.blur_var = laplacian_variance(det.crop) # 计算清晰度
            ok = passes_quality(det, frame_h, config)
            detection_records.append(_detection_record(det, ok))
            if ok:
                kept.append(det)
            else:
                det.crop = None # 没过质量门槛的框不会成为代表，立刻释放

        # 在线跟踪：轨迹在此延伸或封存，代表裁剪图也在此擂台决出。
        tracker.update(index, cut, kept)

        if viz_count > 0 and kept:
            viz_seen += 1
            if len(viz_pool) < viz_count:
                viz_pool.append(_annotate(image, kept))
            else:
                slot = random.randrange(viz_seen)
                if slot < viz_count:
                    viz_pool[slot] = _annotate(image, kept)
        # image 在此失去最后一个引用，整帧内存立即可回收。

    tracks = tracker.finish()
    print(f"[{stem}] {num_frames} frames sampled, {len(tracks)} tracks")

    save_representatives(tracks, os.path.join(out_dir, "crops"))

    if viz_pool:
        viz_dir = os.path.join(out_dir, "viz")
        if os.path.isdir(viz_dir):
            shutil.rmtree(viz_dir)
        os.makedirs(viz_dir, exist_ok=True)
        for i, jpeg in enumerate(viz_pool, start=1):
            with open(os.path.join(viz_dir, f"sample_{i:03d}.jpg"), "wb") as f:
                f.write(jpeg)

    return tracks, detection_records, duration, num_frames


def process_video(
    config: Config,
    video_path: str,
    output_root: str,
    detector: Optional[Detector] = None,
    limit_seconds: Optional[float] = None,
    viz_count: int = 0,
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

    返回：
        摘要字典（也会持久化到 JSON 文件中）。
    """
    if detector is None:
        detector = get_detector(config.detector, config)
    _report_providers(detector)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    tracks, detection_records, duration, num_frames = scan_video(
        config, video_path, out_dir, detector, limit_seconds, viz_count
    )

    # 角色识别：CCIP 聚类给轨迹分配 character_id。
    print(f"[{stem}] identifying characters...")
    num_characters = assign_characters(tracks, out_dir, config)
    print(f"[{stem}] {num_characters} distinct characters identified")

    # 片段选择。
    segments, num_qualified = select_segments(tracks, duration, config)
    print(
        f"[{stem}] {num_qualified} qualified windows, {len(segments)} segments selected"
    )

    # 截取片段。
    clips_dir = os.path.join(out_dir, "clips")
    clip_paths = clip_segments(config, video_path, segments, clips_dir)

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
            "num_characters": num_characters,
            "num_qualified_windows": num_qualified,
            "params": dataclasses.asdict(config),
            "segments": segments,
            "clips": [os.path.relpath(p, out_dir) for p in clip_paths],
        },
    )

    summary = {
        "video": video_path,
        "frames": num_frames,
        "tracks": len(tracks),
        "characters": num_characters,
        "qualified_windows": num_qualified,
        "segments": len(segments),
        "clips": len(clip_paths),
        "output_dir": out_dir,
    }
    print(f"[{stem}] done: {summary}")
    return summary


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
    parser.add_argument("--viz", type=int, default=0, help="Dump N annotated sample frames.") # 导出 N 张带标注的样本帧（蓄水池采样，均匀分布于全片），用于肉眼检查检测/过滤效果（默认 0，不导出）。
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口。"""
    force_utf8_stdout()
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
