"""动漫脸剪辑器的端到端流程。

从视频中找出并截取所有“合格”的 window_seconds（30）秒片段。当一个片段窗口
内出现过至少 min_events_per_window 个不同角色时，该片段视为合格。一条轨迹由
相邻帧中 IoU 重叠的人脸框串联而成（镜头切换会强制断开轨迹）；再用身份特征
对各轨迹的代表裁剪图聚类，得到轨迹级角色身份（character_id），同一角色的
多条轨迹在窗口内只计一次。

处理前先判画风（style.py），据此成套换上检测器、身份特征、阈值与过滤门槛
（config.StyleProfile）：2D 走动漫脸 YOLO + CCIP，3D CG 与真人走 SCRFD + ArcFace。

流式处理：帧由 cv2.VideoCapture 顺序解码，用完即弃，全程不落盘。整条流水线
在内存中同时持有的图像只有“当前帧 + 每条活跃轨迹一张人脸裁剪图”，与视频
长度无关。代表裁剪图必须在检测的当帧就地留下——流式下没有第二次机会回头
读取任意一帧。

从项目根目录运行：

    python src/main.py <video>              # 处理单个视频 -> output/<stem>/
    python src/main.py <video> --viz 8      # 同时导出带标注的示例帧

阶段（按下方分节注释组织）：
    解码抽帧 -> 检测 -> 过滤 -> 跟踪 -> 角色识别 -> 选段 -> 截取

阈值敏感性：本流水线的输出对 identity_threshold 与 min_events_per_window 高度
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
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from config import Config, set_frontal_weight
from detectors import (
    Detection,
    Detector,
    count_eyes,
    crop_bbox,
    expand_bbox,
    get_detector,
)
from embedders import get_embedder
from style import apply_style, classify_style

# 重新导出，方便调用方使用 from src.main import Detection。
# 该类定义在 detectors.py 中，以避免检测器模块出现循环依赖。
__all__ = [
    "Detection", "Track", "crop_bbox", "expand_bbox",
    "filter_short_tracks", "process_video", "run_pipeline", "main",
]


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
        representative_frontal: 该检测结果的正脸分（没算过则为 None）。
        representative_images: 跟踪时在线留存的 BGR 裁剪图，按清晰度降序，
            最多 config.crops_per_track 张；写盘后清空。
        representative_crops: 保存的裁剪图路径（相对于输出目录），与上面同序。
        representative_crop: 其中最清晰的那张（== representative_crops[0]），
            聚类不用它，只是给人看和排查用。
        character_id: 身份聚类得到的角色簇编号；None 表示身份未知
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
    representative_frontal: Optional[float] = None
    representative_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    representative_crops: List[str] = dataclasses.field(default_factory=list)
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


# === 3. 过滤 ===
# crop_bbox / expand_bbox 定义在 detectors.py（裁剪口径和检测器绑死，
# 真人检测器要用关键点对齐覆盖它），这里重新导出给测试和调用方。

def laplacian_variance(crop_bgr: Optional[np.ndarray]) -> float:
    """裁剪图的拉普拉斯方差（聚焦/模糊度量）。

    只取人脸框那块，因为我们只关心脸糊不糊，而不是整帧。空裁剪返回 0.0。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) # 格式转换。把彩色(3 通道 BGR)变成单通道灰度图。因为拉普拉斯算子是作用在单通道亮度上的,彩色三通道没必要分别算。
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) # 用拉普拉斯方差计算并返回清晰度。越清晰值越大


def passes_quality(detection: Detection, frame_height: int, config: Config) -> bool:
    """应用三道便宜的质量门槛：置信度、人脸大小、清晰度。

    这三道都是纯算术，不做推理。正脸判定（眼睛数）单独放在
    :func:`passes_frontal`，因为它要跑一次 ONNX，只值得对已经通过这里的脸算。
    """
    if detection.confidence < config.conf_threshold:
        return False
    face_height = detection.bbox[3] - detection.bbox[1]
    if face_height < config.min_face_height_ratio * frame_height:
        return False
    if (detection.blur_var or 0.0) < config.blur_var_threshold:
        return False
    return True


def passes_frontal(detection: Detection, config: Config) -> bool:
    """正脸门槛：裁剪图上检出的眼睛数达到 ``config.require_eyes``。

    ``require_eyes == 0`` 时直接放行（不跑推理、``num_eyes`` 保持 None）。
    """
    if config.require_eyes <= 0:
        return True
    if detection.num_eyes is None:
        detection.num_eyes = count_eyes(detection.crop, config)
    return detection.num_eyes >= config.require_eyes


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
    的那几张，而这在轨迹结束前无从判定。批处理版可以等全部跑完再回头读那些帧，
    流式下帧已经丢了。于是改为边跟踪边擂台——每条活跃轨迹只留当前最好的
    config.crops_per_track 张裁剪图，来了更好的就挤掉最差的，内存占用与视频
    长度无关。
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
            # 擂台榜：(score, crop, frame_index) 按 score 降序，最多 crops_per_track 条。
            "best": [],
            # 榜首那张脸的位置信息。单独存是因为退化框（crop 为 None）也该留下
            # 溯源记录，而它进不了擂台榜。
            "best_score": -1.0,
            "best_det": None,
        }
        self._next_id += 1
        self._active.append(track)
        self._offer(track, det)

    def _offer(self, track: Dict, det: Detection) -> None:
        """让 det 参与该轨迹的代表擂台；无论胜负都释放 det 持有的裁剪图。

        擂台分 = blur_var × confidence × (1 - w + w × frontal)。正脸权重 w 只改
        「这条轨迹送哪一张脸去聚类」，不决定轨迹的存亡——非正脸的身份特征最不
        可靠，但把它整条丢掉会连角色一起丢（README 第六之二节）。
        """
        score = (det.blur_var or 0.0) * det.confidence
        weight = self._config.frontal_weight
        if weight > 0 and det.frontal is not None:
            score *= 1.0 - weight + weight * det.frontal
        if score > track["best_score"]:
            track["best_score"] = score
            track["best_det"] = det
        board = track["best"]
        if det.crop is not None:
            # 时间去冗余：擂台前 K 名默认会挤在相邻几帧（同一张脸算 K 遍）。要求
            # 榜上各张至少隔开 crop_min_gap_frames 个采样帧，K>1 才真的是多视角。
            gap = self._config.crop_min_gap_frames
            near = next(
                (i for i, item in enumerate(board)
                 if abs(item[2] - det.frame_index) < gap), None
            ) if gap > 0 else None
            if near is not None:
                # 撞上同一段时间：只保留这一段里更好的那张，不占额外名额。
                if score > board[near][0]:
                    board[near] = (score, det.crop, det.frame_index)
                    board.sort(key=lambda item: item[0], reverse=True)
            elif len(board) < self._config.crops_per_track or score > board[-1][0]:
                board.append((score, det.crop, det.frame_index))
                board.sort(key=lambda item: item[0], reverse=True)
                del board[self._config.crops_per_track:]  # 挤出榜的裁剪图在此失去引用
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
                representative_frontal=best.frontal if best else None,
                representative_images=[crop for _score, crop, _index in track["best"]],
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


def filter_short_tracks(tracks: List[Track], min_seconds: float) -> List[Track]:
    """丢掉出镜时长短于 min_seconds 的轨迹。

    这条门槛直接对应**人工标注的口径**——只有"出镜 >1 秒"的脸才被数成一个主体，
    而流水线此前完全没有实现它。代价是一闪而过的脸会各自聚成一个身份，直接抬高
    窗内角色数：`魔女之旅` 整集 788 条轨迹里中位轨迹只有 1 个采样帧。

    时长按 ``end_time - start_time`` 算，所以 n 个采样帧的轨迹时长是
    ``(n - 1) × frame_interval``，单帧轨迹时长为 0。

    调用点在跟踪之后、写代表图之前：时长只有轨迹封存后才知道，而注定被丢的轨迹
    不值得再写一次盘。
    """
    return [t for t in tracks if t.end_time - t.start_time >= min_seconds]


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
        images = track.representative_images
        track.representative_images = []
        for rank, image in enumerate(images):
            crop_name = f"track_{track.track_id}_{rank}.jpg"
            if imwrite_unicode(os.path.join(crops_dir, crop_name), image):
                track.representative_crops.append(os.path.join("crops", crop_name))
        if track.representative_crops:
            track.representative_crop = track.representative_crops[0]


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


def compute_differences(
    tracks: List[Track],
    out_dir: str,
    config: Config,
) -> Tuple[List[Track], Optional[np.ndarray]]:
    """对有代表裁剪图的轨迹批量提取身份特征，返回两两差异矩阵。

    这是整个角色识别里唯一昂贵的一步（要跑 ONNX 推理），且结果与合并阈值
    无关，因此单独拆出来——阈值扫描可以只算一次、复用到所有阈值上。

    用哪个特征由 ``config.embedder`` 决定（画风路由已经换好，见 embedders.py）。

    一条轨迹可能有多张代表图（config.crops_per_track）。特征在**图**这一层
    提取，再把 K×K 的子块取中位数聚合成轨迹间的一个差异值：中位数而不是最小值，
    是因为最小值只要有一张脸偶然像另一个角色就会把两条轨迹粘起来，而
    complete-linkage 的簇内约束会让这个错误继续扩散。

    返回：
        (候选轨迹, N×N 轨迹级差异矩阵)；没有候选轨迹时矩阵为 None。
    """
    candidates: List[Track] = []
    crop_paths: List[str] = []
    owners: List[int] = []  # 每张裁剪图属于第几条候选轨迹
    for track in tracks:
        paths = [
            os.path.join(out_dir, rel) for rel in track.representative_crops
            if os.path.isfile(os.path.join(out_dir, rel))
        ]
        if not paths:
            continue
        owners.extend([len(candidates)] * len(paths))
        candidates.append(track)
        crop_paths.extend(paths)

    if not candidates:
        return [], None

    crop_diff = get_embedder(config.embedder).differences(crop_paths)
    if len(crop_paths) == len(candidates):  # 每条轨迹一张图，无需聚合
        return candidates, crop_diff

    owners = np.asarray(owners)
    groups = [np.flatnonzero(owners == i) for i in range(len(candidates))]
    track_diff = np.zeros((len(candidates), len(candidates)), dtype=float)
    for i, rows in enumerate(groups):
        for j, cols in enumerate(groups[:i]):
            value = float(np.median(crop_diff[np.ix_(rows, cols)]))
            track_diff[i, j] = track_diff[j, i] = value
    return candidates, track_diff


def resolve_identity_threshold(config: Config) -> float:
    """取 config.identity_threshold；为 None 时用该 embedder 自带的默认阈值。"""
    if config.identity_threshold is not None:
        return config.identity_threshold
    return get_embedder(config.embedder).default_threshold()


def assign_characters(tracks: List[Track], out_dir: str, config: Config) -> int:
    """对轨迹代表裁剪图聚类，给每条轨迹写入 character_id。

    对候选轨迹的差异矩阵做 complete-linkage 层次聚类（簇内任意两张裁剪图
    差异都 < 阈值），同一簇视为同一角色。无裁剪图（或文件缺失）的轨迹保持
    character_id=None，不参与后续窗口内的角色计数。

    返回：
        识别出的不同角色总数。
    """
    candidates, diff_matrix = compute_differences(tracks, out_dir, config)
    if diff_matrix is None:
        return 0
    cluster_ids = _cluster_by_difference(diff_matrix, resolve_identity_threshold(config))
    for track, cluster_id in zip(candidates, cluster_ids):
        track.character_id = cluster_id
    return len(set(cluster_ids))


# === 6. 选段 ===

def _characters_in_window(
    ordered: List[Track],
    starts: List[float],
    t: float,
    window: float,
) -> Tuple[List[int], List[Track]]:
    """窗口 [t, t+window) 内出现过的不同角色 id 与相交轨迹。

    主循环和吸附后的复核共用这一份逻辑——复核不是可选的：windows.json 会记录
    每个片段的 character_count 和 character_ids，吸附后不重算就会写出与实际
    片段不符的数字。
    """
    hi = bisect.bisect_left(starts, t + window) # 二分取前缀：下标 [0, hi) 的轨迹满足 start_time < t+window（窗口右端开区间）
    overlapping = [tr for tr in ordered[:hi] if tr.end_time >= t] # 再线性筛掉窗口开始前就已结束的轨迹，剩下的即与窗口相交
    character_ids = sorted(
        {tr.character_id for tr in overlapping if tr.character_id is not None}
    )
    return character_ids, overlapping


def _nearest_cut(t: float, cuts: List[float], max_shift: float) -> Optional[float]:
    """距 t 最近的切镜时刻；没有或超出 max_shift 时返回 None（max_shift=0 即关闭）。"""
    if not cuts or max_shift <= 0:
        return None
    i = bisect.bisect_left(cuts, t)
    candidates = cuts[max(0, i - 1):i + 1]
    best = min(candidates, key=lambda c: abs(c - t))
    return best if abs(best - t) <= max_shift else None


def select_segments(
    tracks: List[Track],
    duration: float,
    config: Config,
    cuts: Optional[List[float]] = None,
) -> Tuple[List[Dict], int]:
    """滑动窗口统计出现过的不同角色数，并贪心选择片段。

    候选窗口起点按 frame_interval（抽帧间隔）步进。窗口 [t, t+W) 内
    “出现过”的轨迹指时间区间与窗口相交的轨迹（start_time < t+W 且
    end_time >= t，包括窗口开始前就在画面中的角色）。这些轨迹中不同
    character_id 的数量达到 min_events_per_window 时窗口合格；
    character_id 为 None（身份未知）的轨迹不参与角色计数。遇到合格
    窗口时输出对应片段，下一个候选窗口跳到 >= t+W，从而保证片段不重叠。

    传入 cuts（scdet 给出的切镜时刻表）时，合格窗口的起点会吸附到
    clip_snap_max_shift 秒内最近的切镜点——否则片段起点只是任意的
    k×frame_interval，大概率切在镜头中间。吸附后在新位置重新统计角色数，
    仍达标才采用，否则保持原起点。

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
        character_ids, overlapping = _characters_in_window(ordered, starts, t, window)
        if len(character_ids) >= config.min_events_per_window:
            num_qualified += 1
            start = t
            snapped = _nearest_cut(t, cuts or [], config.clip_snap_max_shift)
            if snapped is not None and snapped + window <= duration + 1e-6:
                snapped_ids, snapped_overlapping = _characters_in_window(
                    ordered, starts, snapped, window
                )
                if len(snapped_ids) >= config.min_events_per_window:
                    start = snapped
                    character_ids, overlapping = snapped_ids, snapped_overlapping
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(start + window, 3),
                    "character_count": len(character_ids),
                    "character_ids": character_ids,
                    "track_ids": [tr.track_id for tr in overlapping],
                }
            )
            k = math.ceil((start + window) / step - 1e-9) # math.ceil((start + window) / step)取最小的满足条件的整数，后面的 - 1e-9 是为了浮点防抖，防浮点误差
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


def detect_cuts(
    config: Config,
    video_path: str,
    limit_seconds: Optional[float] = None,
) -> List[float]:
    """用 ffmpeg scdet 滤镜全帧率扫一趟，返回镜头切换时刻（秒，升序）。

    为什么不是直方图：旧实现比较相邻采样帧的 HSV 直方图相关性，在 9 段素材上
    召回/精确只有 51%/51%，且在 EVA 这类以亮度变化为主的片段上完全失效（0%）。
    根因是它无状态——只看相邻两帧，无法区分"变化大是因为切镜"和"变化大是因为
    整个镜头本来就在剧变"。scdet 的 score 取「帧间差异」与「帧间差异相对上一次的
    跳变量」的较小值，持续剧变被压制、孤立尖峰才上报，这是改通道或改阈值补不上的
    结构性差异。代价是多一趟全帧率解码（实测 6.4s/300s）。

    参数：
        config: 流程配置（用到 ffmpeg、scdet_threshold）。
        video_path: 源视频。
        limit_seconds: 若给定，只扫前 N 秒。

    返回：
        升序的切镜时刻列表。空列表是合法结果（确实存在整段无切镜的素材）。
    """
    cmd = [config.ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info"]
    if limit_seconds is not None:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    cmd += [
        "-i", video_path,
        "-an",
        "-vf", f"scdet=threshold={config.scdet_threshold}",
        "-f", "null", "-",
    ]
    # encoding/errors 显式指定：中文路径与中文日志在 Windows 默认 ANSI 解码下会炸。
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").splitlines()[-15:])
        raise RuntimeError(f"ffmpeg scdet 失败（{video_path}）：\n{tail}")
    # scdet 把切镜时刻写进帧 metadata，日志里形如 "lavfi.scd.time: 12.345"。
    return sorted(
        float(t) for t in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", result.stderr or "")
    )


def _cut_between(cuts: List[float], prev_time: float, time: float, tol: float) -> bool:
    """区间 (prev_time-tol, time+tol] 内是否存在切镜时刻。

    抽成纯函数是为了不依赖视频即可单测。容差存在的理由：scdet 报的是真实 PTS，
    而采样帧时间戳是 raw_index / fps；叠加溶解转场本身也没有单帧答案。
    往灵敏侧放是对的——误差不对称：漏一刀会把两个角色粘成一条轨迹、静默丢掉一个
    角色；多切一刀只是把轨迹断成两段，阶段 5 的 CCIP 聚类会还原成同一个角色。
    """
    lo = bisect.bisect_right(cuts, prev_time - tol)
    hi = bisect.bisect_right(cuts, time + tol)
    return hi > lo


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
        "num_eyes": det.num_eyes,
        "frontal": round(det.frontal, 3) if det.frontal is not None else None,
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
        "representative_frontal": track.representative_frontal,
        "representative_crops": track.representative_crops,
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
) -> Tuple[List[Track], List[Dict], float, int, List[float]]:
    """流式扫描一遍视频，产出轨迹和落盘的代表裁剪图。

    这是流水线里最昂贵的部分。解码、
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

    切镜表由 detect_cuts 在主循环之前单独跑一趟全帧率解码得到（scdet 需要看到
    每一帧，0.3s 抽帧下镜头内的正常演进已与真切镜混叠）。

    返回：
        (轨迹列表, 检测记录列表, 有效时长秒数, 采样帧数, 切镜时刻列表)。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    duration = probe_duration(config, video_path)
    if limit_seconds is not None:
        duration = min(duration, limit_seconds)

    print(f"[{stem}] scdet scan (threshold={config.scdet_threshold})...")
    cuts = detect_cuts(config, video_path, limit_seconds)
    print(f"[{stem}] {len(cuts)} cuts detected")

    tracker = FaceTracker(config)
    detection_records: List[Dict] = []
    # 正脸分要么进擂台加权、要么当硬门槛；两者都不用时连推理都不跑
    # （2D 支路的动漫眼检测是每张脸一次 ONNX，不是免费的）。
    need_frontal = config.frontal_weight > 0 or config.min_frontal_score > 0
    prev_time: Optional[float] = None
    num_frames = 0
    # 蓄水池采样：流式下不知道总帧数，靠它从整段视频均匀抽 viz_count 张样本帧。
    viz_pool: List[bytes] = []
    viz_seen = 0

    print(f"[{stem}] streaming decode + detect + track...")
    for index, time, image in iter_frames(config, video_path, limit_seconds):
        num_frames += 1
        frame_h = image.shape[0]

        # 上一采样帧与本帧之间夹着切镜时刻则断轨。首帧无前帧，固定为 False。
        cut = prev_time is not None and _cut_between(
            cuts, prev_time, time, config.cut_time_tolerance
        )
        prev_time = time

        # 先检测，再应用三道质量门槛。
        raw = detector.detect(to_pil(image), index, time) # 阶段2拿到原始检测框
        kept = []
        for det in raw:
            det.crop = crop_bbox(image, det.bbox) # 当帧就地裁下来，帧一丢就没有第二次机会
            det.blur_var = laplacian_variance(det.crop) # 计算清晰度（按紧贴人脸框算）
            # 先过三道便宜门槛，再对幸存者跑眼睛检测（正脸判定），避免给注定
            # 被刷掉的脸白跑一次推理。
            ok = passes_quality(det, frame_h, config)
            if ok and need_frontal:
                # 正脸分必须在**紧贴人脸框**的裁剪图上算：外扩后的图会把旁边
                # 那张脸的眼睛也框进来。算完才换成检测器口径的代表裁剪图。
                det.frontal = detector.frontal_score(det.crop, det)
                ok = det.frontal >= config.min_frontal_score
            if ok:
                # 幸存者才换成检测器指定口径的代表裁剪图（动漫脸外扩带上发型、
                # 真人脸按关键点对齐）；注定被刷掉的脸不值得多跑一次。
                det.crop = detector.make_crop(image, det)
            ok = ok and passes_frontal(det, config)
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
    if config.min_track_seconds > 0:
        kept_tracks = filter_short_tracks(tracks, config.min_track_seconds)
        print(f"[{stem}] {len(tracks) - len(kept_tracks)} tracks shorter than "
              f"{config.min_track_seconds}s dropped")
        tracks = kept_tracks
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

    return tracks, detection_records, duration, num_frames, cuts


def process_video(
    config: Config,
    video_path: str,
    output_root: str,
    limit_seconds: Optional[float] = None,
    viz_count: int = 0,
    clip: bool = True,
) -> Dict:
    """对单个视频运行完整流程。

    参数：
        config: 流程配置。
        video_path: 源视频路径。
        output_root: 基础输出目录；结果会写入 <root>/<stem>/。
        limit_seconds: 如果设置，只处理该时间戳之前的帧（用于快速校准）。
        viz_count: 随机导出的标注示例帧数量。
        clip: 是否真的把选中的片段编码出来。False 时只写 JSON——在长片上
            做批量测量时，编码几十个 30 秒片段的耗时会盖过流水线本身。

    返回：
        摘要字典（也会持久化到 JSON 文件中）。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    # 画风路由：先判 2D / 非 2D，再据此换上整套识别参数。必须在扫描之前做，
    # 因为路由后的 config 决定用哪个检测器、怎么裁剪，一路带到聚类阶段。
    style = "-"
    if config.style_routing:
        style, votes = classify_style(video_path, config)
        config = apply_style(config, style)
        print(f"[{stem}] style={style} {votes} -> {config.detector} + "
              f"{config.embedder}@{resolve_identity_threshold(config)}")

    # 检测器实例按名字在进程内复用（get_detector 内部缓存），批量处理时
    # 同画风的视频不会重复加载模型。
    detector = get_detector(config.detector, config)
    _report_providers(detector)

    tracks, detection_records, duration, num_frames, cuts = scan_video(
        config, video_path, out_dir, detector, limit_seconds, viz_count
    )

    # 角色识别：CCIP 聚类给轨迹分配 character_id。
    print(f"[{stem}] identifying characters...")
    num_characters = assign_characters(tracks, out_dir, config)
    print(f"[{stem}] {num_characters} distinct characters identified")

    # 片段选择。
    segments, num_qualified = select_segments(tracks, duration, config, cuts)
    print(
        f"[{stem}] {num_qualified} qualified windows, {len(segments)} segments selected"
    )

    # 截取片段。
    clips_dir = os.path.join(out_dir, "clips")
    clip_paths = clip_segments(config, video_path, segments, clips_dir) if clip else []

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
            "style": style,
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

    检测器与身份特征模型都在各自的注册表里按名字缓存，所以多个视频之间
    自动复用同一份已加载模型，即使它们被路由到不同画风。
    """
    return [
        process_video(config, video_path, output_root, **kwargs)
        for video_path in video_paths
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime face clipper.")
    # 位置参数：输入视频，可传多个。例：python src/main.py a.mp4 b.mp4
    parser.add_argument("videos", nargs="+", help="Input video path(s).")
    # 输出根目录，默认 output。
    parser.add_argument("--output-dir", default="output", help="Output base directory.")
    # 便于校准的覆盖参数：不填则用 config.py 中的默认值（见 main() 应用逻辑）。
    parser.add_argument("--conf", type=float, help="Override conf_threshold.") # 置信度阈值（默认 0.5）。调高更严格：误检少、漏检多。
    parser.add_argument("--blur-var", type=float, help="Override blur_var_threshold.") # 模糊过滤的拉普拉斯方差下限（默认 50.0）。调高丢弃更多模糊/拖影脸。
    parser.add_argument("--scdet-threshold", type=float, help="Override scdet_threshold (0-100); lower detects more cuts.") # ffmpeg scdet 阈值（默认 10.0）。调低检测到的切换更多；CG 动作素材建议 5。
    parser.add_argument("--min-events", type=int, help="Override min_events_per_window.") # 窗口内所需不同角色数（默认 4）。调低则更多片段合格、出片更多。
    parser.add_argument("--identity-threshold", type=float, help="Override identity_threshold.") # 身份合并阈值。调高更容易把不同轨迹合并为同一角色。跨 embedder 不可比（CCIP ≈0.178，ArcFace ≈0.6）。
    parser.add_argument("--crop-margin", type=float, help="Override crop_margin.") # 代表裁剪图相对人脸框的外扩比例。CCIP 靠它把发型带进特征；ArcFace 走关键点对齐，不受影响。
    parser.add_argument("--frame-interval", type=float, help="Override frame_interval.") # 抽帧间隔秒数（默认 0.3）。调小则采样更密、更慢更准。
    parser.add_argument("--encoder", help="Override video encoder (e.g. libx264).") # 视频编码器（默认 h264_nvenc）。失败会自动回退到 libx264；无 GPU 时显式传 libx264。
    parser.add_argument("--frontal-weight", type=float, metavar="W", help="Weight of the frontal score in representative-crop ranking (0 = off).") # 正脸分在代表图擂台里的权重。只换「这条轨迹送哪张脸去聚类」，不丢轨迹。
    parser.add_argument("--min-frontal", type=float, metavar="S", help="Drop faces whose frontal score is below S (0 = off).") # 正脸硬门槛
    parser.add_argument("--min-track-seconds", type=float, metavar="S", help="Drop tracks shorter than S seconds on screen (0 = off).") # 最短出镜时长，对应人工标注的"出镜 >1s"口径 # 正脸硬门槛：分数低于 S 的框直接丢。默认 0（关）——硬筛会连角色一起丢，见 README 第六之二节。
    parser.add_argument("--eyes", type=int, metavar="N", help="Require N eyes per face (legacy frontal filter). 0 = off (default).") # 正脸过滤：脸上至少检出 N 只眼才保留。默认 0（关）——实测有害，见 README 第六之二节；传 2 可复现那组对照。
    parser.add_argument("--no-style-routing", action="store_true", help="Disable style routing; use one detector/embedder/threshold for all.") # 关掉画风路由，所有素材共用 config 里的 detector / embedder / identity_threshold。
    # 运行 / 调试参数。
    parser.add_argument("--limit-seconds", type=float, help="Only process first N seconds.") # 只处理前 N 秒；调参时先跑短片段很有用。
    parser.add_argument("--viz", type=int, default=0, help="Dump N annotated sample frames.")
    parser.add_argument("--no-clip", action="store_true", help="Analyse only; skip encoding the selected clips.") # 只跑分析、不编码片段。批量实测长片时用：JSON 照写，省掉几十个 30 秒片段的编码时间。 # 导出 N 张带标注的样本帧（蓄水池采样，均匀分布于全片），用于肉眼检查检测/过滤效果（默认 0，不导出）。
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
    if args.scdet_threshold is not None:
        config.scdet_threshold = args.scdet_threshold
    if args.min_events is not None:
        config.min_events_per_window = args.min_events
    if args.identity_threshold is not None:
        config.identity_threshold = args.identity_threshold
    if args.crop_margin is not None:
        config.crop_margin = args.crop_margin
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    if args.encoder is not None:
        config.encoder = args.encoder
    if args.eyes is not None:
        config.require_eyes = args.eyes
    if args.frontal_weight is not None:
        set_frontal_weight(config, args.frontal_weight)
    if args.min_frontal is not None:
        config.min_frontal_score = args.min_frontal
    if args.min_track_seconds is not None:
        config.min_track_seconds = args.min_track_seconds
    if args.no_style_routing:
        config.style_routing = False

    run_pipeline(
        config,
        args.videos,
        args.output_dir,
        limit_seconds=args.limit_seconds,
        viz_count=args.viz,
        clip=not args.no_clip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
