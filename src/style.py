"""画风路由：判断一段视频是 2D 动画、3D CG 还是真人实拍。

为什么需要：三种画风需要**成套不同**的模型，不是同一套模型换个阈值。
动漫脸 YOLO 在真人和 3D CG 上明显弱于 SCRFD，SCRFD 在 2D 上又明显弱于它；
CCIP 按动漫角色图训练，换到真人脸上误差方向翻转。所以先判画风，再整套换
检测器 + 身份特征 + 阈值 + 过滤门槛（见 config.StyleProfile）。

怎么判，分两步（都在同一批抽样帧上做，一次解码）：

1. **2D vs 非 2D**：``imgutils.validate.anime_classify`` 多数投票。它把每帧分成
   bangumi / illustration / comic / 3d 等类别，3d 这一类同时收下真人与 3D CG。
   9 段素材上这个二分判别 9/9 正确。
2. **3D CG vs 真人**：比两个人脸检测器在这些帧上的证据强度
   （置信度之和的比值 anime / scrfd）。动机是这个比值直接量的就是
   「这些脸有多像动漫脸」，而这正是选检测器要回答的问题。实测 9 段素材上
   2D ≥1.11、3D CG 0.64~0.69、真人 0.32~0.38，三段之间有明显间隔。
   **注意这条规则只有 4 个非 2D 样本，是拟合出来的**，换素材要重新验。
   为什么不用 ``imgutils.validate.anime_real``：它把 凡人修仙传_1040s
   判成 real（6:2），3D CG 和真人分不开。

抽帧用 seek 而非顺序解码：这是主循环之外的一次性小开销（实测 <1s/段），
seek 被吸附到关键帧对画风判断没有影响。
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from config import Config

# anime_classify 的类别 → 非 2D 分支。未列出的类别按 2D 处理
# （bangumi / illustration / comic 都是平面画）。
_NON_2D_CLASSES = {"3d"}

STYLE_2D = "2d"
STYLE_3D = "3d"
STYLE_REAL = "real"


def _probe_frames(video_path: str, config: Config) -> List[np.ndarray]:
    """在整段视频上均匀抽 ``style_probe_frames`` 帧，返回 BGR 图像列表。

    打不开视频或一帧都读不到时直接抛错——静默默认成某个风格会让后面所有
    参数都用错，而且不会有任何迹象。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = max(1, config.style_probe_frames)
        frames = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"画风判别一帧都没读到：{video_path}")
    return frames


def _detector_evidence(images: List[Image.Image], name: str, config: Config) -> float:
    """某个检测器在这批帧上的证据强度：所有检测框置信度之和。

    用置信度之和而不是框数：框数只数"有没有"，置信度还带上了"像不像"，
    而这里要比的正是两个检测器谁更认得这些脸。
    """
    from detectors import get_detector

    detector = get_detector(name, config)
    return sum(
        det.confidence
        for i, image in enumerate(images)
        for det in detector.detect(image, i, 0.0)
    )


def classify_style(video_path: str, config: Config) -> Tuple[str, Dict[str, object]]:
    """判断视频画风，返回 ``(风格, 判据)``。

    参数：
        video_path: 源视频。
        config: 取 ``style_probe_frames`` 与 ``style_real_evidence_ratio``。

    返回：
        ``("2d" | "3d" | "real", {判据字段: 值})``；判据里带上 anime_classify
        的逐类得票和（非 2D 时）两个检测器的证据强度，便于事后核对路由是否合理。
    """
    from imgutils.validate import anime_classify

    frames = _probe_frames(video_path, config)
    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

    votes: Dict[str, int] = {}
    for image in images:
        label, _score = anime_classify(image)
        votes[label] = votes.get(label, 0) + 1

    non_2d = sum(c for label, c in votes.items() if label in _NON_2D_CLASSES)
    if non_2d * 2 <= sum(votes.values()):
        return STYLE_2D, {"votes": votes}

    # 非 2D：再分 3D CG 与真人。两者的 ArcFace 最优阈值差得远
    # （3D ≈0.85、真人 ≈1.05），共用一个会把非 2D 的 F1 从 ~0.97 压到 ~0.83。
    anime_evidence = _detector_evidence(images, "anime_face_imgutils", config)
    real_evidence = _detector_evidence(images, "real_face_scrfd", config)
    ratio = anime_evidence / real_evidence if real_evidence > 0 else float("inf")
    style = STYLE_REAL if ratio < config.style_real_evidence_ratio else STYLE_3D
    return style, {"votes": votes, "evidence_ratio": round(ratio, 2)}


def apply_style(config: Config, style: str) -> Config:
    """按风格换上整套识别参数（见 config.StyleProfile），返回新的 Config。

    换的是「检测器 + 身份特征 + 阈值 + 裁剪外扩 + 人脸大小门槛」一整套，
    不是单个阈值——这几项互相绑死，拆开单换会拼出无意义的组合。
    """
    import dataclasses

    profile = config.style_profiles.get(style)
    if profile is None:
        return config
    return dataclasses.replace(
        config,
        detector=profile.detector,
        embedder=profile.embedder,
        identity_threshold=profile.identity_threshold,
        crop_margin=profile.crop_margin,
        min_face_height_ratio=profile.min_face_height_ratio,
        frontal_weight=profile.frontal_weight,
    )
