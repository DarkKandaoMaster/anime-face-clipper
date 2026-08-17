"""画风路由：判断一段视频是 2D 动画还是非 2D（真人 / 3D CG）。

为什么需要：CCIP 按动漫角色图训练，2D 与非 2D 素材在它的特征空间里表现相反——
同一个 ccip_threshold 下 2D 普遍高估角色数、真人与 3D 普遍低估（README 第六节）。
一个阈值对不上全部素材，只能先判画风再选参数。

怎么判：均匀抽若干帧，``imgutils.validate.anime_classify`` 多数投票。
它把每帧分成 bangumi / illustration / comic / 3d 等类别，其中 3d 这一类
同时收下了真人与 3D CG——**它区分不了真人和 3D CG**。这里不强求区分：
两者在 CCIP 上的误差方向一致（都低估），共用一套参数即可。9 段素材上
这个二分判别 9/9 正确（见 evaluate.py 的 style 列）。

抽帧用 seek 而非顺序解码：这是主循环之外的一次性小开销（实测 0.3s/段），
seek 被吸附到关键帧对画风判断没有影响。
"""

from typing import Dict, Tuple

import cv2
from PIL import Image

from config import Config

# anime_classify 的类别 → 路由分支。未列出的类别按 2D 处理
# （bangumi / illustration / comic 都是平面画）。
_NON_2D_CLASSES = {"3d"}

STYLE_2D = "2d"
STYLE_NON_2D = "non_2d"


def classify_style(video_path: str, config: Config) -> Tuple[str, Dict[str, int]]:
    """判断视频画风，返回 ``(风格, 逐类得票数)``。

    参数：
        video_path: 源视频。
        config: 取 ``style_probe_frames``。

    返回：
        ``("2d" | "non_2d", {原始类别: 票数})``。

    打不开视频或一帧都读不到时直接抛错——静默默认成某个风格会让后面所有
    参数都用错，而且不会有任何迹象。
    """
    from imgutils.validate import anime_classify

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = max(1, config.style_probe_frames)
        votes: Dict[str, int] = {}
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
            ok, frame = cap.read()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label, _score = anime_classify(image)
            votes[label] = votes.get(label, 0) + 1
    finally:
        cap.release()

    if not votes:
        raise RuntimeError(f"画风判别一帧都没读到：{video_path}")

    non_2d = sum(c for label, c in votes.items() if label in _NON_2D_CLASSES)
    total_votes = sum(votes.values())
    style = STYLE_NON_2D if non_2d * 2 > total_votes else STYLE_2D
    return style, votes


def apply_style(config: Config, style: str) -> Config:
    """按风格覆盖参数，返回新的 Config（不改原对象）。

    目前只路由 ``ccip_threshold``——它是唯一被实测证明按画风翻转误差方向的参数。
    要加更多按风格分岔的参数，在 config.py 的 ``style_ccip_threshold`` 旁边
    再加一张表，别把分支散进逻辑里。
    """
    import dataclasses

    threshold = config.style_ccip_threshold.get(style)
    if threshold is None:
        return config
    return dataclasses.replace(config, ccip_threshold=threshold)
