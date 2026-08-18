"""正脸评分：给每个人脸框打一个 [0, 1] 的「有多正」分数。

## 为什么要它

跨镜头身份去重是本项目最大的难点，而它的输入是**每条轨迹一张代表裁剪图**。
侧脸、背身、闭眼的脸，CCIP / ArcFace 特征最不可靠——实测长片上同一个角色
被拆成 4~5 个簇，拆分的边界恰好是「暗 / 糊 / 侧脸」而不是身份。

## 怎么用它（这是与旧方案的关键区别）

旧的 ``config.require_eyes`` 把非正脸的**检测框直接丢掉**，实测有害：
某个角色可能全片只有侧脸出镜，丢掉它等于丢掉一个真值身份（README 第六之二节）。

这里改成**只影响代表图的挑选顺序**：轨迹一条不丢，只是在轨迹内部的擂台
（``FaceTracker._offer``）里，把 ``blur_var × confidence`` 乘上正脸权重，
让送进聚类的那一张尽量是正脸。检测器认不出正脸时分数一律偏低而不是 0/1 突变，
排序自然退化回原来的清晰度擂台，没有召回代价。

## 两条支路（判据和检测器绑死，所以实现挂在 Detector 上）

- **真人 / 3D CG**：SCRFD 已经顺带给了 5 点关键点，纯几何算偏航角，零推理成本。
- **2D 动画**：动漫脸 YOLO 不给关键点，退而用动漫眼检测器的**框位置**
  （不是旧方案的「数几只眼」）：两眼的水平间距相对脸宽是偏航角的直接代理。
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# --- 关键点支路（SCRFD 5 点）---
# 鼻尖/嘴心相对两眼中点的水平偏移（以两眼间距为单位）。正脸 ≈0，全侧脸 ≈0.5。
# 取 0.45 当满偏：超过它一律记 0 分。
_YAW_FULL = 0.45

# --- 眼框支路（动漫眼检测）---
# 正脸时两眼中心的水平间距约占脸宽的 40%；侧过头这个间距被投影压缩。
_EYE_SPAN_FULL = 0.40
# 两眼中点相对脸框中心的水平偏移，超过脸宽的这个比例记 0 分。
_EYE_OFFSET_FULL = 0.30
# 只看得到一只眼 = 明确的侧脸，但仍比「一只都看不到」（遮挡/背身/检测失败）强。
_ONE_EYE_SCORE = 0.30
_NO_EYE_SCORE = 0.05


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def frontal_from_landmarks(landmarks: Optional[Sequence[Sequence[float]]]) -> float:
    """由 5 点关键点（左眼、右眼、鼻、左嘴角、右嘴角）算正脸分。

    做法：把坐标投影到「两眼连线」这条轴上，量鼻尖与嘴心相对两眼中点的横向偏移，
    再除以两眼间距归一化。这样得到的量对图像内的旋转与缩放都不敏感，只反映偏航。
    正脸时鼻尖与嘴心都落在两眼中点上（偏移 ≈0），侧过头则一起向近侧滑。

    鼻和嘴取平均而不是只用鼻：鼻尖在动作幅度大的帧上抖得厉害，嘴心更稳。

    参数：
        landmarks: (5, 2) 的像素坐标；None 或点数不对时返回 1.0（当作「未知不惩罚」）。

    返回：
        [0, 1]，越大越正。
    """
    if landmarks is None:
        return 1.0
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (5, 2):
        return 1.0

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    axis = right_eye - left_eye
    span = float(np.linalg.norm(axis))
    if span < 1e-6:  # 两眼重合：全侧脸的退化情形
        return 0.0
    axis = axis / span

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    nose_offset = abs(float(np.dot(nose - eye_mid, axis))) / span
    mouth_offset = abs(float(np.dot(mouth_mid - eye_mid, axis))) / span
    return _clip01(1.0 - (nose_offset + mouth_offset) / 2.0 / _YAW_FULL)


def frontal_from_eyes(
    eyes: List[Tuple[Tuple[int, int, int, int], str, float]],
    crop_width: int,
) -> float:
    """由动漫眼检测器在人脸裁剪图上的**框位置**算正脸分。

    旧方案只数眼睛个数（``require_eyes``），信息全丢在阈值里；这里用位置：

    - 两眼水平间距 / 脸宽：正脸约 0.40，侧过头被投影压缩，是偏航角的直接代理。
    - 两眼中点相对脸框中心的偏移：整张脸转过去时两眼一起往一侧滑。

    参数：
        eyes: ``imgutils.detect.detect_eyes`` 的返回值 [(bbox, label, score), ...]。
        crop_width: 人脸裁剪图的宽度（像素），用于归一化。

    返回：
        [0, 1]，越大越正。检不出眼睛不给 0：闭眼、刘海遮挡、戴墨镜都会让检测器
        哑火，而这些脸未必是侧脸，给一个小的正值让它仍能在没有更好选择时当代表。
    """
    if crop_width <= 0:
        return _NO_EYE_SCORE
    if not eyes:
        return _NO_EYE_SCORE
    if len(eyes) == 1:
        return _ONE_EYE_SCORE

    top = sorted(eyes, key=lambda item: item[2], reverse=True)[:2]
    centers = sorted((box[0] + box[2]) / 2.0 for box, _label, _score in top)
    span = (centers[1] - centers[0]) / crop_width
    offset = abs((centers[0] + centers[1]) / 2.0 - crop_width / 2.0) / crop_width
    return _clip01(span / _EYE_SPAN_FULL) * _clip01(1.0 - offset / _EYE_OFFSET_FULL)
