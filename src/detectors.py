"""可插拔检测器层。

本模块定义统一的检测数据契约（:class:`Detection`）、抽象
:class:`Detector` 基类、一个基于名称的小型注册表，以及一个由
``dghs-imgutils`` 动漫脸检测支持的具体实现。

要把系统扩展到新的检测目标（动物、物体等），只需在这里新增一个
:class:`Detector` 子类，并用 ``@register("<name>")`` 装饰。下游的
跟踪、计数、截取阶段只消费 :class:`Detection` 对象，不需要修改。
"""

import abc
import dataclasses
from typing import Dict, List, Optional, Tuple, Type

from config import Config


@dataclasses.dataclass
class Detection:
    """单帧上的单个检测框。

    这是检测器与流程其余部分之间的稳定契约。``blur_var`` 与 ``crop`` 都不由
    检测器设置；它们会在后续质量过滤阶段填充，供跟踪阶段在线挑选代表裁剪图。

    属性：
        frame_index: 采样帧的从零开始索引。
        time: 帧时间戳（秒）。
        bbox: 像素坐标中的 ``(x1, y1, x2, y2)`` 框。
        confidence: 检测器置信度，范围为 ``[0, 1]``。
        label: 检测类别（例如 ``"anime_face"``）。
        blur_var: 裁剪图的拉普拉斯方差；由过滤阶段填充。
        num_eyes: 裁剪图上检出的眼睛数，由过滤阶段填充（正脸判定）。None 表示没算过
            （关闭了正脸过滤，或这张脸在更便宜的门槛上就被刷掉了）。
        crop: 该框对应的 BGR 裁剪图，由过滤阶段填充。流式处理中整帧读完即弃，
            无法事后回查，因此裁剪图必须在当帧就地留下。跟踪器在把检测结果
            并入轨迹时会挑出最优的一张留存、其余立即置 None 释放，
            所以内存中最多只有"每条活跃轨迹一张脸"。不参与 JSON 序列化。
    """

    frame_index: int
    time: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    blur_var: Optional[float] = None
    num_eyes: Optional[int] = None
    crop: Optional["object"] = None


class Detector(abc.ABC):
    """所有检测器的抽象基类。"""

    def __init__(self, config: Config):
        self._config = config

    @abc.abstractmethod
    def detect(self, image, frame_index: int, time: float) -> List[Detection]:
        """在单帧上检测目标。

        参数：
            image: 已解码的帧，``PIL.Image.Image``（RGB）。流程是流式的，
                帧不落盘，因此这里传的是内存中的图像对象而非路径。
                ``PIL.Image.Image`` 是 imgutils 的 ``ImageTyping`` 成员，
                可直接送入其模型；NumPy 数组不是，需调用方先行转换。
            frame_index: 帧的从零开始索引。
            time: 帧时间戳（秒）。

        返回：
            该帧的 :class:`Detection` 列表。
        """

    def actual_providers(self) -> Optional[List[str]]:
        """尽力报告当前启用的 ONNX 执行 providers。

        如果后端未暴露该信息，则返回 ``None``。
        仅用于 GPU 就绪情况的验证步骤。
        """
        return None


# === 注册表 ===

_REGISTRY: Dict[str, Type[Detector]] = {}


def register(name: str):
    """类装饰器：以 ``name`` 注册检测器。"""

    def _decorator(cls: Type[Detector]) -> Type[Detector]:
        if name in _REGISTRY:
            raise ValueError(f"Detector {name!r} is already registered.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_detector(name: str, config: Config) -> Detector:
    """按名称实例化已注册的检测器。"""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector {name!r}. Registered: {sorted(_REGISTRY)!r}"
        )
    return _REGISTRY[name](config)


# === 正脸判定（眼睛计数）===

_detect_eyes = None


def count_eyes(crop_bgr, config: Config) -> int:
    """数一张人脸裁剪图上能检出几只眼睛。

    正脸过滤的判据：两只眼都在 = 正脸。侧脸只看得到一只，背身/低头一只都没有，
    而这两类脸的 CCIP 特征最不可靠，是跨镜头身份去重的主要噪声来源。

    流式约束：必须在检测的当帧、拿着裁剪图时就算完，结果挂回 Detection。
    帧一丢就没有第二次机会回头读像素。

    参数：
        crop_bgr: 人脸区域的 BGR 裁剪图（``crop_bbox`` 的产物）。
        config: 取 ``eye_conf_threshold``。

    返回：
        检出的眼睛数量；裁剪图为空时返回 0。
    """
    global _detect_eyes
    if _detect_eyes is None:
        # 延迟导入：模型较重，且只有开启正脸过滤时才需要。
        from imgutils.detect import detect_eyes

        _detect_eyes = detect_eyes
    if crop_bgr is None or crop_bgr.size == 0:
        return 0
    import cv2
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return len(_detect_eyes(image, conf_threshold=config.eye_conf_threshold))


# === imgutils 动漫脸检测器 ===

_IMGUTILS_REPO_ID = "deepghs/anime_face_detection"


@register("anime_face_imgutils")
class AnimeFaceImgutils(Detector):
    """由 ``imgutils.detect.detect_faces``（YOLOv8）支持的动漫脸检测器。

    底层模型在 imgutils 内部以进程级单例缓存，因此只会加载一次，
    并在每一帧、每个视频之间复用。
    """

    LABEL = "anime_face"

    def __init__(self, config: Config):
        super().__init__(config)
        # 延迟导入，使本模块在缺少较重的可选依赖时仍可被导入。
        from imgutils.detect import detect_faces

        self._detect_faces = detect_faces

    def detect(self, image, frame_index: int, time: float) -> List[Detection]:
        results = self._detect_faces(
            image,
            level=self._config.detector_level,
            version=self._config.detector_version,
            conf_threshold=self._config.conf_threshold,
        )
        return [
            Detection(
                frame_index=frame_index,
                time=time,
                bbox=tuple(int(v) for v in bbox),
                confidence=float(score),
                label=self.LABEL,
            )
            for bbox, _label, score in results
        ]

    def actual_providers(self) -> Optional[List[str]]:
        # 访问 imgutils 缓存的 ONNX session，读取真实 providers。
        # 这里依赖私有内部结构，因此用宽泛保护做尽力诊断。
        try:
            from imgutils.generic.yolo import _open_models_for_repo_id

            model = _open_models_for_repo_id(_IMGUTILS_REPO_ID)
            for cached in model._models.values():  # noqa: SLF001
                session = cached[0]
                return list(session.get_providers())
        except Exception:  # pragma: no cover - 仅用于诊断
            return None
        return None
