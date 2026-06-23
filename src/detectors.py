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

    这是检测器与流程其余部分之间的稳定契约。``blur_var`` 不由检测器设置；
    它会在后续质量过滤阶段填充（裁剪图的拉普拉斯方差），并保存在检测结果上，
    以便轨迹选择最清晰的代表结果。

    属性：
        frame_index: 采样帧的从零开始索引。
        time: 帧时间戳（秒）。
        bbox: 像素坐标中的 ``(x1, y1, x2, y2)`` 框。
        confidence: 检测器置信度，范围为 ``[0, 1]``。
        label: 检测类别（例如 ``"anime_face"``）。
        blur_var: 裁剪图的拉普拉斯方差；由过滤阶段填充。
    """

    frame_index: int
    time: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    blur_var: Optional[float] = None


class Detector(abc.ABC):
    """所有检测器的抽象基类。"""

    def __init__(self, config: Config):
        self._config = config

    @abc.abstractmethod
    def detect(self, image_path: str, frame_index: int, time: float) -> List[Detection]:
        """在单帧上检测目标。

        参数：
            image_path: 帧图片路径。实现应从该路径读取
                （imgutils 的 ``load_image`` 会按 RGB 解码）；
                这里有意不支持直接传入原始 NumPy 数组。
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

    def detect(self, image_path: str, frame_index: int, time: float) -> List[Detection]:
        results = self._detect_faces(
            image_path,
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
