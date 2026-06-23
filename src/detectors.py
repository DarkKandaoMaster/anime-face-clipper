"""Pluggable detector layer.

This module defines the unified detection data contract (:class:`Detection`),
an abstract :class:`Detector` base class, a tiny name-based registry, and one
concrete implementation backed by ``dghs-imgutils`` anime face detection.

Extending the system to a new detection target (animals, objects, ...) only
means adding another :class:`Detector` subclass here and decorating it with
``@register("<name>")``. The downstream tracking / counting / clipping stages
consume :class:`Detection` objects and never need to change.
"""

import abc
import dataclasses
from typing import Dict, List, Optional, Tuple, Type

from config import Config


@dataclasses.dataclass
class Detection:
    """A single detected box on a single frame.

    This is the stable contract between detectors and the rest of the
    pipeline. ``blur_var`` is *not* set by detectors; it is populated later by
    the quality-filter stage (Laplacian variance of the crop) and carried on
    the detection so a track can pick its sharpest representative.

    Attributes:
        frame_index: Zero-based index of the sampled frame.
        time: Timestamp of the frame in seconds.
        bbox: Box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Detector confidence in ``[0, 1]``.
        label: Detection category (e.g. ``"anime_face"``).
        blur_var: Laplacian variance of the crop; filled by the filter stage.
    """

    frame_index: int
    time: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    blur_var: Optional[float] = None


class Detector(abc.ABC):
    """Abstract base class for all detectors."""

    def __init__(self, config: Config):
        self._config = config

    @abc.abstractmethod
    def detect(self, image_path: str, frame_index: int, time: float) -> List[Detection]:
        """Detect targets on one frame.

        Args:
            image_path: Path to the frame image. Implementations should read
                from the path (imgutils' ``load_image`` decodes it as RGB);
                passing a raw NumPy array is intentionally unsupported.
            frame_index: Zero-based index of the frame.
            time: Timestamp of the frame in seconds.

        Returns:
            A list of :class:`Detection` for this frame.
        """

    def actual_providers(self) -> Optional[List[str]]:
        """Best-effort report of the active ONNX execution providers.

        Returns ``None`` if the backend does not expose this information.
        Used only for the GPU-readiness verification step.
        """
        return None


# === 注册表 ===

_REGISTRY: Dict[str, Type[Detector]] = {}


def register(name: str):
    """Class decorator that registers a detector under ``name``."""

    def _decorator(cls: Type[Detector]) -> Type[Detector]:
        if name in _REGISTRY:
            raise ValueError(f"Detector {name!r} is already registered.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_detector(name: str, config: Config) -> Detector:
    """Instantiate a registered detector by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector {name!r}. Registered: {sorted(_REGISTRY)!r}"
        )
    return _REGISTRY[name](config)


# === imgutils 动漫脸检测器 ===

_IMGUTILS_REPO_ID = "deepghs/anime_face_detection"


@register("anime_face_imgutils")
class AnimeFaceImgutils(Detector):
    """Anime face detector backed by ``imgutils.detect.detect_faces`` (YOLOv8).

    The underlying model is cached as a process-wide singleton inside imgutils,
    so it is loaded once and reused across every frame and every video.
    """

    LABEL = "anime_face"

    def __init__(self, config: Config):
        super().__init__(config)
        # Imported lazily so this module can be imported without the heavy
        # optional dependency present.
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
        # Reach into imgutils' cached ONNX session to read the real providers.
        # Private internals, hence best-effort with a broad guard.
        try:
            from imgutils.generic.yolo import _open_models_for_repo_id

            model = _open_models_for_repo_id(_IMGUTILS_REPO_ID)
            for cached in model._models.values():  # noqa: SLF001
                session = cached[0]
                return list(session.get_providers())
        except Exception:  # pragma: no cover - diagnostic only
            return None
        return None
