"""Central configuration for the anime face clipper pipeline.

All tunable parameters live in a single :class:`Config` dataclass so that
calibration for different art styles only requires editing one place (or
overriding fields from the CLI). Nothing here performs I/O.
"""

import dataclasses


@dataclasses.dataclass
class Config:
    """Tunable parameters for the whole pipeline.

    Attributes are grouped by pipeline stage. Values flagged as empirical
    (``blur_var_threshold``, ``scene_cut_threshold``) are expected to be
    calibrated against the actual footage; the defaults are deliberately
    conservative for a first pass.
    """

    # === 抽帧 / 镜头切换 ===
    # Seconds between sampled frames (ffmpeg fps = 1 / frame_interval).
    frame_interval: float = 0.3
    # HSV histogram correlation below this between adjacent sampled frames is
    # treated as a shot cut. Lower => fewer cuts detected. Calibrate per style.
    scene_cut_threshold: float = 0.6

    # === 检测 ===
    # Registered detector name (see detectors.py).
    detector: str = "anime_face_imgutils"
    # imgutils YOLOv8 model selection: level 's' (accurate) or 'n' (fast).
    detector_level: str = "s"
    detector_version: str = "v1.4"
    # Detections below this confidence are dropped at the detector itself.
    conf_threshold: float = 0.5

    # === 过滤（三重质量过滤）===
    # Minimum face-box height as a fraction of frame height (drops far/tiny faces).
    min_face_height_ratio: float = 0.045
    # Minimum Laplacian variance of the face crop (drops blurry/motion-smeared
    # faces). Empirical; start conservative (low) and raise after inspecting output.
    blur_var_threshold: float = 50.0

    # === 跟踪（IoU + 镜头切换断轨）===
    # Adjacent-frame IoU at/above this links detections into one track.
    iou_threshold: float = 0.3
    # Number of consecutive missed frames a track may survive before closing.
    track_gap_tolerance: int = 1

    # === 选段（滑窗计数 + 贪心）===
    window_seconds: float = 15.0
    # A window qualifies when it contains at least this many track starts.
    min_events_per_window: int = 13

    # === 截取 ===
    # Preferred (GPU) encoder; falls back to encoder_fallback on failure.
    encoder: str = "h264_nvenc"
    encoder_fallback: str = "libx264"

    # === 外部工具 ===
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
