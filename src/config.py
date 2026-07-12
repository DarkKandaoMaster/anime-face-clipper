"""动漫脸剪辑流程的集中配置。

所有可调参数都放在单个 :class:`Config` 数据类中，这样为不同画风校准时
只需要改一个地方（或从命令行覆盖字段）。本文件不执行 I/O。
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Config:
    """整个流程的可调参数。

    属性按流程阶段分组。标记为经验值的参数
    （``blur_var_threshold``、``scene_cut_threshold``）应结合实际素材校准；
    默认值有意设得偏保守，适合作为首次处理的起点。
    """

    # === 抽帧 / 镜头切换 ===
    # 采样帧之间的秒数（ffmpeg fps = 1 / frame_interval）。
    frame_interval: float = 0.3
    # 相邻采样帧的 HSV 直方图相关性低于该值时视为镜头切换。
    # 值越低，检测到的切换越少。应按画风校准。
    scene_cut_threshold: float = 0.6

    # === 检测 ===
    # 已注册的检测器名称（见 detectors.py）。
    detector: str = "anime_face_imgutils"
    # imgutils YOLOv8 模型选择：level 's'（准确）或 'n'（快速）。
    detector_level: str = "s"
    detector_version: str = "v1.4"
    # 置信度低于该值的检测结果会在检测器内部被丢弃。
    conf_threshold: float = 0.5

    # === 过滤（三重质量过滤）===
    # 人脸框高度占画面高度的最小比例（用于丢弃远处或过小的人脸）。
    min_face_height_ratio: float = 0.045
    # 人脸裁剪图的最小拉普拉斯方差（用于丢弃模糊或运动拖影的人脸）。
    # 经验值；先用偏保守的低值，检查输出后再提高。
    blur_var_threshold: float = 50.0

    # === 跟踪（IoU + 镜头切换断轨）===
    # 相邻帧 IoU 大于等于该值时，将检测结果连接为同一条轨迹。
    iou_threshold: float = 0.3
    # 轨迹关闭前允许连续丢失的帧数。
    track_gap_tolerance: int = 1

    # === 角色识别（CCIP）===
    # 两条轨迹代表裁剪图的 CCIP 差异低于该阈值时视为同一角色。
    # None = 使用 imgutils 的 ccip_default_threshold()（约 0.178）。
    # 调低更严格、更容易把同一角色拆成多个；调高更容易合并。
    ccip_threshold: Optional[float] = 0.05

    # === 选段（滑窗计数 + 贪心）===
    window_seconds: float = 15.0
    # 窗口内至少出现过这么多不同角色时，该窗口才合格。
    min_events_per_window: int = 13

    # === 截取 ===
    # 首选（GPU）编码器；失败时回退到 encoder_fallback。
    encoder: str = "h264_nvenc"
    encoder_fallback: str = "libx264"

    # === 外部工具 ===
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
