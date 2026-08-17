"""动漫脸剪辑流程的集中配置。

所有可调参数都放在单个 :class:`Config` 数据类中，这样为不同画风校准时
只需要改一个地方（或从命令行覆盖字段）。本文件不执行 I/O。
"""

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class Config:
    """整个流程的可调参数。

    属性按流程阶段分组。标记为经验值的参数
    （``blur_var_threshold``、``scdet_threshold``）应结合实际素材校准；
    默认值有意设得偏保守，适合作为首次处理的起点。
    """

    # === 抽帧 / 镜头切换 ===
    # 采样帧之间的秒数（ffmpeg fps = 1 / frame_interval）。
    frame_interval: float = 0.3
    # ffmpeg scdet 滤镜的阈值（0-100），越低检测到的切换越多。
    # 10 在手绘作画素材上验证过；CG 动作素材建议 5。
    scdet_threshold: float = 10.0
    # 切镜时刻与采样区间匹配的容差（秒）。scdet 报的是真实 PTS，
    # 而采样帧的时间戳是 raw_index / fps，两者有亚秒级偏差。
    cut_time_tolerance: float = 0.1
    # 片段起点吸附到最近切镜点的最大位移（秒）。设 0 关闭吸附。
    clip_snap_max_shift: float = 2.0

    # === 检测 ===
    # 已注册的检测器名称（见 detectors.py）。
    detector: str = "anime_face_imgutils"
    # imgutils YOLOv8 模型选择：level 's'（准确）或 'n'（快速）。
    detector_level: str = "s"
    detector_version: str = "v1.4"
    # 置信度低于该值的检测结果会在检测器内部被丢弃。
    conf_threshold: float = 0.5

    # === 过滤（质量 + 正脸）===
    # 人脸框高度占画面高度的最小比例（用于丢弃远处或过小的人脸）。
    min_face_height_ratio: float = 0.045
    # 人脸裁剪图的最小拉普拉斯方差（用于丢弃模糊或运动拖影的人脸）。
    # 经验值；先用偏保守的低值，检查输出后再提高。
    blur_var_threshold: float = 50.0
    # 正脸判定：人脸裁剪图上至少要检出这么多只眼睛才保留。0 = 关闭。
    # 动机本来是跨镜头身份去重——非正脸的 CCIP 特征最不可靠，宁可丢掉。
    #
    # 实测结论是**默认关掉**（见 README 第六节的 A/B）：imgutils 的 detect_eyes
    # 是动漫眼检测器，对真人和 3D CG 几乎不触发，开启后 爱情神话_1525s 的轨迹
    # 从 23 条掉到 0 条（召回直接归零）。而在它本该有效的 2D 素材上，把 CCIP
    # 阈值重新调到各自最优后，开与不开的 F1 是 0.82 vs 0.89——开着反而更差。
    # 代码保留，用 --no-eyes 的反面（改这个值）可以复现该对照。
    require_eyes: int = 0
    # 眼睛检测的置信度门槛（imgutils.detect.detect_eyes）。
    eye_conf_threshold: float = 0.3

    # === 跟踪（IoU + 镜头切换断轨）===
    # 相邻帧 IoU 大于等于该值时，将检测结果连接为同一条轨迹。
    iou_threshold: float = 0.3
    # 轨迹关闭前允许连续丢失的帧数。
    track_gap_tolerance: int = 1

    # === 画风路由 ===
    # 关闭后所有素材共用下面的 ccip_threshold。
    style_routing: bool = True
    # 判画风时均匀抽多少帧投票（imgutils.validate.anime_classify）。
    style_probe_frames: int = 8
    # 每种风格用的 CCIP 阈值。CCIP 按动漫角色图训练，2D 与非 2D 的误差方向
    # 相反（2D 高估角色数、真人/3D 低估），所以这是唯一必须按风格分岔的参数。
    # 取值来自 evaluate.py 在 9 段素材上按 macro F1 挑出的分风格最优
    # （2D 的最优阈值恰好落在 imgutils 自带的默认值 0.178 上，非 2D 明显更低）。
    style_ccip_threshold: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"2d": 0.178, "non_2d": 0.08}
    )

    # === 角色识别（CCIP）===
    # 两条轨迹代表裁剪图的 CCIP 差异低于该阈值时视为同一角色。
    # None = 使用 imgutils 的 ccip_default_threshold()（约 0.178）。
    # 调低更严格、更容易把同一角色拆成多个；调高更容易合并。
    ccip_threshold: Optional[float] = 0.05

    # === 选段（滑窗计数 + 贪心）===
    # 需求口径就是"时长 ≤30 秒、出镜主体达到 X 人的片段"，因此窗口固定 30 秒。
    window_seconds: float = 30.0
    # 窗口内至少出现过这么多不同角色时，该窗口才合格（需求里的 X）。
    # 9 段素材的真值角色数落在 4~11，取下界 4 作为默认值。
    min_events_per_window: int = 4

    # === 截取 ===
    # 首选（GPU）编码器；失败时回退到 encoder_fallback。
    encoder: str = "h264_nvenc"
    encoder_fallback: str = "libx264"

    # === 外部工具 ===
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
