"""动漫脸剪辑流程的集中配置。

所有可调参数都放在单个 :class:`Config` 数据类中，这样为不同画风校准时
只需要改一个地方（或从命令行覆盖字段）。本文件不执行 I/O。
"""

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class StyleProfile:
    """一种画风对应的整套识别参数。

    这四个参数是耦合的，必须成套换：真人素材要用真人脸检测器，而真人脸检测器
    产出的是对齐好的 112×112 人脸，只有 ArcFace 认；ArcFace 的差异尺度又和
    CCIP 完全不同，阈值不能共用。拆成四张独立的表就会拼出无意义的组合。

    属性：
        detector: detectors.py 里注册的检测器名。
        embedder: embedders.py 里注册的身份特征名。
        identity_threshold: 两条轨迹的代表图差异低于它就算同一个人。
        crop_margin: 代表裁剪图相对人脸框的外扩比例（ArcFace 走对齐，不看它）。
        min_face_height_ratio: 人脸框高度占画面高度的下限。这个也必须跟着
            检测器走：动漫脸 YOLO 的框连着大半个头，SCRFD 的框只到眉毛~下巴，
            同一张脸后者的框明显更矮，共用一个比例会让真人素材放进大量
            远景路人（实测 爱情神话_925s 因此从 48 条轨迹涨到 128 条）。
    """

    detector: str
    embedder: str
    identity_threshold: float
    crop_margin: float
    min_face_height_ratio: float


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
    # 代表裁剪图在人脸框基础上向外扩多少（框宽/高的倍数）。0 = 用紧贴的人脸框。
    # 动机：CCIP 按「角色图」训练，而检测器给的是紧贴五官的人脸框，发型/发色
    # ——动漫角色身份最强的线索——恰好被裁在框外。外扩把它带回来。
    # 只影响送进 CCIP 的裁剪图；质量过滤（大小、清晰度）仍按原始人脸框算。
    crop_margin: float = 0.0
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
    # 每条轨迹留几张代表裁剪图参与身份聚类。1 = 只留最清晰的那张。
    # >1 时轨迹间的差异取「两侧代表图两两差异的中位数」，单张脸偶然拍糊、
    # 侧过头、被遮挡就不再能单独决定这条轨迹的身份。内存代价是每条**活跃**
    # 轨迹多留 K-1 张小图，与视频长度无关。
    #
    # 实测结论是**默认 1**（见 README 第六之三节的 A/B）：K=3 在 9 段素材上
    # 与 K=1 打平偏差（2d 0.89 vs 0.91、3d 0.97 vs 1.00、real 持平）。原因是
    # 榜单按 blur_var×confidence 排序，前三名往往来自相邻几帧、内容高度冗余，
    # 多出来的两张没带来新视角，中位数反而抬高了同一角色之间的距离。
    # 要复现该对照把它改成 3。
    crops_per_track: int = 1

    # === 画风路由 ===
    # 关闭后所有素材共用下面的 detector / embedder / identity_threshold。
    style_routing: bool = True
    # 判画风时均匀抽多少帧投票。
    style_probe_frames: int = 8
    # 非 2D 素材再分「真人 / 3D CG」的门槛：抽样帧上
    # 动漫脸检测器证据 / 真人脸检测器证据 低于它就判真人。
    # 9 段素材实测 3D CG 落在 0.64~0.69、真人落在 0.32~0.38，取中间值。
    style_real_evidence_ratio: float = 0.5
    # 每种画风的整套识别参数（见 StyleProfile）。取值来自 evaluate.py 在 9 段
    # 素材上按 macro F1 挑出的分风格最优，换素材必须重新扫网格定值。
    style_profiles: Dict[str, StyleProfile] = dataclasses.field(
        default_factory=lambda: {
            # 2D：动漫脸 YOLO + CCIP，裁剪外扩 0.6 倍把发型带进特征。
            "2d": StyleProfile("anime_face_imgutils", "ccip", 0.20, 0.6, 0.03),
            # 3D CG / 真人：SCRFD + ArcFace，按关键点对齐（不外扩）。
            # 两者的最优 ArcFace 阈值相同（0.85），差别只在人脸尺寸门槛：
            # 真人实拍的街景里全是远景路人，而人工标注只数出镜主体，
            # 门槛不抬到 0.09 会把它们全放进来（爱情神话_925s 128 条轨迹 vs 真值 11 个角色）。
            "3d": StyleProfile("real_face_scrfd", "arcface", 0.85, 0.0, 0.045),
            "real": StyleProfile("real_face_scrfd", "arcface", 0.85, 0.0, 0.09),
        }
    )

    # === 角色识别 ===
    # 身份特征名（见 embedders.py）：ccip = 动漫角色，arcface = 真人脸。
    embedder: str = "ccip"
    # 两条轨迹代表裁剪图的差异低于该阈值时视为同一角色。
    # None = 使用该 embedder 自带的默认阈值。
    # 调低更严格、更容易把同一角色拆成多个；调高更容易合并。
    identity_threshold: Optional[float] = 0.178

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
