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
        frontal_weight: 正脸分在代表图擂台里的权重（见 Config.frontal_weight）。
            也是按画风走的：真人/3D 走关键点几何（零成本、连续可信），
            2D 走动漫眼检测（要多跑一次推理，且闭眼/遮挡时哑火），两条支路的
            分数可信度不同，权重没有理由取同一个值。
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
    frontal_weight: float = 0.0


@dataclasses.dataclass
class Config:
    """整个流程的可调参数。

    属性按流程阶段分组。标记为经验值的参数
    （``blur_var_threshold``、``scdet_threshold``）应结合实际素材校准；
    默认值有意设得偏保守，适合作为首次处理的起点。
    """

    # === 并行 ===
    # 解码预取的队列深度：0 = 完全串行（原行为），>0 = 解码在后台线程里跑，
    # 主线程同时做检测推理。
    # 动机：iter_frames 是纯 CPU（H.264 解码 + 色彩转换），detector.detect 是纯
    # GPU，串在一个线程里两者只能交替执行、各自等对方。挪开之后两块硬件真正
    # 重叠。解码在 OpenCV 的 C 代码里放开了 GIL，所以线程就够，不需要进程。
    # 代价是内存里最多同时有 prefetch_frames + 1 帧而不是 1 帧——仍与视频长度
    # 无关，但不能设大（1080p 一帧 6MB）。
    prefetch_frames: int = 4

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
    # 正脸分（frontal.py，[0,1]）在代表图擂台里的权重：
    # 擂台分 = blur_var × confidence × (1 - w + w × frontal)。
    # 0 = 完全关闭（连正脸推理都不跑，退化回纯清晰度擂台）；1 = 完全由正脸分缩放。
    #
    # 这是正脸判定的**主用法**，和旧的 require_eyes 有本质区别：它一条轨迹都不丢，
    # 只改「这条轨迹送哪一张脸去聚类」。README 第六之二节里 require_eyes 的亏损
    # 来自整条轨迹被筛掉（某个角色可能全片只有侧脸出镜），加权不会有这个代价。
    frontal_weight: float = 0.0
    # 正脸硬门槛：分数低于它的检测框直接丢掉。0 = 关闭。
    # 默认关，理由同上；留着是为了能扫出「加权 vs 硬筛」的对照曲线。
    min_frontal_score: float = 0.0

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

    # 一条轨迹的最短出镜时长（秒），短于它的轨迹在跟踪结束后整条丢弃。0 = 关闭。
    # 这条门槛直接对应**人工标注的口径**（"出镜 >1 秒"才算一个主体），而流水线
    # 此前完全没有实现它：实测 `魔女之旅` 整集 788 条轨迹里，**中位轨迹只有 1 个
    # 采样帧**、54% 短于两帧——一闪而过的脸各自成为一个"角色"，是片长尺度上
    # 角色数虚高的主要来源之一。
    # 时长按 end_time - start_time 算，所以 n 个采样帧的轨迹时长是 (n-1)×frame_interval。
    min_track_seconds: float = 0.0

    # 多张代表图之间至少相隔这么多个采样帧。0 = 不作要求（原行为）。
    # 动机：擂台按 blur_var × confidence 取前 K 名，而这 K 名往往来自**相邻几帧**
    # ——内容高度冗余，等于把同一张脸算了 K 遍。中位数聚合因此拿不到新视角，
    # 实测 crops_per_track=3 在完整片源上反而更差（魔女之旅 54 → 60 个角色）。
    # 要求时间上分开，前 K 名才是真正的多视角（不同光照、不同角度）。
    # 只在 crops_per_track > 1 时有意义。
    crop_min_gap_frames: int = 0

    # === 画风路由 ===
    # 关闭后所有素材共用下面的 detector / embedder / identity_threshold。
    style_routing: bool = True
    # 人工指定画风（"2d" / "3d" / "real"），非空时直接套对应 profile，不跑判别。
    # 存在的理由不是"方便"，而是自动判别有一步**已证明分不开**：3D CG 与真人
    # 在片长尺度上证据比重叠（见 style_ambiguous_band），而这一步选错就整段
    # 用错 min_face_height_ratio。人比模型更清楚手上的片子是实拍还是 CG。
    force_style: Optional[str] = None
    # 判画风时均匀抽多少帧投票。16 而不是 8：2D / 非 2D 这一步的间隔很宽
    # （实测 2D 的非 3d 得票率 ≥88%、非 2D ≤31%），多抽几帧只为压掉投票噪声，
    # 代价是每段视频多零点几秒。3D CG / 真人那一步加多少帧都救不了，见下。
    style_probe_frames: int = 16
    # 非 2D 素材再分「真人 / 3D CG」的门槛：抽样帧上
    # 动漫脸检测器证据 / 真人脸检测器证据 低于它就判真人。
    # 30 秒素材实测 3D CG 落在 0.64~0.69、真人落在 0.32~0.38，取中间值。
    style_real_evidence_ratio: float = 0.5
    # 上面那个门槛**在片长尺度上不成立**的区间：落进来就打一条警告，提示手动
    # 传 --style。取值来自 26 段素材（9 段切片 + 17 段完整片源）的实测：
    # 3D CG 的证据比是 0.65 / 0.67 / 0.74，真人是 0.33 / 0.38 / **0.65**——
    # 整片的 `爱情神话`（真人）与 `凡人修仙传`（3D CG）撞在同一个值上，
    # 门槛取多少都分不开（README 第六之五节问题三）。
    #
    # 已实测**无效**的四条替代判据，不要重试：
    #   1. imgutils 的 anime_real：整帧上 3d 31~84% / real 66~97%，重叠；
    #      换成 SCRFD 人脸裁剪图更糟（3d 79~97% / real 81~89%，全判 real）。
    #   2. anime_classify 打在人脸裁剪图上：3d 50~76% / real 0~55%，重叠。
    #   3. 把证据比改成「逐帧比值的中位数」（只算检出到脸的帧，直击分母塌陷
    #      这个机制）：3d 0.395~0.466 / real 0.234~0.420，反而更糟。
    #   4. 加抽样帧（8 → 32）：间隔没有变宽，因为错的是判据不是样本量。
    # 要真正解决它，需要的是更多非 2D 片源（现在只有 2 部），不是更多调参。
    style_ambiguous_band: tuple = (0.45, 0.80)
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

    # 聚类范围："video" = 全片一次聚类（原行为），"window" = 只在每个 30 秒
    # 候选窗口内部对相交的轨迹重新聚类。
    # 动机：需求只问"这 30 秒里有几个不同角色"，从来不需要全片一致的身份编号。
    # 而 complete-linkage 对离群点零容忍——轨迹越多越容易撞上一张糊脸/侧脸，
    # 把本该合并的两簇卡住，所以过拆的严重程度随片长单调恶化（README 第六之五节：
    # `魔女之旅` 整集 788 条轨迹 → 54 个角色，白发主角一人占 5 个簇）。
    # 把聚类范围压回窗口，参与聚类的轨迹数就与片长无关，不再随片长恶化。
    # 代价：不同窗口的簇编号互不相通，`character_id` 仍是全片口径（给 montage
    # 和排查用），窗口内的角色数另算——两者在 30 秒素材上恒等。
    # 差异矩阵仍然只算一次（全片），窗口聚类只是取子矩阵重跑 linkage，几乎免费。
    #
    # 默认 "window"：实测在 9 个**片长尺度**真值窗口上 macro F1 0.79 → 0.90
    # （clip30 上限 0.96），9 个里 8 个变好、1 个略差；平均 ratio 1.57 → 0.99，
    # 也就是过拆基本消失。在 30 秒素材上两者**恒等**（整段就是一个窗口），
    # 所以 evaluate.py 的出厂基线一个数字都不变。
    # 用 --cluster-scope video 可以切回全片口径复现对照（见 evaluate_long.py）。
    cluster_scope: str = "window"

    # 一个角色在计数范围内的最短**累计**出镜时长（秒），短于它的角色不计入
    # 窗内角色数。0 = 关闭。
    #
    # 这条门槛就是人工标注自己的口径——标注只数「出镜 >1s」的脸。流水线此前
    # 唯一实现过的近似是 min_track_seconds（按**轨迹**算），它在 30 秒真值集上
    # 单调变差，原因是位置放错了：轨迹被切镜强制打断，一个角色演满 3 秒中间切
    # 两刀就是 3 条各 1 秒的轨迹，按轨迹卡会把这个真值身份整个抹掉。累计出镜
    # 时长是**聚类之后**才知道的量，所以只能在计数这一步算（main.count_characters）。
    #
    # 单条轨迹的时长按「采样检测数 × frame_interval」算而不是 end-start：
    # 长片上超过一半的轨迹只有一个采样帧，按 end-start 会算成 0 秒。
    min_character_seconds: float = 0.0

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


def set_frontal_weight(config: Config, weight: float) -> None:
    """把正脸权重同时压到全局字段和每个画风 profile 上（就地修改）。

    单独一个函数是因为画风路由会用 profile 覆盖全局字段（见 style.apply_style），
    只改 Config.frontal_weight 的话，路由一开就被 profile 的值盖掉，命令行覆盖
    会静默失效。做 A/B 时必须两边一起改。
    """
    config.frontal_weight = weight
    config.style_profiles = {
        name: dataclasses.replace(profile, frontal_weight=weight)
        for name, profile in config.style_profiles.items()
    }
