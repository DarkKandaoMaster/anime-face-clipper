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

import cv2
import numpy as np

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
        num_eyes: 裁剪图上检出的眼睛数，由过滤阶段填充（旧的 require_eyes 硬门槛）。
            None 表示没算过（关闭了该门槛，或这张脸在更便宜的门槛上就被刷掉了）。
        frontal: 正脸分 [0, 1]（见 frontal.py），由过滤阶段填充。None 表示没算过
            （frontal_weight 与 min_frontal_score 都为 0）。跟踪阶段用它给
            代表图擂台加权——非正脸的身份特征最不可靠。
        landmarks: 5 点关键点（左眼、右眼、鼻、左嘴角、右嘴角）的 (5, 2) 数组，
            只有能给出关键点的检测器才填（真人 SCRFD 填，动漫脸 YOLO 不填）。
            用于把脸对齐到 ArcFace 的标准姿态——不对齐的话人脸识别特征会失真。
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
    frontal: Optional[float] = None
    landmarks: Optional["object"] = None
    crop: Optional["object"] = None


# === 裁剪几何 ===

def crop_bbox(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """按框裁出子图（拷贝一份，使其不再持有整帧的内存）。

    框会先裁回画面边界内；空框/退化框返回 None。
    """
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))  # 把框裁回画面边界内
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:  # 空框/退化框
        return None
    # copy() 是关键：numpy 切片是原帧的视图，留着它等于留着整帧不放，
    # 流式处理的内存优势会被这一个引用全部抵消。
    return image_bgr[y1:y2, x1:x2].copy()


def expand_bbox(bbox: Tuple[int, int, int, int], margin: float) -> Tuple[int, int, int, int]:
    """把框按自身宽高的 margin 倍向四周外扩（不裁边界，交给 crop_bbox）。"""
    x1, y1, x2, y2 = bbox
    dx = int(round((x2 - x1) * margin))
    dy = int(round((y2 - y1) * margin))
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


class Detector(abc.ABC):
    """所有检测器的抽象基类。"""

    def __init__(self, config: Config):
        self._config = config

    def make_crop(self, image_bgr: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """从当前帧裁出这条检测的代表图，供后续身份特征提取使用。

        放在检测器上而不是流水线里：裁剪口径和身份模型是绑死的——动漫脸要
        外扩带上发型给 CCIP，真人脸要按关键点仿射对齐给 ArcFace。默认实现是
        「人脸框外扩 crop_margin」，需要别的口径的检测器覆盖它。

        流式约束：必须在检测的当帧调用，帧一丢就没有第二次机会回头读像素。
        """
        return crop_bbox(image_bgr, expand_bbox(det.bbox, self._config.crop_margin))

    def frontal_score(self, crop_bgr: np.ndarray, det: Detection) -> float:
        """这张脸有多「正」，返回 [0, 1]（见 frontal.py 的动机与用法）。

        和 :meth:`make_crop` 一样挂在检测器上：正脸的判据和检测器绑死——真人
        SCRFD 顺带给了 5 点关键点可以纯几何算，动漫脸 YOLO 只有框，得另外
        跑一次动漫眼检测。默认实现返回 1.0（未知不惩罚），让没实现它的检测器
        的排序退化回原来的清晰度擂台。

        流式约束：必须在检测的当帧、拿着紧贴人脸框的裁剪图时算完。

        参数：
            crop_bgr: 紧贴人脸框的 BGR 裁剪图（未按 crop_margin 外扩——外扩后
                会把旁边那张脸的五官也框进来）。
            det: 对应的检测结果（关键点支路从这里取 landmarks）。
        """
        return 1.0

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
_INSTANCES: Dict[str, Detector] = {}


def register(name: str):
    """类装饰器：以 ``name`` 注册检测器。"""

    def _decorator(cls: Type[Detector]) -> Type[Detector]:
        if name in _REGISTRY:
            raise ValueError(f"Detector {name!r} is already registered.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_detector(name: str, config: Config) -> Detector:
    """按名称取检测器实例（进程内按名字缓存，模型只加载一次）。

    每次取用都把 ``config`` 重新挂上：画风路由会为每个视频生成一份新的
    Config，而底层 ONNX session 是进程级单例，没有理由跟着重建。
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown detector {name!r}. Registered: {sorted(_REGISTRY)!r}"
        )
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name](config)
    detector = _INSTANCES[name]
    detector._config = config
    return detector


# === 正脸判定（眼睛计数）===

_detect_eyes = None


def detect_eye_boxes(crop_bgr, config: Config):
    """在人脸裁剪图上跑动漫眼检测，返回 [(bbox, label, score), ...]。

    正脸判定的原料。位置比个数信息量大得多（见 frontal.frontal_from_eyes），
    因此这里交出原始框，由调用方决定是数个数还是算几何。

    流式约束：必须在检测的当帧、拿着裁剪图时就跑完，帧一丢就没有第二次机会。
    """
    global _detect_eyes
    if _detect_eyes is None:
        # 延迟导入：模型较重，且只有开启正脸判定时才需要。
        from imgutils.detect import detect_eyes

        _detect_eyes = detect_eyes
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return _detect_eyes(image, conf_threshold=config.eye_conf_threshold)


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
    return len(detect_eye_boxes(crop_bgr, config))


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

    def frontal_score(self, crop_bgr: np.ndarray, det: Detection) -> float:
        """动漫脸没有关键点，用动漫眼检测器的框位置算（见 frontal.frontal_from_eyes）。"""
        from frontal import frontal_from_eyes

        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        return frontal_from_eyes(detect_eye_boxes(crop_bgr, self._config), crop_bgr.shape[1])

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


# === 真人脸检测器（InsightFace SCRFD-10G）===

_INSIGHTFACE_REPO_ID = "public-data/insightface"
_SCRFD_FILE = "models/buffalo_l/det_10g.onnx"

# ArcFace 的标准 112×112 五点模板（左眼、右眼、鼻尖、左嘴角、右嘴角）。
# 识别模型就是在对齐到这个姿态的脸上训练的，不对齐直接送裁剪图特征会明显退化。
_ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小二乘相似变换（旋转+等比缩放+平移），返回 2×3 仿射矩阵。

    用 Umeyama 闭式解而不是 cv2.estimateAffinePartial2D：后者是 RANSAC/LMEDS
    的鲁棒估计，5 个点上结果不稳定且不可复现；这里的点数固定为 5、且都可信，
    闭式最小二乘既确定又更准。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    cov = dst_demean.T @ src_demean / src.shape[0]
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    u, s, vt = np.linalg.svd(cov)
    rotation = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / src_demean.var(axis=0).sum())
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def _distance2points(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """SCRFD 的距离编码解码：每个 anchor 中心 + 到各边/各关键点的偏移。"""
    points = distances.reshape(distances.shape[0], -1, 2).copy()
    points[:, :, 0] += centers[:, None, 0]
    points[:, :, 1] += centers[:, None, 1]
    return points


@register("real_face_scrfd")
class RealFaceScrfd(Detector):
    """真人脸检测器：InsightFace buffalo_l 的 SCRFD-10G（ONNX）。

    为什么另开一个检测器：动漫脸 YOLO 在真人素材上是离分布的，而真人素材
    （爱情神话）本来就是需求里的一类。SCRFD 顺带给出 5 点关键点，
    :meth:`make_crop` 用它把脸对齐成 ArcFace 的标准姿态——真人身份特征
    必须走这一步。

    模型从 HuggingFace ``public-data/insightface`` 拉取，进程内单例复用。
    """

    LABEL = "real_face"
    INPUT_SIZE = 640          # SCRFD-10G 的训练分辨率，按长边等比缩放后补零
    STRIDES = (8, 16, 32)     # FPN 三层
    NUM_ANCHORS = 2           # 每个位置 2 个 anchor
    NMS_IOU = 0.4

    _session = None           # 类级单例：一个进程只加载一次

    def __init__(self, config: Config):
        super().__init__(config)
        if RealFaceScrfd._session is None:
            import onnxruntime
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(_INSIGHTFACE_REPO_ID, _SCRFD_FILE)
            RealFaceScrfd._session = onnxruntime.InferenceSession(
                path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        self._input_name = RealFaceScrfd._session.get_inputs()[0].name

    def detect(self, image, frame_index: int, time: float) -> List[Detection]:
        frame = np.asarray(image)[:, :, ::-1]  # PIL RGB -> BGR（blobFromImage 会再换回来）
        height, width = frame.shape[:2]
        scale = self.INPUT_SIZE / max(height, width)
        resized = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))))
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (self.INPUT_SIZE, self.INPUT_SIZE),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        outputs = RealFaceScrfd._session.run(None, {self._input_name: blob})

        # 输出顺序：3 层的分数、3 层的框偏移、3 层的关键点偏移。
        boxes, scores, keypoints = [], [], []
        for level, stride in enumerate(self.STRIDES):
            level_scores = outputs[level].reshape(-1)
            keep = level_scores >= self._config.conf_threshold
            if not keep.any():
                continue
            size = self.INPUT_SIZE // stride
            centers = np.stack(np.mgrid[:size, :size][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)[keep]
            box = _distance2points(centers, outputs[level + 3].reshape(-1, 4)[keep] * stride)
            kps = _distance2points(centers, outputs[level + 6].reshape(-1, 10)[keep] * stride)
            # 框的编码是「中心到左上、到右下」两组距离，左上那组要取负。
            box[:, 0] = 2 * centers - box[:, 0]
            boxes.append(box.reshape(-1, 4))
            scores.append(level_scores[keep])
            keypoints.append(kps)

        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        scores = np.concatenate(scores)
        keypoints = np.concatenate(keypoints) / scale

        xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        kept = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(), self._config.conf_threshold, self.NMS_IOU
        )
        return [
            Detection(
                frame_index=frame_index,
                time=time,
                bbox=tuple(int(v) for v in boxes[i]),
                confidence=float(scores[i]),
                label=self.LABEL,
                landmarks=keypoints[i],
            )
            for i in np.array(kept).reshape(-1)
        ]

    def frontal_score(self, crop_bgr: np.ndarray, det: Detection) -> float:
        """5 点关键点纯几何算偏航角，零推理成本（见 frontal.frontal_from_landmarks）。"""
        from frontal import frontal_from_landmarks

        return frontal_from_landmarks(det.landmarks)

    def make_crop(self, image_bgr: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """按 5 点关键点把脸仿射对齐到 ArcFace 的 112×112 标准姿态。"""
        matrix = _umeyama(det.landmarks, _ARCFACE_TEMPLATE)
        return cv2.warpAffine(image_bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)

    def actual_providers(self) -> Optional[List[str]]:
        return list(RealFaceScrfd._session.get_providers())
