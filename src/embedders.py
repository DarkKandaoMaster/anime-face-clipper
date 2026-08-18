"""可插拔身份特征层：把代表裁剪图变成两两差异矩阵。

跨镜头身份去重是本项目最大的难点，而「用哪个特征」和「素材是什么画风」绑死：

- ``ccip``：动漫角色身份特征（deepghs/ccip_onnx），按动漫角色图训练。
- ``arcface``：真人脸识别特征（InsightFace buffalo_l / w600k_r50），
  只在脸已按 5 点关键点对齐到 112×112 时才成立（见 detectors.RealFaceScrfd）。

两者都只暴露一个方法 :meth:`Embedder.differences`，返回 N×N 的差异矩阵
（越小越像同一个人）。聚类阶段只消费这个矩阵，不关心它是谁算的。

差异矩阵与合并阈值无关，因此扫描阈值网格时只需要算一次——sweep.py 和
evaluate.py 都依赖这个切分。
"""

import abc
import os
from typing import Dict, List, Type

import cv2
import numpy as np

# ArcFace 权重与检测模型同仓。
_INSIGHTFACE_REPO_ID = "public-data/insightface"
_ARCFACE_FILE = "models/buffalo_l/w600k_r50.onnx"


class Embedder(abc.ABC):
    """身份特征的抽象基类。"""

    @abc.abstractmethod
    def differences(self, crop_paths: List[str]) -> np.ndarray:
        """对一批代表裁剪图返回 N×N 对称差异矩阵（对角线为 0）。"""

    @abc.abstractmethod
    def default_threshold(self) -> float:
        """该特征自带的「同一身份」判定阈值。"""


_REGISTRY: Dict[str, Type[Embedder]] = {}
_INSTANCES: Dict[str, Embedder] = {}


def register(name: str):
    def _decorator(cls: Type[Embedder]) -> Type[Embedder]:
        if name in _REGISTRY:
            raise ValueError(f"Embedder {name!r} is already registered.")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_embedder(name: str) -> Embedder:
    """按名称取实例（进程内复用，模型只加载一次）。"""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown embedder {name!r}. Registered: {sorted(_REGISTRY)!r}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def imread_unicode(path: str) -> np.ndarray:
    """读图片，兼容非 ASCII 路径。

    与 main.imwrite_unicode 同一个原因：cv2.imread 在 Windows 上按本地 ANSI
    代码页解析路径，中文目录名会**静默返回 None**。素材文件名全是中文，
    输出目录直接取自片名，这条必踩。
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"读不出裁剪图（文件损坏？）：{path}")
    return image


@register("ccip")
class CcipEmbedder(Embedder):
    """动漫角色身份特征（imgutils CCIP）。"""

    # 一次性送入几百张图会让 ONNX 推理内存分配失败（bad allocation）。
    # 机器内存紧张时（本机实测可用物理内存 <3GB 就会踩）连 32 都会失败，
    # 用 CCIP_BATCH_SIZE 环境变量临时调小，不必改代码。
    BATCH_SIZE = int(os.environ.get("CCIP_BATCH_SIZE", "32"))

    def differences(self, crop_paths: List[str]) -> np.ndarray:
        from imgutils.metrics import ccip_batch_differences, ccip_batch_extract_features

        features = np.concatenate([
            ccip_batch_extract_features(crop_paths[i:i + self.BATCH_SIZE])
            for i in range(0, len(crop_paths), self.BATCH_SIZE)
        ])
        return ccip_batch_differences(features)

    def default_threshold(self) -> float:
        from imgutils.metrics import ccip_default_threshold

        return ccip_default_threshold()


@register("arcface")
class ArcFaceEmbedder(Embedder):
    """真人脸识别特征（InsightFace w600k_r50）。

    输入必须是已对齐的 112×112 人脸（RealFaceScrfd.make_crop 的产物）。
    差异定义为 ``1 - 余弦相似度``，取值范围 [0, 2]，与 CCIP 的差异同向
    （越小越像），因此聚类阶段可以原样复用。
    """

    BATCH_SIZE = 32
    # 人脸识别的常规同人门槛是余弦相似度 ≈0.4（InsightFace 在 IJB-C 上的工作点），
    # 换算成这里的差异就是 0.6。真值网格上再校准，见 config.style_profiles。
    DEFAULT_THRESHOLD = 0.6

    def __init__(self):
        import onnxruntime
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_INSIGHTFACE_REPO_ID, _ARCFACE_FILE)
        # 该 onnx 的输出形状被静态写成 [1, 512]，喂一批就会每次刷一行
        # "Expected shape ... does not match actual shape" 的告警。结果本身是对的
        # （返回的就是 [N, 512]），这是模型导出时的元数据问题，不是这里的 bug。
        # 把 session 日志压到 Error，真出错仍然会说话。
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            path, options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def differences(self, crop_paths: List[str]) -> np.ndarray:
        features = []
        for i in range(0, len(crop_paths), self.BATCH_SIZE):
            images = [imread_unicode(p) for p in crop_paths[i:i + self.BATCH_SIZE]]
            blob = cv2.dnn.blobFromImages(
                images, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
            )
            features.append(self._session.run(None, {self._input_name: blob})[0])
        features = np.concatenate(features)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        return 1.0 - features @ features.T

    def default_threshold(self) -> float:
        return self.DEFAULT_THRESHOLD
