"""通向 backend/core/ 的桥。

Web 进程只 import core 里**不带重依赖**的两个模块：
  - config.py  参数的唯一来源（窗口长度、吸附位移、编码器名……）
  - segments.py 选段重放（后端拖 X 走的就是流水线自己那份实现）
import main.py 会把 cv2 / onnxruntime / imgutils 拖进来，几百 MB 常驻，
Web 进程用不上——那些只在 worker 子进程里 import。
"""

import os
import sys

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from config import Config            # noqa: E402
from segments import select_from_scan  # noqa: E402  (re-export)

__all__ = ["Config", "pipeline_config", "select_from_scan", "CORE"]


def pipeline_config() -> Config:
    """出厂配置。运行前零参数——唯一可调的 X 只在结果页出现，不在这里。"""
    return Config()
