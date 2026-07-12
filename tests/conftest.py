"""pytest 配置：把 src/ 加入 sys.path。

main.py 使用平铺导入（from config import Config），因此测试运行时
必须让 src/ 目录本身位于 sys.path 中，而不是把它当作包导入。
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
