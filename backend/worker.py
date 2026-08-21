"""在**子进程**里跑一段视频的分析。

长任务绝对不能待在请求处理函数里：uvicorn 是单线程异步的，几分钟的 GPU 活会
把事件循环卡死，期间连查进度都查不了。所以整条流水线扔进 ProcessPoolExecutor，
`max_workers = GPU_COUNT`，每个进程用 CUDA_VISIBLE_DEVICES 绑一张卡。

本模块会被 spawn 出来的子进程重新 import，所以顶层不要有副作用，重依赖
（cv2 / onnxruntime / imgutils）一律在函数里 import。
"""

import json
import os
import sys
from typing import Dict


class Cancelled(RuntimeError):
    """用户点了取消。靠 on_progress 回调抛出来中断扫描循环。"""


def analyse(
    task_id: str,
    item_id: str,
    video_path: str,
    out_root: str,
    out_name: str,
    gpu_index: int,
    progress,
    cancel,
) -> Dict:
    """跑一段视频，返回写回数据库需要的字段。

    progress / cancel 是 multiprocessing.Manager 的字典代理：本机 1 张显卡、
    Web 是单个 uvicorn 进程，这就够了，不需要 Redis。
    """
    # 必须在 import onnxruntime 之前设好，否则绑卡不生效。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # 模型已缓存；离线模式免掉 imgutils 每个新进程首次检测的联网列表请求。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    core = os.path.abspath(os.path.join(os.path.dirname(__file__), "core"))
    if core not in sys.path:
        sys.path.insert(0, core)

    from config import Config
    from main import force_utf8_stdout, process_video

    force_utf8_stdout()

    def on_progress(stage: str, fraction: float) -> None:
        if cancel.get(task_id):
            raise Cancelled("已取消")
        # Manager 的字典必须整体赋值才会同步回父进程，不能就地改。
        progress[item_id] = {"stage": stage, "fraction": round(float(fraction), 4)}

    on_progress("cuts", 0.0)
    config = Config()   # 运行前零参数：X 只在结果页调
    summary = process_video(
        config, video_path, out_root,
        clip=False,           # 分析时不编码片段：X 还没定，编出来的一大半会被拖走
        out_name=out_name,
        on_progress=on_progress,
    )
    out_dir = summary["output_dir"]
    style = "-"
    windows = os.path.join(out_dir, "windows.json")
    if os.path.isfile(windows):
        with open(windows, encoding="utf-8") as f:
            style = json.load(f).get("style", "-")
    progress[item_id] = {"stage": "done", "fraction": 1.0}
    return {
        "item_id": item_id,
        "out_dir": out_dir,
        "style": style,
        "num_tracks": summary["tracks"],
        "num_characters": summary["characters"],
    }
