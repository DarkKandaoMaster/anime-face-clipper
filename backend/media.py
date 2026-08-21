"""ffprobe / ffmpeg 的薄封装：探测、封面、缩略图、现场切片。

切片参数（编码器、窗口长度）从 backend/core/config.py 的 Config 读，不另立一套；
但这里不 import backend/core/main.py——那会把 cv2 / onnxruntime / imgutils 一起拖进
Web 进程（几百 MB 常驻），只为 shell 出一条 ffmpeg 命令不值得。
"""

import json
import os
import subprocess
from typing import Dict

from pipeline import pipeline_config


class MediaError(RuntimeError):
    """探测失败或格式不受支持。消息会原样回给前端，所以要写清楚。"""


def _run(cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def probe(path: str) -> Dict:
    """探一次视频，返回 duration/width/height/codec/size_bytes。

    上传完必须探这一次：`cv2.VideoCapture` / `ffmpeg scdet` / `ffmpeg -ss` 都要
    一个能随机定位的 H.264 MP4，不是就 fail fast 报错，不做转码。
    """
    config = pipeline_config()
    result = _run([
        config.ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ])
    if result.returncode != 0:
        raise MediaError(f"ffprobe 打不开这个文件：{result.stderr.strip()[-400:]}")
    info = json.loads(result.stdout)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise MediaError("文件里没有视频流。")
    video = streams[0]
    fmt = info.get("format", {})
    codec = video.get("codec_name", "?")
    format_name = fmt.get("format_name", "")
    if codec != "h264":
        raise MediaError(f"只支持 H.264，这个文件是 {codec}。请先转码后再上传（本工具不做转码）。")
    if "mp4" not in format_name:
        raise MediaError(f"只支持 MP4 容器，这个文件是 {format_name}。请先转封装后再上传。")
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise MediaError("拿不到时长，文件可能损坏或未写完。")
    return {
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "codec": codec,
        "size_bytes": os.path.getsize(path),
    }


def grab_frame(video_path: str, t: float, out_path: str, width: int = 480) -> str:
    """抽一帧存成 JPEG（封面与结果页缩略图都用它）。已存在就直接复用。"""
    if os.path.isfile(out_path):
        return out_path
    config = pipeline_config()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = _run([
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", out_path,
    ])
    if result.returncode != 0 or not os.path.isfile(out_path):
        raise MediaError(f"抽帧失败：{result.stderr.strip()[-400:]}")
    return out_path


def encode_clip(video_path: str, start: float, out_path: str) -> str:
    """现场编码一个 30 秒片段（GPU 编码器失败就回退 CPU）。

    分析阶段刻意不编码片段（等价 --no-clip）：X 还没定，编出来的一大半会被
    拖走。只有点「下载」时才走到这里。命令与 main._encode_clip 一致。
    """
    if os.path.isfile(out_path):
        return out_path
    config = pipeline_config()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for encoder in (config.encoder, config.encoder_fallback):
        result = _run([
            config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", video_path,
            "-t", f"{config.window_seconds:.3f}",
            "-c:v", encoder, "-c:a", "aac", "-movflags", "+faststart",
            out_path,
        ])
        if result.returncode == 0 and os.path.isfile(out_path):
            return out_path
    raise MediaError(f"片段编码失败（{config.encoder} 与 {config.encoder_fallback} 都失败）")
