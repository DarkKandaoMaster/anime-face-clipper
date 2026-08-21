"""storage/ 下的路径布局。所有路径拼接只走这里。

    storage/
      assets/{asset_id}/source.mp4   上传的原视频
                       /poster.jpg   封面
                       /meta.json    时长/分辨率/编码/体积
                       /frames/      结果页缩略图（按需生成，可随时删）
      uploads/{upload_id}/           分片上传的临时落点
      tasks/{task_id}/{stem}/        run_pipeline 的输出目录
"""

import os
import re

from settings import settings

ROOT = settings.storage_root


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def asset_dir(asset_id: str) -> str:
    return _ensure(os.path.join(ROOT, "assets", asset_id))


def asset_source(asset_id: str) -> str:
    return os.path.join(ROOT, "assets", asset_id, "source.mp4")


def asset_poster(asset_id: str) -> str:
    return os.path.join(ROOT, "assets", asset_id, "poster.jpg")


def asset_frames_dir(asset_id: str) -> str:
    return _ensure(os.path.join(ROOT, "assets", asset_id, "frames"))


def upload_dir(upload_id: str) -> str:
    return _ensure(os.path.join(ROOT, "uploads", upload_id))


def task_dir(task_id: str) -> str:
    return _ensure(os.path.join(ROOT, "tasks", task_id))


def item_dir(task_id: str, stem: str) -> str:
    return os.path.join(ROOT, "tasks", task_id, stem)


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def abs_of(relative: str) -> str:
    return os.path.join(ROOT, relative.replace("/", os.sep))


_UNSAFE = re.compile(r'[\/:*?"<>|\r\n\t]')


def safe_stem(filename: str) -> str:
    """原始文件名 → 能当目录名用的 stem。中文原样保留，只挖掉路径分隔符等非法字符。

    中文必须留住：输出目录名、zip 条目名一路都用它，换成 id 就没法看了
    （中文路径本身是本项目最容易踩的坑，见 CLAUDE.md）。
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = _UNSAFE.sub("_", stem).strip().strip(".")
    return stem or "video"
