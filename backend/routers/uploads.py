"""分片上传：init → chunk ×N → complete。

为什么分片：整个 POST 一个大文件会撞反向代理的 client_max_body_size
（Nginx 默认 1 MB），报 413 且很难查。分片 8 MB。

上传与分析是两个阶段，视频必须完整落盘才能开跑：cv2.VideoCapture /
ffmpeg scdet / ffmpeg -ss 都要一个能随机定位的文件路径，吃不了字节流。
（所谓「流式不落盘」指的是解码出来的**帧**不落盘，不是视频文件。）
"""

import os
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

import db
import media
import storage
from security import require_access

router = APIRouter(prefix="/api/uploads", tags=["uploads"], dependencies=[Depends(require_access)])

CHUNK_SIZE = 8 * 1024 * 1024


class InitIn(BaseModel):
    filename: str
    size: int


@router.post("/init")
def init(payload: InitIn):
    upload_id = uuid.uuid4().hex
    directory = storage.upload_dir(upload_id)
    with open(os.path.join(directory, "name.txt"), "w", encoding="utf-8") as f:
        f.write(payload.filename)
    # 先占好位：分片可能乱序到达，直接按偏移写。
    open(os.path.join(directory, "blob"), "wb").close()
    return {"upload_id": upload_id, "chunk_size": CHUNK_SIZE}


@router.post("/{upload_id}/chunk")
async def chunk(upload_id: str, index: int = Form(...), chunk: UploadFile = File(...)):
    directory = storage.upload_dir(upload_id)
    blob = os.path.join(directory, "blob")
    if not os.path.isfile(blob):
        raise HTTPException(status_code=404, detail="上传会话不存在或已完成。")
    data = await chunk.read()
    with open(blob, "r+b") as f:
        f.seek(index * CHUNK_SIZE)
        f.write(data)
    return {"received": len(data)}


@router.post("/{upload_id}/complete")
def complete(upload_id: str, session: Session = Depends(db.session_scope)):
    """把分片拼成的文件收进素材库：探一次格式，抽封面，写库。

    格式不对就地删文件并报错——不做转码，也不留半个坏素材在库里。
    """
    directory = storage.upload_dir(upload_id)
    blob = os.path.join(directory, "blob")
    name_file = os.path.join(directory, "name.txt")
    if not (os.path.isfile(blob) and os.path.isfile(name_file)):
        raise HTTPException(status_code=404, detail="上传会话不存在或已完成。")
    with open(name_file, encoding="utf-8") as f:
        filename = f.read().strip() or "video.mp4"

    asset_id = uuid.uuid4().hex
    target = storage.asset_source(asset_id)
    storage.asset_dir(asset_id)
    shutil.move(blob, target)
    shutil.rmtree(directory, ignore_errors=True)

    try:
        info = media.probe(target)
        media.grab_frame(target, min(3.0, info["duration"] / 2), storage.asset_poster(asset_id))
    except media.MediaError as exc:
        shutil.rmtree(os.path.dirname(target), ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc))

    asset = db.Asset(
        id=asset_id,
        filename=filename,
        path=storage.rel(target),
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        codec=info["codec"],
        size_bytes=info["size_bytes"],
    )
    session.add(asset)
    session.commit()
    with open(os.path.join(storage.asset_dir(asset_id), "meta.json"), "w", encoding="utf-8") as f:
        import json

        json.dump({"filename": filename, **info}, f, ensure_ascii=False, indent=2)
    return {
        "id": asset_id, "filename": filename, "duration": info["duration"],
        "width": info["width"], "height": info["height"],
        "codec": info["codec"], "size_bytes": info["size_bytes"],
    }
