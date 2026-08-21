"""素材库：列表、删除、封面、缩略图、按 Range 播原视频。

预览刻意不编码：`<video>` + HTTP Range 直接播原视频，`currentTime` 跳到片段
起点、播满 30 秒停。零编码零额外磁盘，只有点「下载」时才现场编码。
"""

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import db
import jobs
import media
import results
import storage
from pipeline import pipeline_config
from security import require_access

router = APIRouter(prefix="/api/assets", tags=["assets"], dependencies=[Depends(require_access)])


def _latest_item(session: Session, asset_id: str):
    # 按**任务创建时间**排，不能按 TaskItem.id：id 是随机 uuid，排出来的"最近一次"
    # 是随机的，重跑过的素材会时而显示上一次的状态。
    return (
        session.query(db.TaskItem)
        .join(db.Task, db.TaskItem.task_id == db.Task.id)
        .filter(db.TaskItem.asset_id == asset_id)
        .order_by(db.Task.created_at.desc())
        .first()
    )


@router.get("")
def list_assets(session: Session = Depends(db.session_scope)):
    """素材列表，附带最近一次分析的状态——工作台那张表要的就是这些列。"""
    default_x = pipeline_config().min_events_per_window
    out = []
    for asset in session.query(db.Asset).order_by(db.Asset.created_at).all():
        item = _latest_item(session, asset.id)
        status = "idle"
        if item is not None:
            try:
                status = jobs.get_jobs().effective_status(item.status, item.id)
            except RuntimeError:
                status = item.status
        row = {
            "id": asset.id,
            "filename": asset.filename,
            "duration": asset.duration,
            "width": asset.width,
            "height": asset.height,
            "codec": asset.codec,
            "size_bytes": asset.size_bytes,
            "task_id": item.task_id if item else None,
            "status": status,
            "style": item.style if item else None,
            "num_characters": item.num_characters if item else None,
            "num_segments": None,
            "error": item.error if item else None,
            "percent": 0,
            "stage": None,
        }
        if item is not None:
            try:
                progress = jobs.get_jobs().item_progress(item.id)
                row["stage"] = progress.get("stage")
                row["percent"] = (
                    100 if status == "done"
                    else jobs.percent(progress.get("stage", ""), progress.get("fraction", 0.0))
                )
            except RuntimeError:
                pass
            if status == "done" and item.out_dir:
                out_dir = storage.abs_of(item.out_dir)
                row["num_segments"] = len(results.segments_for(out_dir, default_x))
        out.append(row)
    return out


@router.delete("/{asset_id}")
def delete_asset(asset_id: str, session: Session = Depends(db.session_scope)):
    asset = session.get(db.Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="素材不存在。")
    session.query(db.TaskItem).filter(db.TaskItem.asset_id == asset_id).delete()
    session.delete(asset)
    session.commit()
    shutil.rmtree(os.path.join(storage.ROOT, "assets", asset_id), ignore_errors=True)
    return {"ok": True}


def _source(session: Session, asset_id: str) -> str:
    asset = session.get(db.Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="素材不存在。")
    path = storage.abs_of(asset.path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="源文件已不在磁盘上。")
    return path


@router.get("/{asset_id}/poster")
def poster(asset_id: str, session: Session = Depends(db.session_scope)):
    path = storage.asset_poster(asset_id)
    if not os.path.isfile(path):
        source = _source(session, asset_id)
        media.grab_frame(source, 3.0, path)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{asset_id}/frame")
def frame(asset_id: str, t: float = Query(0.0, ge=0.0),
          session: Session = Depends(db.session_scope)):
    """片段缩略图：结果页每张卡片一张。抽过一次就落盘复用。"""
    source = _source(session, asset_id)
    path = os.path.join(storage.asset_frames_dir(asset_id), f"{t:.1f}.jpg")
    media.grab_frame(source, t, path)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{asset_id}/stream")
def stream(asset_id: str, session: Session = Depends(db.session_scope)):
    """播原视频。Range 由 starlette 的 FileResponse 处理（206 + Content-Range）。"""
    return FileResponse(_source(session, asset_id), media_type="video/mp4")
