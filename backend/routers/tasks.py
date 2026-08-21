"""任务：建、看进度（SSE）、取消、按 X 现算片段、下载。

改 X 走的是 results.segments_for → segments.select_from_scan，也就是流水线
自己那份选段实现，两边不会走偏。
"""

import asyncio
import io
import json
import os
import uuid
import zipfile
from typing import Dict, List
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import db
import jobs
import media
import results
import storage
from pipeline import pipeline_config
from security import require_access

router = APIRouter(prefix="/api/tasks", tags=["tasks"], dependencies=[Depends(require_access)])


class CreateIn(BaseModel):
    asset_ids: List[str]


def _task_or_404(session: Session, task_id: str) -> db.Task:
    task = session.get(db.Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return task


def _items(session: Session, task_id: str) -> List[db.TaskItem]:
    return (session.query(db.TaskItem)
            .filter(db.TaskItem.task_id == task_id)
            .order_by(db.TaskItem.id).all())


def _item_by_stem(session: Session, task_id: str, stem: str) -> db.TaskItem:
    """按输出目录名找 item。顺带就是路径校验：stem 必须真属于这个任务。"""
    for item in _items(session, task_id):
        if item.out_dir and os.path.basename(item.out_dir) == stem:
            return item
    raise HTTPException(status_code=404, detail="这个任务里没有这个片源。")


@router.post("")
def create(payload: CreateIn, session: Session = Depends(db.session_scope)):
    if not payload.asset_ids:
        raise HTTPException(status_code=400, detail="没有选择素材。")
    task_id = uuid.uuid4().hex
    session.add(db.Task(id=task_id, status="queued"))

    submissions = []
    used: Dict[str, int] = {}
    for asset_id in payload.asset_ids:
        asset = session.get(db.Asset, asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"素材 {asset_id} 不存在。")
        # 输出目录名用原始（中文）文件名；同名素材加序号，否则会互相覆盖。
        stem = storage.safe_stem(asset.filename)
        used[stem] = used.get(stem, 0) + 1
        if used[stem] > 1:
            stem = f"{stem}_{used[stem]}"
        item_id = uuid.uuid4().hex
        session.add(db.TaskItem(
            id=item_id, task_id=task_id, asset_id=asset_id, status="queued",
            out_dir=storage.rel(storage.item_dir(task_id, stem)),
        ))
        submissions.append({
            "item_id": item_id,
            "video_path": storage.abs_of(asset.path),
            "out_name": stem,
        })
    session.commit()
    jobs.get_jobs().submit(task_id, submissions)
    return {"task_id": task_id}


@router.get("")
def list_tasks(session: Session = Depends(db.session_scope)):
    rows = session.query(db.Task).order_by(db.Task.created_at.desc()).limit(50).all()
    return [{"id": t.id, "status": t.status,
             "created_at": t.created_at.isoformat() if t.created_at else None} for t in rows]


def _state(session: Session, task_id: str) -> Dict:
    task = _task_or_404(session, task_id)
    pool = jobs.get_jobs()
    items = []
    for item in _items(session, task_id):
        asset = session.get(db.Asset, item.asset_id)
        progress = pool.item_progress(item.id)
        stage = progress.get("stage", "queued")
        status = pool.effective_status(item.status, item.id)
        items.append({
            "item_id": item.id,
            "asset_id": item.asset_id,
            "filename": asset.filename if asset else "?",
            "duration": asset.duration if asset else 0.0,
            "status": status,
            "stage": stage,
            "percent": 100 if status == "done"
                       else jobs.percent(stage, progress.get("fraction", 0.0)),
            "style": item.style,
            "num_tracks": item.num_tracks,
            "num_characters": item.num_characters,
            "error": item.error,
        })
    return {
        "task_id": task.id,
        "status": task.status,
        "stages": [{"key": k, "label": label} for k, _w, label in jobs.STAGE_WEIGHTS],
        "items": items,
    }


@router.get("/{task_id}")
def get_task(task_id: str, session: Session = Depends(db.session_scope)):
    return _state(session, task_id)


@router.get("/{task_id}/events")
async def events(task_id: str, request: Request):
    """SSE 进度流。EventSource 带不了自定义头，但会带 cookie——鉴权走 cookie 正是为此。"""

    def snapshot() -> Dict:
        with db.SessionLocal() as session:
            return _state(session, task_id)

    async def stream():
        while True:
            if await request.is_disconnected():
                break
            state = await asyncio.to_thread(snapshot)
            yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
            if state["status"] in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@router.post("/{task_id}/cancel")
def cancel(task_id: str, session: Session = Depends(db.session_scope)):
    _task_or_404(session, task_id)
    jobs.get_jobs().cancel_task(task_id)
    return {"ok": True}


@router.get("/{task_id}/segments")
def segments(task_id: str, x: int = Query(..., ge=1, le=30),
             session: Session = Depends(db.session_scope)):
    """当前 X 下的全部片段。纯 Python 比较，不碰模型，几十毫秒。"""
    _task_or_404(session, task_id)
    config = pipeline_config()
    items, scans = [], []
    for item in _items(session, task_id):
        if item.status != "done" or not item.out_dir:
            continue
        out_dir = storage.abs_of(item.out_dir)
        scan = results.load_scan(out_dir)
        if not scan:
            continue
        scans.append(scan)
        asset = session.get(db.Asset, item.asset_id)
        crops = results.crop_by_character(out_dir)
        picked = results.segments_for(out_dir, x)
        for segment in picked:
            segment["faces"] = [c for c in segment["characters"] if c in crops]
        items.append({
            "item_id": item.id,
            "asset_id": item.asset_id,
            # 全片所有有代表图的角色：结果页的「角色印相表」用它，
            # 相当于 montage.py 的接触印相表——没有真值时唯一能核对聚类的手段。
            "all_faces": sorted(crops),
            "stem": os.path.basename(item.out_dir),
            "title": os.path.splitext(asset.filename)[0] if asset else "?",
            "style": item.style,
            "duration": scan["duration"],
            "num_characters": item.num_characters,
            "curve": results.curve(scan),
            "segments": picked,
        })
    return {
        "task_id": task_id,
        "x": x,
        "window_seconds": config.window_seconds,
        "sensitivity": results.sensitivity(scans),
        "items": items,
    }


@router.get("/{task_id}/{stem}/crops/{character_id}")
def crop(task_id: str, stem: str, character_id: int,
         session: Session = Depends(db.session_scope)):
    """某个角色的代表裁剪图。character_id 是全片口径，与 crops/ 目录一致。"""
    item = _item_by_stem(session, task_id, stem)
    out_dir = storage.abs_of(item.out_dir)
    relative = results.crop_by_character(out_dir).get(character_id)
    if not relative:
        raise HTTPException(status_code=404, detail="这个角色没有代表裁剪图。")
    path = os.path.join(out_dir, relative.replace("/", os.sep))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="裁剪图文件已不在磁盘上。")
    return FileResponse(path, media_type="image/jpeg")


def _clip_path(out_dir: str, stem: str, start: float) -> str:
    return os.path.join(out_dir, "clips", f"{stem}_{_stamp(start)}.mp4")


def _stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}m{int(seconds % 60):02d}s"


@router.get("/{task_id}/clip")
def clip(task_id: str, stem: str, start: float = Query(..., ge=0.0),
         session: Session = Depends(db.session_scope)):
    """下载单个片段：现场编码。分析阶段刻意没编，见交接文档第 2 节。"""
    item = _item_by_stem(session, task_id, stem)
    asset = session.get(db.Asset, item.asset_id)
    out_dir = storage.abs_of(item.out_dir)
    path = _clip_path(out_dir, stem, start)
    try:
        media.encode_clip(storage.abs_of(asset.path), start, path)
    except media.MediaError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


class _Sink(io.RawIOBase):
    """zipfile 往里写，我们一段段取走——边压边发，不在内存里攒整个包。"""

    def __init__(self) -> None:
        self._chunks: List[bytes] = []

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self._chunks.append(bytes(data))
        return len(data)

    def drain(self) -> bytes:
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


@router.get("/{task_id}/download")
def download(task_id: str, x: int = Query(..., ge=1, le=30),
             session: Session = Depends(db.session_scope)):
    """打包下载当前 X 下的全部片段。

    StreamingResponse 边压边发，压缩级别 0（视频再压没有意义，只会白烧 CPU）。
    任何环节都不 base64：体积 +33%，而且必须整个读进内存。
    zip 条目名保留中文——zipfile 会自动打上 UTF-8 标志位。
    """
    _task_or_404(session, task_id)
    plan = []
    for item in _items(session, task_id):
        if item.status != "done" or not item.out_dir:
            continue
        asset = session.get(db.Asset, item.asset_id)
        out_dir = storage.abs_of(item.out_dir)
        stem = os.path.basename(item.out_dir)
        for segment in results.segments_for(out_dir, x):
            plan.append({
                "source": storage.abs_of(asset.path),
                "out_dir": out_dir,
                "stem": stem,
                "start": segment["start"],
                "count": segment["count"],
            })
    if not plan:
        raise HTTPException(status_code=404, detail="当前门槛下没有命中任何片段。")

    def stream():
        sink = _Sink()
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as zf:
            for entry in plan:
                path = _clip_path(entry["out_dir"], entry["stem"], entry["start"])
                media.encode_clip(entry["source"], entry["start"], path)
                name = (f"{entry['stem']}/{entry['stem']}_"
                        f"{_stamp(entry['start'])}_{entry['count']}人.mp4")
                with zf.open(name, "w") as dst, open(path, "rb") as src:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
                        chunk = sink.drain()
                        if chunk:
                            yield chunk
                chunk = sink.drain()
                if chunk:
                    yield chunk
        yield sink.drain()

    filename = f"headcount_x{x}_{len(plan)}段.zip"
    return StreamingResponse(stream(), media_type="application/zip", headers={
        "Content-Disposition":
            f"attachment; filename*=UTF-8''{quote(filename)}",
    })
