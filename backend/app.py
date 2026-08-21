"""FastAPI 应用入口。

跑法（项目根目录，PowerShell）：
    & $env:PYTHON_BIN -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

平铺导入（`import db`、`import storage`），与 backend/core/ 同一风格，所以必须
`--app-dir backend`。不要开 --reload：进程池和 Manager 会跟着重启，跑到一半的
任务会被切断。
"""

import contextlib

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import db
import jobs
import media
from routers import assets, auth, tasks, uploads


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    jobs.start()
    try:
        yield
    finally:
        jobs.stop()


app = FastAPI(title="headcount-30s", lifespan=lifespan)


@app.exception_handler(media.MediaError)
async def media_error(_request, exc: media.MediaError):
    """ffprobe/ffmpeg 的失败原因原样回给前端——第 9 节第 5 条要的就是这个。"""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(auth.router)
app.include_router(uploads.router)
app.include_router(assets.router)
app.include_router(tasks.router)


@app.get("/api/health")
def health():
    from settings import settings

    # 只回结构性信息，不回任何凭据值。
    return {"ok": True, "gpu_count": settings.gpu_count}
