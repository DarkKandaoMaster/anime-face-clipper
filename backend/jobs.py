"""任务队列：一个进程池 + 两个共享字典，没有 Redis。

理由（交接文档第 2 节）：本机 1 张显卡 → 并行度 1；Web 是单个 uvicorn 进程
→ 进度用 multiprocessing.Manager().dict() 就够。Redis 留给上公网那一步。
"""

import datetime
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import db
import storage
from settings import settings
from worker import analyse

# 各阶段在进度条里占的权重。scan 是整条流水线里唯一昂贵的一段（GPU 检测），
# 其余几步在长片上也就几秒到几十秒。名字与 main.on_progress 报的 stage 一一对应。
STAGE_WEIGHTS: List[Tuple[str, float, str]] = [
    ("cuts", 0.10, "切镜检测"),
    ("scan", 0.75, "抽帧检测 · 过滤跟踪"),
    ("cluster", 0.10, "角色聚类"),
    ("select", 0.05, "选段"),
]


def percent(stage: str, fraction: float) -> int:
    """(阶段, 阶段内进度) → 全局百分比。"""
    if stage == "done":
        return 100
    done = 0.0
    for name, weight, _label in STAGE_WEIGHTS:
        if name == stage:
            return int(round((done + weight * max(0.0, min(1.0, fraction))) * 100))
        done += weight
    return 0


class Jobs:
    """进程池的持有者。应用启动时建一次，关闭时收掉。"""

    def __init__(self) -> None:
        self._manager = multiprocessing.Manager()
        self.progress = self._manager.dict()
        self.cancel = self._manager.dict()
        # 并行度 = 显卡数：同一张卡上多进程跑多个视频实测没有收益
        # （README 第五轮：两部片 145s vs 147s，瓶颈全在 GPU）。
        self.pool = ProcessPoolExecutor(max_workers=settings.gpu_count)
        # task_id -> [(item_id, future)]，回收时要靠它认出是哪个 item。
        self._futures: Dict[str, List[Tuple[str, object]]] = {}

    def submit(self, task_id: str, items: List[Dict]) -> None:
        """items: [{item_id, video_path, out_name}]，顺序即排队顺序。"""
        out_root = storage.task_dir(task_id)
        self.cancel[task_id] = False
        pairs: List[Tuple[str, object]] = []
        for i, item in enumerate(items):
            self.progress[item["item_id"]] = {"stage": "queued", "fraction": 0.0}
            future = self.pool.submit(
                analyse,
                task_id, item["item_id"], item["video_path"], out_root, item["out_name"],
                i % settings.gpu_count, self.progress, self.cancel,
            )
            future.add_done_callback(
                lambda f, tid=task_id, iid=item["item_id"]: self._finish_item(tid, iid, f)
            )
            pairs.append((item["item_id"], future))
        self._futures[task_id] = pairs
        with db.SessionLocal() as session:
            task = session.get(db.Task, task_id)
            task.status = "running"
            task.started_at = datetime.datetime.now()
            session.commit()

    def _finish_item(self, task_id: str, item_id: str, future) -> None:
        """子进程结束时把结果写回数据库（在池的回调线程里跑）。"""
        try:
            result = future.result()
            self._write_item(item_id, "done", result)
        except Exception as exc:                    # noqa: BLE001 - 原因要带给前端
            cancelled = bool(self.cancel.get(task_id)) or future.cancelled()
            reason = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._write_item(item_id, "cancelled" if cancelled else "failed", {"error": reason})
        self._maybe_finish_task(task_id)

    def _write_item(self, item_id: str, status: str, data: Dict) -> None:
        with db.SessionLocal() as session:
            item = session.get(db.TaskItem, item_id)
            if item is None:
                return
            item.status = status
            item.error = data.get("error")
            if data.get("out_dir"):
                item.out_dir = storage.rel(data["out_dir"])
                item.style = data.get("style")
                item.num_tracks = data.get("num_tracks")
                item.num_characters = data.get("num_characters")
            session.commit()

    def _maybe_finish_task(self, task_id: str) -> None:
        with db.SessionLocal() as session:
            items = session.query(db.TaskItem).filter(db.TaskItem.task_id == task_id).all()
            if any(i.status in ("queued", "running") for i in items):
                return
            task = session.get(db.Task, task_id)
            if task is None or task.status in ("done", "failed", "cancelled"):
                return
            if items and all(i.status == "cancelled" for i in items):
                task.status = "cancelled"
            elif any(i.status == "failed" for i in items):
                task.status = "failed"
                task.error = next((i.error for i in items if i.error), None)
            else:
                task.status = "done"
            task.finished_at = datetime.datetime.now()
            session.commit()

    def cancel_task(self, task_id: str) -> None:
        """还没开跑的直接撤单；已经在跑的靠 on_progress 回调抛异常中断。"""
        self.cancel[task_id] = True
        for _item_id, future in self._futures.get(task_id, []):
            future.cancel()

    def effective_status(self, db_status: str, item_id: str) -> str:
        """数据库里没有 "running" 这个状态，它是从进度推出来的。

        原因：进程池只在**结束**时给回调，开跑这件事父进程没有钩子。而子进程
        每报一次进度就等于说"我在跑"，拿它当依据比让子进程另开一条数据库连接
        去写一行状态更省事。代价是 Web 进程崩了之后残留的 item 会停在 queued。
        """
        if db_status != "queued":
            return db_status
        stage = (self.progress.get(item_id) or {}).get("stage")
        return "running" if stage not in (None, "queued") else "queued"

    def item_progress(self, item_id: str) -> Dict:
        return dict(self.progress.get(item_id) or {"stage": "queued", "fraction": 0.0})

    def shutdown(self) -> None:
        self.pool.shutdown(cancel_futures=True)
        self._manager.shutdown()


_jobs: Optional[Jobs] = None


def start() -> Jobs:
    global _jobs
    if _jobs is None:
        _jobs = Jobs()
    return _jobs


def stop() -> None:
    global _jobs
    if _jobs is not None:
        _jobs.shutdown()
        _jobs = None


def get_jobs() -> Jobs:
    if _jobs is None:
        raise RuntimeError("任务池还没启动。")
    return _jobs
