"""MySQL 表定义与会话。

只存任务元数据：视频、切片、裁剪图这些二进制全部留在文件系统（大文件进
数据库要整个读进内存，而且没法做 HTTP Range 断点续传）。
**不建片段表**——片段随 X 变化，是算出来的，不是存下来的。
"""

import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from settings import settings

Base = declarative_base()
engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _now() -> datetime.datetime:
    return datetime.datetime.now()


class Asset(Base):
    __tablename__ = "assets"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)   # 原始文件名，中文原样保留
    path = Column(String(512), nullable=False)       # 相对 STORAGE_ROOT
    duration = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    codec = Column(String(32), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    # queued / running / done / failed / cancelled
    status = Column(String(16), nullable=False, default="queued")
    created_at = Column(DateTime, default=_now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)


class TaskItem(Base):
    __tablename__ = "task_items"

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    asset_id = Column(String(36), ForeignKey("assets.id"), nullable=False)
    status = Column(String(16), nullable=False, default="queued")
    style = Column(String(8), nullable=True)
    num_tracks = Column(Integer, nullable=True)
    num_characters = Column(Integer, nullable=True)
    out_dir = Column(String(512), nullable=True)     # 相对 STORAGE_ROOT
    error = Column(Text, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def session_scope():
    """给请求处理函数用的会话（FastAPI 依赖）。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
