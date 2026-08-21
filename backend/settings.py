"""环境配置：只从环境变量（或项目根 .env）读，缺一个就退出。

凭据一律不写默认值、不进日志——缺什么就报条目名，让人自己去补。
"""

import os
from typing import Dict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_REQUIRED = (
    "MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB",
    "STORAGE_ROOT", "APP_ACCESS_CODE", "PYTHON_BIN", "GPU_COUNT",
)


def _load_dotenv(path: str) -> None:
    """把 .env 里还没在环境里的键塞进 os.environ。

    自己解析而不是引 python-dotenv：需要的只有 KEY=VALUE 和 # 注释两条规则。
    已存在的环境变量优先，方便临时覆盖。
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings:
    """一次性读齐所有环境变量；缺项直接 RuntimeError。"""

    def __init__(self) -> None:
        _load_dotenv(os.path.join(_ROOT, ".env"))
        missing = [k for k in _REQUIRED if not os.environ.get(k)]
        if missing:
            raise RuntimeError(
                "缺少环境变量：" + ", ".join(missing) + "；请在项目根 .env 里补齐（不要写进代码）"
            )
        env: Dict[str, str] = os.environ
        self.root = _ROOT
        self.mysql_host = env["MYSQL_HOST"]
        self.mysql_port = int(env["MYSQL_PORT"])
        self.mysql_user = env["MYSQL_USER"]
        self.mysql_password = env["MYSQL_PASSWORD"]
        self.mysql_db = env["MYSQL_DB"]
        self.storage_root = os.path.abspath(os.path.join(_ROOT, env["STORAGE_ROOT"]))
        self.access_code = env["APP_ACCESS_CODE"]
        self.python_bin = env["PYTHON_BIN"]
        self.gpu_count = max(1, int(env["GPU_COUNT"]))

    @property
    def database_url(self) -> str:
        from urllib.parse import quote_plus

        return (
            f"mysql+pymysql://{quote_plus(self.mysql_user)}:{quote_plus(self.mysql_password)}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


settings = Settings()
