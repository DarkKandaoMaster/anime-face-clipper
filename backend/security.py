"""共享访问码。

不做用户系统：公网上传 + GPU + 下载，无门槛等于开放转码服务器；一个环境变量
加一个依赖项就够。校验通过后种一个 HttpOnly cookie——`<video src>`、`<img src>`
和下载链接都是浏览器直接发的请求，带不了自定义头，只有 cookie 能过去。
"""

import hmac

from fastapi import Cookie, Header, HTTPException, Response

from settings import settings

COOKIE_NAME = "headcount_access"


def is_valid(code: str) -> bool:
    return bool(code) and hmac.compare_digest(code, settings.access_code)


def set_cookie(response: Response, code: str) -> None:
    response.set_cookie(
        COOKIE_NAME, code, httponly=True, samesite="lax", max_age=7 * 24 * 3600, path="/",
    )


async def require_access(
    headcount_access: str = Cookie(default=""),
    x_access_code: str = Header(default=""),
) -> None:
    """所有接口的守门人（/api/auth/verify 除外）。"""
    if not (is_valid(headcount_access) or is_valid(x_access_code)):
        raise HTTPException(status_code=401, detail="访问码不正确或已失效。")
