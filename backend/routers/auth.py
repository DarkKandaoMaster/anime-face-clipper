"""访问码校验。整个鉴权就这一个接口 + security.require_access 一个依赖项。"""

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel

import security

router = APIRouter(prefix="/api/auth", tags=["auth"])


class VerifyIn(BaseModel):
    code: str


@router.post("/verify")
def verify(payload: VerifyIn, response: Response):
    """校验访问码，通过就种 cookie。

    错的码一律回同一句话，不区分"空"和"不对"，也不回显用户输入的值。
    """
    if not security.is_valid(payload.code):
        raise HTTPException(status_code=401, detail="访问码不正确。")
    security.set_cookie(response, payload.code)
    return {"ok": True}


@router.get("/session")
def session(headcount_access: str = Cookie(default="")):
    """开页时问一句「我这张 cookie 还算数吗」，省得每个接口都先吃一个 401。"""
    return {"ok": security.is_valid(headcount_access)}
