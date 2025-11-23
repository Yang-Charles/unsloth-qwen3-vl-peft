from fastapi import FastAPI, Request, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated
import uvicorn
import logging

app = FastAPI()


# 1. 定义返回格式
class UserOut(BaseModel):
    id: int
    name: str
    age: int


# 2. 路由：接收 GET 查询参数并返回 JSON
@app.get("/user", response_model=UserOut)
def get_user(
    user_id: int = Query(..., description="用户ID", gt=0),
    name: str = Query(..., description="用户名", min_length=1, max_length=20),
    age: int = Query(..., description="年龄", ge=0, le=120),
):
    """浏览器 / curl 访问：
    http://localhost:8000/user?user_id=1&name=Tom&age=18
    """
    return UserOut(id=user_id, name=name, age=age)


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
