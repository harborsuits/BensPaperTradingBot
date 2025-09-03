import os, time, jwt
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginReq(BaseModel):
    username: str
    password: str

class LoginRes(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_in: int

@router.post("/login", response_model=LoginRes)
def login(body: LoginReq):
    if body.username != os.getenv("ADMIN_USERNAME", "admin") or body.password != os.getenv("ADMIN_PASSWORD", ""):
        raise HTTPException(401, "Invalid credentials")
    secret = os.getenv("JWT_SECRET", "devsecret")
    exp = int(time.time()) + 60*60*8
    token = jwt.encode({"sub": body.username, "exp": exp}, secret, algorithm="HS256")
    return LoginRes(token=token, expires_in=60*60*8)

def verify_jwt(token: str):
    try:
        return jwt.decode(token, os.getenv("JWT_SECRET","devsecret"), algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
