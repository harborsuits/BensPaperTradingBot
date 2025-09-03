import os
from fastapi import Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def require_jwt(creds: HTTPAuthorizationCredentials = Depends(security)):
    # Minimal check only for presence; pyjwt verify in auth.py dependency instead
    if not creds or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return creds.credentials

def require_api_key(x_api_key: str = Header(default=None)):
    if os.getenv("REQUIRE_API_KEY", "true").lower() == "true":
        expected = os.getenv("API_KEY_PRIMARY", "")
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")
