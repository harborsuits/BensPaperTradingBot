"""
User authentication models.
"""

import os
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import jwt
from pydantic import BaseModel, EmailStr, Field, validator
import uuid

# Import typed settings
from trading_bot.config.typed_settings import APISettings, TradingBotSettings, load_config

# Setup logging
logger = logging.getLogger(__name__)

# Load API settings if available
api_settings = None
try:
    config = load_config()
    api_settings = config.api
    logger.debug("Loaded API settings from typed config")
except Exception as e:
    logger.warning(f"Could not load typed API settings: {str(e)}. Using defaults.")
    api_settings = APISettings()

# Secret key for JWT tokens - load from environment variable
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "YOUR_SECRET_KEY_CHANGE_ME")
JWT_ALGORITHM = "HS256"
# Use typed settings token expiry if available, otherwise fall back to env vars or defaults
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", str(api_settings.token_expiry_minutes if api_settings else 30)))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class UserInDB(BaseModel):
    """User model stored in the database"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_user_create(cls, user_create: 'UserCreate') -> 'UserInDB':
        """Create a UserInDB from UserCreate model"""
        hashed_password = cls.get_password_hash(user_create.password)
        return cls(
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password
        )
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password for storing"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, plain_password: str) -> bool:
        """Verify a stored password against a provided password"""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            self.hashed_password.encode('utf-8')
        )
    
    def create_access_token(self) -> str:
        """Create a new access token for the user"""
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": self.id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "exp": expire
        }
        return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    def create_refresh_token(self) -> str:
        """Create a new refresh token for the user"""
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = {
            "sub": self.id,
            "token_type": "refresh",
            "exp": expire
        }
        return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)


class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: EmailStr
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserResponse(BaseModel):
    """User response model (without sensitive data)"""
    id: str
    username: str
    email: EmailStr
    is_admin: bool
    created_at: datetime


class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Token payload model"""
    sub: Optional[str] = None
    username: Optional[str] = None
    exp: Optional[int] = None


class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    username: Optional[str] = None
    is_admin: Optional[bool] = False 