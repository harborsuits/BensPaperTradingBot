"""
Authentication service for trading bot users.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Import typed settings
from trading_bot.config.typed_settings import APISettings, TradingBotSettings, load_config

from trading_bot.auth.models import (
    SECRET_KEY, JWT_ALGORITHM, 
    TokenData, UserInDB, UserCreate, UserResponse
)

# Setup logging
logger = logging.getLogger(__name__)

# Load API settings if available
api_settings = None
try:
    config = load_config()
    api_settings = config.api
    logger.info("Loaded API settings from typed config")
except Exception as e:
    logger.warning(f"Could not load typed API settings: {str(e)}. Using defaults.")
    api_settings = APISettings()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Path to users database file (try from config first, fall back to env var or default)
USERS_DB_PATH = os.environ.get("USERS_DB_PATH", "data/users.json")

# Ensure the directory exists
os.makedirs(os.path.dirname(USERS_DB_PATH), exist_ok=True)


class AuthService:
    """Service for user authentication and management"""

    @staticmethod
    def get_users() -> Dict[str, UserInDB]:
        """Get all users from the database"""
        try:
            if os.path.exists(USERS_DB_PATH):
                with open(USERS_DB_PATH, "r") as f:
                    data = json.load(f)
                    return {
                        user_id: UserInDB(**user_data)
                        for user_id, user_data in data.items()
                    }
            return {}
        except Exception as e:
            logger.error(f"Error reading users database: {str(e)}")
            return {}

    @staticmethod
    def save_users(users: Dict[str, UserInDB]) -> bool:
        """Save users to the database"""
        try:
            # Convert user objects to dictionaries
            users_dict = {
                user_id: user.dict()
                for user_id, user in users.items()
            }
            
            with open(USERS_DB_PATH, "w") as f:
                json.dump(users_dict, f, indent=2, default=str)
            
            return True
        except Exception as e:
            logger.error(f"Error saving users database: {str(e)}")
            return False

    @classmethod
    def get_user_by_id(cls, user_id: str) -> Optional[UserInDB]:
        """Get a user by ID"""
        users = cls.get_users()
        return users.get(user_id)

    @classmethod
    def get_user_by_username(cls, username: str) -> Optional[UserInDB]:
        """Get a user by username"""
        users = cls.get_users()
        for user in users.values():
            if user.username.lower() == username.lower():
                return user
        return None

    @classmethod
    def get_user_by_email(cls, email: str) -> Optional[UserInDB]:
        """Get a user by email"""
        users = cls.get_users()
        for user in users.values():
            if user.email.lower() == email.lower():
                return user
        return None

    @classmethod
    def create_user(cls, user_create: UserCreate) -> UserInDB:
        """Create a new user"""
        # Check if username exists
        if cls.get_user_by_username(user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email exists
        if cls.get_user_by_email(user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Create new user
        new_user = UserInDB.from_user_create(user_create)
        
        # Save user to database
        users = cls.get_users()
        users[new_user.id] = new_user
        cls.save_users(users)
        
        return new_user

    @classmethod
    def authenticate_user(cls, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user with username and password"""
        # Find user by username
        user = cls.get_user_by_username(username)
        
        # Check if user exists and password is correct
        if user and user.verify_password(password):
            return user
        
        return None

    @staticmethod
    def decode_token(token: str) -> TokenData:
        """Decode and validate a JWT token"""
        try:
            # Decode token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Extract user ID from token
            user_id = payload.get("sub")
            username = payload.get("username")
            is_admin = payload.get("is_admin", False)
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Return token data
            return TokenData(
                user_id=user_id,
                username=username,
                is_admin=is_admin
            )
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @classmethod
    def get_current_user(cls, token: str = Depends(oauth2_scheme)) -> UserInDB:
        """Get the current user from token"""
        # Decode token
        token_data = cls.decode_token(token)
        
        # Get user by ID
        user = cls.get_user_by_id(token_data.user_id)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user

    @classmethod
    def get_current_active_user(cls, token: str = Depends(oauth2_scheme)) -> UserInDB:
        """Get the current active user"""
        return cls.get_current_user(token)


# Optional dependency for endpoints that accept anonymous users
def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    try:
        return AuthService.get_current_user(token)
    except Exception:
        return None

    @classmethod
    def user_to_response(cls, user: UserInDB) -> UserResponse:
        """Convert a UserInDB to UserResponse"""
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_admin=user.is_admin,
            created_at=user.created_at
        ) 