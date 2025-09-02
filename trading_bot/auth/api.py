"""
Authentication API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from trading_bot.auth.models import (
    Token, UserCreate, UserResponse, UserInDB
)
from trading_bot.auth.service import AuthService

# Create router
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(user_create: UserCreate):
    """Register a new user"""
    # Create user
    user = AuthService.create_user(user_create)
    
    # Return user response
    return AuthService.user_to_response(user)


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with username and password"""
    # Authenticate user
    user = AuthService.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = user.create_access_token()
    refresh_token = user.create_refresh_token()
    
    # Return tokens
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    try:
        # Decode refresh token
        token_data = AuthService.decode_token(refresh_token)
        
        # Get user
        user = AuthService.get_user_by_id(token_data.user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create new tokens
        access_token = user.create_access_token()
        new_refresh_token = user.create_refresh_token()
        
        # Return tokens
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(current_user: UserInDB = Depends(AuthService.get_current_active_user)):
    """Get current user information"""
    return AuthService.user_to_response(current_user) 