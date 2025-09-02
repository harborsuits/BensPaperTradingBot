# Authentication

The BensBot Trading System implements a comprehensive authentication system to secure APIs, web dashboards, and data endpoints.

## Overview

The authentication system provides:

1. **User Management**: Registration, login, and profile management
2. **Token-based Security**: JWT (JSON Web Token) authentication
3. **Role-based Access Control**: Permission levels for different users
4. **API Key Management**: Secure handling of external service credentials
5. **Session Management**: Refresh tokens and session expiration

## Architecture

The authentication system follows a modern token-based architecture:

```
AuthService
├── UserManager
├── TokenManager
├── PermissionManager
└── ExternalCredentialsManager
```

## Configuration

Authentication is configured through the typed settings system:

```python
class AuthSettings(BaseModel):
    """Authentication configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    min_password_length: int = 8
    require_password_complexity: bool = True
    max_failed_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_cookie_name: str = "trading_session"
    
    @validator('access_token_expire_minutes', 'refresh_token_expire_days')
    def validate_expiry_times(cls, v, field):
        if field.name == 'access_token_expire_minutes' and v < 5:
            raise ValueError("Access token expiry should be at least 5 minutes")
        if field.name == 'refresh_token_expire_days' and v < 1:
            raise ValueError("Refresh token expiry should be at least 1 day")
        return v
```

## User Management

The authentication system manages user accounts:

```python
class User(BaseModel):
    """User model for authentication."""
    id: Optional[int] = None
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    role: str = "user"  # user, admin, readonly
```

## Token-Based Authentication

The system uses JWT tokens for authentication:

```python
class Token(BaseModel):
    """Token model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: int  # Unix timestamp
```

### Token Generation

```python
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a new JWT access token."""
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.api.access_token_expire_minutes)
        
    to_encode.update({"exp": expire})
    
    # Create encoded JWT
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.api.jwt_secret, 
        algorithm=settings.api.jwt_algorithm
    )
    
    return encoded_jwt
```

## Authentication Flow

### Registration Process

1. User submits registration with username, email, and password
2. System validates input and checks for existing users
3. Password is hashed and stored securely
4. New user account is created
5. Welcome email is sent
6. Initial access and refresh tokens are issued

### Login Process

1. User submits username/email and password
2. System verifies credentials
3. If valid, generate access and refresh tokens
4. Return tokens to client
5. Update last login timestamp

### Token Refresh

1. Client submits expired access token and valid refresh token
2. System validates refresh token
3. If valid, generate new access token
4. Return new access token to client

### Authentication Middleware

The system uses FastAPI's dependency injection for authentication:

```python
async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Validate token and return current user."""
    
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.api.jwt_secret, 
            algorithms=[settings.api.jwt_algorithm]
        )
        
        # Extract user ID from payload
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Extract token scopes
        token_scopes = payload.get("scopes", [])
        
    except JWTError:
        raise credentials_exception
        
    # Get user from database
    user = get_user_by_id(db, int(user_id))
    if user is None:
        raise credentials_exception
        
    # Check scopes
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
            
    return user
```

## Security Measures

### Password Hashing

Passwords are securely hashed using Bcrypt:

```python
def get_password_hash(password: str) -> str:
    """Hash a password securely."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)
```

### Rate Limiting

The authentication API implements rate limiting to prevent brute force attacks:

```python
@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint with rate limiting."""
    # ... login logic
```

### Failed Login Protection

The system protects against brute force attacks:

```python
def record_failed_login(username: str):
    """Record a failed login attempt."""
    failed_attempts = failed_login_attempts.get(username, 0) + 1
    failed_login_attempts[username] = failed_attempts
    
    if failed_attempts >= settings.auth.max_failed_login_attempts:
        # Lock account temporarily
        account_lockouts[username] = datetime.utcnow() + timedelta(
            minutes=settings.auth.lockout_duration_minutes
        )
```

## API Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user |
| `/api/auth/login` | POST | Log in and get tokens |
| `/api/auth/refresh` | POST | Refresh access token |
| `/api/auth/logout` | POST | Log out (revoke tokens) |
| `/api/auth/me` | GET | Get current user info |
| `/api/auth/change-password` | POST | Change password |
| `/api/auth/reset-password` | POST | Request password reset |

## External API Integration

The authentication system securely manages external API credentials:

### API Key Management

```python
class APIKeyManager:
    """Manages external API keys securely."""
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service."""
        # Try environment variable first
        env_var_name = f"{service.upper()}_KEY"
        api_key = os.environ.get(env_var_name)
        
        if api_key:
            return api_key
            
        # Try settings
        if service in settings.api.api_keys:
            return settings.api.api_keys[service]
            
        # Try database
        return self._get_api_key_from_db(service)
```

## Client Usage

### React Dashboard

```javascript
// Authentication in React dashboard
const login = async (username, password) => {
  try {
    const response = await fetch('http://localhost:8000/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        'username': username,
        'password': password
      })
    });
    
    if (!response.ok) {
      throw new Error('Login failed');
    }
    
    const data = await response.json();
    
    // Store tokens
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    localStorage.setItem('token_expiry', data.expires_at);
    
    return data;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

// Authenticated API request
const fetchPortfolio = async () => {
  try {
    const token = localStorage.getItem('access_token');
    
    if (!token) {
      throw new Error('Not authenticated');
    }
    
    const response = await fetch('http://localhost:8000/api/v1/portfolio', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (response.status === 401) {
      // Token expired, try refresh
      await refreshToken();
      return fetchPortfolio();
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching portfolio:', error);
    throw error;
  }
};
```

### Python Client

```python
class TradingBotClient:
    """Client for interacting with the trading bot API."""
    
    def __init__(self, base_url, username=None, password=None):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        if username and password:
            self.login(username, password)
    
    def login(self, username, password):
        """Log in to the API."""
        response = self.session.post(
            f"{self.base_url}/api/auth/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self.token_expiry = data["expires_at"]
        
        # Set default authorization header
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}"
        })
        
        return data
    
    def refresh(self):
        """Refresh the access token."""
        response = self.session.post(
            f"{self.base_url}/api/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.token_expiry = data["expires_at"]
        
        # Update authorization header
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}"
        })
        
        return data
```
