"""
OAuth2 + JWT Authentication System
Implements secure authentication with role-based access control
Part of AgriSense Production Optimization Blueprint
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production-use-secrets-manager")
ALGORITHM = "HS256"
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "15"))  # Short-lived tokens
REFRESH_TOKEN_EXP_DAYS = int(os.getenv("REFRESH_TOKEN_EXP_DAYS", "7"))

# Password hashing
# Support bcrypt when available, fall back to pbkdf2_sha256 for environments
pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# ============================================================================
# Pydantic Models
# ============================================================================

class UserRole:
    """User role constants"""
    FARMER = "farmer"
    ADMIN = "admin"
    GUARD = "guard"  # For field security monitoring
    VIEWER = "viewer"  # Read-only access


class Token(BaseModel):
    """OAuth2 token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Decoded JWT token data"""
    username: str
    user_id: str
    role: str
    exp: datetime


class UserBase(BaseModel):
    """Base user model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    role: str = UserRole.FARMER
    disabled: bool = False


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)


class UserInDB(UserBase):
    """User in database with hashed password"""
    user_id: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None


class UserPublic(UserBase):
    """User model for public API responses (no sensitive data)"""
    user_id: str
    created_at: datetime
    last_login: Optional[datetime] = None


# ============================================================================
# Password Hashing
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Fallback: try pbkdf2_sha256 verifier explicitly
        try:
            fallback = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
            return fallback.verify(plain_password, hashed_password)
        except Exception:
            return False


def get_password_hash(password: str) -> str:
    """Hash password"""
    try:
        return pwd_context.hash(password)
    except Exception:
        # Fallback to pbkdf2_sha256 if bcrypt backend fails
        fallback = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        return fallback.hash(password)


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Token payload data
        expires_delta: Optional custom expiration
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: str) -> str:
    """
    Create refresh token with longer expiration
    
    Args:
        user_id: User identifier
        
    Returns:
        Refresh token
    """
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXP_DAYS)
    data = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    """
    Decode and validate JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData object
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        exp: int = payload.get("exp")
        
        if username is None or user_id is None:
            raise credentials_exception
        
        return TokenData(
            username=username,
            user_id=user_id,
            role=role,
            exp=datetime.fromtimestamp(exp)
        )
    except JWTError:
        raise credentials_exception


# ============================================================================
# User Authentication
# ============================================================================

class UserRepository:
    """
    User repository interface
    Implement this with your actual database (SQLite, MongoDB, etc.)
    """
    
    def __init__(self):
        # In-memory storage for demo (replace with actual database)
        self._users = {}
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        return self._users.get(username)
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        for user in self._users.values():
            if user.user_id == user_id:
                return user
        return None
    
    async def create_user(self, user: UserCreate) -> UserInDB:
        """Create new user"""
        import uuid
        
        if user.username in self._users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        user_in_db = UserInDB(
            user_id=str(uuid.uuid4()),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            hashed_password=get_password_hash(user.password),
            created_at=datetime.utcnow(),
            disabled=False
        )
        
        self._users[user.username] = user_in_db
        logger.info(f"✅ Created user: {user.username} ({user.role})")
        
        return user_in_db
    
    async def update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        user = await self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()


# Global user repository
_user_repo = UserRepository()


async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate user with username and password
    
    Args:
        username: Username
        password: Plain password
        
    Returns:
        UserInDB object if authentication successful, None otherwise
    """
    user = await _user_repo.get_user_by_username(username)
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if user.disabled:
        return None
    
    # Update last login
    await _user_repo.update_last_login(user.user_id)
    
    return user


# ============================================================================
# Dependency Injection for Protected Routes
# ============================================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Get current authenticated user from JWT token
    
    Usage in FastAPI routes:
        @app.get("/protected")
        async def protected_route(current_user: UserInDB = Depends(get_current_user)):
            return {"username": current_user.username}
    """
    token_data = decode_token(token)
    
    user = await _user_repo.get_user_by_username(token_data.username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Get current active (non-disabled) user"""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


class RoleChecker:
    """
    Dependency to check user role
    
    Usage:
        @app.get("/admin-only", dependencies=[Depends(RoleChecker([UserRole.ADMIN]))])
        async def admin_endpoint():
            return {"message": "Admin access granted"}
    """
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    async def __call__(self, current_user: UserInDB = Depends(get_current_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return current_user


# Predefined role checkers
require_admin = RoleChecker([UserRole.ADMIN])
require_farmer_or_admin = RoleChecker([UserRole.FARMER, UserRole.ADMIN])


# ============================================================================
# Authentication Routes (to be included in main.py)
# ============================================================================

from fastapi import APIRouter, Form

auth_router = APIRouter(prefix="/auth", tags=["authentication"])


@auth_router.post("/register", response_model=UserPublic)
async def register(user: UserCreate):
    """
    Register new user
    
    Public endpoint - no authentication required
    """
    try:
        user_in_db = await _user_repo.create_user(user)
        return UserPublic(**user_in_db.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@auth_router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible login endpoint
    
    Returns access token and refresh token
    """
    user = await authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=JWT_EXP_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.user_id,
            "role": user.role
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(user.user_id)
    
    logger.info(f"✅ User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_EXP_MINUTES * 60,
        refresh_token=refresh_token
    )


@auth_router.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str = Form(...)):
    """
    Get new access token using refresh token
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        user = await _user_repo.get_user_by_id(user_id)
        
        if not user or user.disabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or disabled"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=JWT_EXP_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "user_id": user.user_id,
                "role": user.role
            },
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=JWT_EXP_MINUTES * 60
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@auth_router.get("/me", response_model=UserPublic)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    """
    Get current user information
    
    Protected endpoint - requires authentication
    """
    return UserPublic(**current_user.dict())


@auth_router.post("/change-password")
async def change_password(
    old_password: str = Form(...),
    new_password: str = Form(..., min_length=8),
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Change user password
    
    Protected endpoint - requires authentication
    """
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    # Update password (implement in your user repository)
    current_user.hashed_password = get_password_hash(new_password)
    logger.info(f"✅ Password changed for user: {current_user.username}")
    
    return {"message": "Password updated successfully"}


# ============================================================================
# Initialization
# ============================================================================

async def create_default_admin():
    """
    Create default admin user if none exists
    Call this during application startup
    """
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "change-this-password")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@agrisense.local")
    
    existing_admin = await _user_repo.get_user_by_username(admin_username)
    
    if not existing_admin:
        admin_user = UserCreate(
            username=admin_username,
            email=admin_email,
            password=admin_password,
            full_name="System Administrator",
            role=UserRole.ADMIN
        )
        
        await _user_repo.create_user(admin_user)
        logger.warning(f"⚠️ Created default admin user: {admin_username}")
        logger.warning(f"⚠️ Default password: {admin_password}")
        logger.warning("⚠️ CHANGE THIS PASSWORD IMMEDIATELY IN PRODUCTION!")


if __name__ == "__main__":
    # Test authentication system
    import asyncio
    
    async def test_auth():
        # Create test user
        test_user = UserCreate(
            username="test_farmer",
            email="farmer@test.com",
            password="secure_password_123",
            full_name="Test Farmer",
            role=UserRole.FARMER
        )
        
        user_in_db = await _user_repo.create_user(test_user)
        print(f"Created user: {user_in_db.username}")
        
        # Test authentication
        authenticated = await authenticate_user("test_farmer", "secure_password_123")
        print(f"Authentication successful: {authenticated is not None}")
        
        # Create token
        token = create_access_token({
            "sub": authenticated.username,
            "user_id": authenticated.user_id,
            "role": authenticated.role
        })
        print(f"Token created: {token[:50]}...")
        
        # Decode token
        token_data = decode_token(token)
        print(f"Token decoded: {token_data.username}, role={token_data.role}")
    
    asyncio.run(test_auth())
