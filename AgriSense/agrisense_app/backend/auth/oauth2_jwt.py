"""
OAuth2 + JWT Authentication System
Role-Based Access Control (RBAC) for AgriSense
"""
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ..config.optimization import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UserRole(str, Enum):
    """User roles for RBAC"""
    FARMER = "farmer"
    ADMIN = "admin"
    GUARD = "guard"  # Security personnel for BLUE SHIELD AI
    TECHNICIAN = "technician"


class User(BaseModel):
    """User model"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    role: UserRole = UserRole.FARMER


class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = settings.jwt_exp_minutes * 60


class TokenData(BaseModel):
    """JWT token payload"""
    username: Optional[str] = None
    role: Optional[UserRole] = None


# ===== PASSWORD UTILITIES =====

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


# ===== JWT UTILITIES =====

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_exp_minutes)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (longer expiration)"""
    expires_delta = timedelta(days=settings.jwt_refresh_exp_days)
    return create_access_token(data, expires_delta)


def decode_token(token: str) -> TokenData:
    """
    Decode and validate JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData with username and role
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, role=role)
        return token_data
    
    except JWTError:
        raise credentials_exception


# ===== USER DATABASE (Replace with real DB) =====

# In production, replace this with database queries
fake_users_db = {
    "farmer1": {
        "username": "farmer1",
        "full_name": "John Farmer",
        "email": "farmer@example.com",
        "hashed_password": get_password_hash("password123"),
        "disabled": False,
        "role": UserRole.FARMER,
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@agrisense.ai",
        "hashed_password": get_password_hash("admin123"),
        "disabled": False,
        "role": UserRole.ADMIN,
    },
}


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ===== DEPENDENCY INJECTION =====

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current authenticated user from JWT token.
    Use as FastAPI dependency.
    """
    token_data = decode_token(token)
    user = get_user(username=token_data.username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (not disabled)"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ===== ROLE-BASED ACCESS CONTROL =====

class RoleChecker:
    """
    Check if user has required role.
    
    Usage:
        @app.get("/admin", dependencies=[Depends(RoleChecker([UserRole.ADMIN]))])
        async def admin_endpoint():
            return {"message": "Admin only"}
    """
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_active_user)):
        if current_user.role not in self.allowed_roles:
            logger.warning(
                f"User {current_user.username} with role {current_user.role} "
                f"attempted to access endpoint requiring {self.allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in self.allowed_roles]}"
            )
        return current_user


# Convenience role checkers
require_admin = RoleChecker([UserRole.ADMIN])
require_farmer_or_admin = RoleChecker([UserRole.FARMER, UserRole.ADMIN])
require_technician = RoleChecker([UserRole.TECHNICIAN, UserRole.ADMIN])
