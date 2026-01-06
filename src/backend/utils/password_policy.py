"""
Secure Password Policy Enforcement
Ensures strong passwords meeting security standards
"""
import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PasswordPolicy:
    """
    Password policy requirements for AgriSense.
    Enforces:
    - Minimum 12 characters
    - Mix of uppercase, lowercase, numbers, and special characters
    - No common patterns
    - Not reused from history
    """

    MIN_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_NUMBERS = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    # Common weak passwords to reject
    WEAK_PASSWORDS = {
        "password",
        "admin123",
        "123456",
        "password123",
        "admin",
        "agrisense",
        "farmer",
        "farm123",
        "agri123",
        "qwerty",
        "abc123",
        "welcome",
        "letmein",
        "login",
    }

    @classmethod
    def validate(cls, password: str) -> tuple[bool, Optional[str]]:
        """
        Validate password against policy.
        Returns: (is_valid, error_message)
        """
        if not password:
            return False, "Password cannot be empty"

        if len(password) < cls.MIN_LENGTH:
            return (
                False,
                f"Password must be at least {cls.MIN_LENGTH} characters long (got {len(password)})",
            )

        if password.lower() in cls.WEAK_PASSWORDS:
            return False, "Password is too common - please choose a stronger password"

        # Check for common substitutions
        weak_substitutions = {
            "0": "o",
            "1": "i",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
        }
        normalized = password.lower()
        for digit, char in weak_substitutions.items():
            normalized = normalized.replace(digit, char)

        if normalized in cls.WEAK_PASSWORDS:
            return False, "Password is too similar to a common password"

        # Check character requirements
        has_uppercase = bool(re.search(r"[A-Z]", password))
        has_lowercase = bool(re.search(r"[a-z]", password))
        has_numbers = bool(re.search(r"\d", password))
        has_special = bool(any(char in cls.SPECIAL_CHARS for char in password))

        if cls.REQUIRE_UPPERCASE and not has_uppercase:
            return False, "Password must contain at least one uppercase letter"

        if cls.REQUIRE_LOWERCASE and not has_lowercase:
            return False, "Password must contain at least one lowercase letter"

        if cls.REQUIRE_NUMBERS and not has_numbers:
            return False, "Password must contain at least one number"

        if cls.REQUIRE_SPECIAL and not has_special:
            return (
                False,
                f"Password must contain at least one special character: {cls.SPECIAL_CHARS}",
            )

        # Check for sequential patterns
        if cls._has_sequential_pattern(password):
            return False, "Password contains sequential characters - please choose a different password"

        return True, None

    @staticmethod
    def _has_sequential_pattern(password: str, min_length: int = 3) -> bool:
        """Check for sequential characters (abc, 123, etc.)"""
        password_lower = password.lower()

        # Check for number sequences
        for i in range(10 - min_length):
            if str(i) * min_length in password_lower:
                return True

        # Check for letter sequences
        for i in range(ord("a"), ord("z") - min_length):
            pattern = "".join(chr(c) for c in range(i, i + min_length))
            if pattern in password_lower:
                return True

        return False

    @classmethod
    def get_requirements_text(cls) -> str:
        """Get human-readable password requirements"""
        requirements = [
            f"At least {cls.MIN_LENGTH} characters long",
            "Include uppercase letters (A-Z)" if cls.REQUIRE_UPPERCASE else None,
            "Include lowercase letters (a-z)" if cls.REQUIRE_LOWERCASE else None,
            "Include numbers (0-9)" if cls.REQUIRE_NUMBERS else None,
            f"Include special characters ({cls.SPECIAL_CHARS})" if cls.REQUIRE_SPECIAL else None,
            "No sequential patterns (abc, 123, etc.)",
            "Not a common password",
        ]

        return "\n".join(f"• {req}" for req in requirements if req)


class PasswordHasher:
    """
    Secure password hashing using bcrypt with salt rounds.
    """

    # Use passlib's CryptContext for password hashing
    from passlib.context import CryptContext

    pwd_context = CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
        bcrypt__rounds=12,  # Higher rounds = slower but more secure
    )

    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hash password using bcrypt"""
        return cls.pwd_context.hash(password)

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return cls.pwd_context.verify(plain_password, hashed_password)

    @classmethod
    def needs_rehash(cls, hashed_password: str) -> bool:
        """Check if hash needs updating (rounds changed, etc.)"""
        return cls.pwd_context.needs_update(hashed_password)


class PasswordHistory:
    """
    Track password history to prevent reuse.
    In production, store these in database.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        # In production: Load from database
        self.history: dict = {}

    def add_password(self, username: str, hashed_password: str) -> None:
        """Add password to history"""
        if username not in self.history:
            self.history[username] = []

        self.history[username].append(hashed_password)

        # Keep only last N passwords
        if len(self.history[username]) > self.max_history:
            self.history[username] = self.history[username][-self.max_history :]

    def check_reuse(self, username: str, new_password: str) -> bool:
        """
        Check if new password reuses a previous password.
        Returns: True if reuse detected (password should be rejected)
        """
        if username not in self.history:
            return False

        for hashed_previous in self.history[username]:
            if PasswordHasher.verify_password(new_password, hashed_previous):
                logger.warning(
                    f"User {username} attempted to reuse a previous password"
                )
                return True

        return False
