"""
Input Validation & Sanitization Utilities
Prevents injection attacks, XSS, and invalid data processing
"""
import re
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, field_validator, ValidationError

logger = logging.getLogger(__name__)


class SanitizationRules:
    """Define sanitization rules for different input types"""

    # SQL injection patterns (also applies to NoSQL)
    INJECTION_PATTERNS = [
        r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
        r"(--|;|'{0,1}\s*(=|<>|<|>|<=|>=)\s*'{0,1})",
        r"(\$\w+)",  # MongoDB operators like $ne, $gt
        r"({.*:.*})",  # JSON-like injection
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers
        r"<iframe[^>]*>",
        r"<embed[^>]*>",
        r"<object[^>]*>",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.",
        r"%2e%2e",
        r"..\\",
    ]

    @staticmethod
    def is_safe_string(
        value: str, check_injection: bool = False, check_xss: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Check if string is safe from common attacks.
        Returns: (is_safe, error_message)
        """
        if not isinstance(value, str):
            return False, "Value must be a string"

        if len(value) > 10000:
            return False, "Input exceeds maximum length (10000 chars)"

        if check_injection:
            for pattern in SanitizationRules.INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    return False, f"Potential injection attack detected (pattern: {pattern})"

        if check_xss:
            for pattern in SanitizationRules.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    return False, f"Potential XSS attack detected (pattern: {pattern})"

        return True, None

    @staticmethod
    def is_safe_filename(filename: str) -> tuple[bool, Optional[str]]:
        """
        Check if filename is safe (prevents path traversal attacks).
        Returns: (is_safe, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"

        # Check for path traversal
        for pattern in SanitizationRules.PATH_TRAVERSAL_PATTERNS:
            if pattern in filename:
                return False, "Filename contains path traversal characters"

        # Only allow alphanumeric, dash, underscore, and dot
        if not re.match(r"^[a-zA-Z0-9._\-]+$", filename):
            return False, "Filename contains invalid characters"

        if len(filename) > 255:
            return False, "Filename exceeds maximum length (255 chars)"

        return True, None

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Basic sanitization: trim and escape dangerous characters.
        """
        # Trim whitespace
        value = value.strip()

        # Limit length
        if len(value) > max_length:
            value = value[:max_length]

        # HTML escape
        value = (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

        return value


class ValidatedInput(BaseModel):
    """Base class for validated input"""

    @field_validator("*", mode="before")
    @classmethod
    def validate_all_fields(cls, v):
        """Validate all fields for injection attacks"""
        if isinstance(v, str):
            # Check for injection patterns
            is_safe, error = SanitizationRules.is_safe_string(
                v, check_injection=True, check_xss=True
            )
            if not is_safe:
                logger.warning(f"Unsafe input detected: {error}")
                raise ValueError(f"Invalid input: {error}")
        return v


class SensorReadingValidation(ValidatedInput):
    """Validated sensor reading input"""

    device_id: str
    temperature: float
    humidity: float
    moisture: Optional[float] = None
    ph: Optional[float] = None

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v):
        """Validate device ID format"""
        if not re.match(r"^[A-Z0-9_-]{1,32}$", v):
            raise ValueError("Invalid device ID format")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature range"""
        if not -20 <= v <= 60:
            raise ValueError("Temperature out of valid range (-20°C to 60°C)")
        return v

    @field_validator("humidity")
    @classmethod
    def validate_humidity(cls, v):
        """Validate humidity range"""
        if not 0 <= v <= 100:
            raise ValueError("Humidity out of valid range (0-100%)")
        return v


class FileUploadValidation:
    """Validate file uploads"""

    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    MAX_FILE_SIZE_MB = 10

    @staticmethod
    def validate_image_upload(
        filename: str, content_type: str, file_size: int
    ) -> tuple[bool, Optional[str]]:
        """
        Validate image file upload.
        Returns: (is_valid, error_message)
        """
        # Check filename
        is_safe, error = SanitizationRules.is_safe_filename(filename)
        if not is_safe:
            return False, error

        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in FileUploadValidation.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"File extension {file_ext} not allowed"

        # Check MIME type
        if content_type not in FileUploadValidation.ALLOWED_IMAGE_TYPES:
            return False, f"MIME type {content_type} not allowed"

        # Check file size
        max_size_bytes = FileUploadValidation.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            return False, f"File size exceeds maximum ({FileUploadValidation.MAX_FILE_SIZE_MB}MB)"

        return True, None


class DatabaseQueryValidation:
    """Validate database queries to prevent injection"""

    @staticmethod
    def validate_sql_input(value: str) -> bool:
        """Check if value is safe for SQL queries"""
        # SQL injection patterns
        dangerous_patterns = [
            r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP)\b)",
            r"(--|;)",
            r"('\s*(OR|AND)\s*')",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.error(f"SQL injection attempt detected: {value[:50]}")
                return False

        return True

    @staticmethod
    def validate_filter_value(key: str, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate filter parameters for database queries.
        Returns: (is_valid, error_message)
        """
        if isinstance(value, str):
            if not DatabaseQueryValidation.validate_sql_input(value):
                return False, f"Invalid value for filter '{key}'"

        return True, None


# Import Path for file operations
from pathlib import Path
