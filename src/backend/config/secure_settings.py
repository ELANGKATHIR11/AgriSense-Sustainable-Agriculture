"""
Secure environment configuration for AgriSense
Prevents exposure of sensitive data and enforces security best practices
"""
import os
import warnings
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class SecureSettings(BaseSettings):
    """
    Secure settings with validation for production environments.
    Ensures no hardcoded secrets and proper security configuration.
    """

    # ===== ENVIRONMENT & DEBUG =====
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = environment == "development"

    # Validate that DEBUG is False in production
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    # ===== SECURITY: SECRETS & KEYS =====
    # CRITICAL: These MUST be set in environment variables in production
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_exp_minutes: int = int(os.getenv("JWT_EXP_MINUTES", "15"))

    # Database encryption key (for Fernet encryption)
    database_encryption_key: Optional[str] = os.getenv("DATABASE_ENCRYPTION_KEY", None)

    # API Keys (external services)
    openweather_api_key: Optional[str] = os.getenv("OPENWEATHER_API_KEY", None)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    # HTTPS & SSL
    https_only: bool = self.is_production
    ssl_verify: bool = True
    ssl_cert_path: Optional[str] = os.getenv("SSL_CERT_PATH", None)
    ssl_key_path: Optional[str] = os.getenv("SSL_KEY_PATH", None)

    # ===== DATABASE SECURITY =====
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./sensors.db")
    # CRITICAL: Validate database URL format
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    db_pool_pre_ping: bool = True  # Verify connections before using
    db_echo: bool = False  # Don't log all SQL statements
    db_isolation_level: str = "READ_COMMITTED"  # Prevent dirty reads

    # ===== AUTHENTICATION & AUTHORIZATION =====
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15

    # Token security
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    token_url: str = "/auth/login"

    # ===== RATE LIMITING & THROTTLING =====
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_ml_requests_per_minute: int = 10
    # Per-user limits
    per_user_rate_limit: int = 1000  # requests per hour

    # ===== CORS & SECURITY HEADERS =====
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: list = ["*"]
    cors_max_age: int = 600  # 10 minutes

    # ===== INPUT VALIDATION =====
    max_file_upload_size_mb: int = 10
    allowed_image_formats: list = ["jpg", "jpeg", "png", "webp"]
    allowed_document_formats: list = ["pdf", "xlsx", "csv"]

    # ===== SENSOR DATA VALIDATION =====
    sensor_min_temp: float = -20.0
    sensor_max_temp: float = 60.0
    sensor_min_humidity: float = 0.0
    sensor_max_humidity: float = 100.0
    sensor_min_moisture: float = 0.0
    sensor_max_moisture: float = 100.0
    sensor_min_ph: float = 3.5
    sensor_max_ph: float = 9.5

    # ===== LOGGING & MONITORING =====
    log_level: str = os.getenv("LOG_LEVEL", "INFO" if self.is_production else "DEBUG")
    enable_structured_logging: bool = True
    log_sensitive_data: bool = False  # Never log passwords, tokens, API keys
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN", None)
    sentry_enabled: bool = sentry_dsn is not None

    # ===== CACHING =====
    enable_cache: bool = True
    cache_backend: str = os.getenv("CACHE_BACKEND", "memory")  # memory, redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = 300

    # ===== ML SECURITY =====
    enable_ml: bool = not os.getenv("AGRISENSE_DISABLE_ML", "0") == "1"
    ml_model_verification: bool = True  # Verify model signatures
    ml_confidenc_threshold: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_production_config(self) -> list:
        """
        Validate that all required security settings are configured for production.
        Returns list of validation errors (empty if valid).
        """
        errors = []

        if self.is_production:
            if not self.jwt_secret_key or self.jwt_secret_key == "":
                errors.append("JWT_SECRET_KEY must be set in production")

            if self.debug:
                errors.append("DEBUG must be False in production")

            if not self.https_only:
                errors.append("HTTPS_ONLY must be True in production")

            if "sqlite" in self.database_url.lower():
                errors.append("SQLite database cannot be used in production")

            if self.jwt_secret_key == "change-me-in-production":
                errors.append("JWT_SECRET_KEY has default value - must be changed")

            if self.database_encryption_key is None:
                warnings.warn(
                    "DATABASE_ENCRYPTION_KEY not set - sensitive data will not be encrypted"
                )

        return errors

    def print_security_report(self) -> None:
        """Print security configuration report"""
        print("\n" + "=" * 60)
        print("🔒 AgriSense Security Configuration Report")
        print("=" * 60)

        report = {
            "Environment": self.environment,
            "Debug Mode": self.debug,
            "HTTPS Only": self.https_only,
            "Database Type": "SQLite" if "sqlite" in self.database_url else "Production DB",
            "JWT Algorithm": self.jwt_algorithm,
            "Token Expiry": f"{self.jwt_exp_minutes} minutes",
            "Rate Limiting": self.rate_limit_enabled,
            "ML Enabled": self.enable_ml,
            "Sentry Monitoring": self.sentry_enabled,
            "Structured Logging": self.enable_structured_logging,
            "CORS Enabled": len(self.cors_origins) > 0,
        }

        for key, value in report.items():
            print(f"  {key}: {value}")

        # Validate production config
        if self.is_production:
            print("\n🔐 Production Configuration Validation:")
            errors = self.validate_production_config()
            if errors:
                print("  ⚠️  Issues found:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print("  ✅ All production security checks passed")

        print("=" * 60 + "\n")
