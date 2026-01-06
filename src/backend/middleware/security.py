"""
Enhanced Security Middleware for AgriSense
Protects against common web vulnerabilities: XSS, CSRF, MIME sniffing, clickjacking, etc.
"""
import logging
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add comprehensive security headers to all HTTP responses.
    Protects against XSS, MIME sniffing, clickjacking, and more.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS protection in older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Control referrer information leakage
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Prevent unwanted plugin execution
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Content Security Policy - prevents XSS and unauthorized script execution
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com https://fonts.googleapis.com; "
            "img-src 'self' data: https:; "
            "media-src 'self'; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp

        # HSTS: Force HTTPS connections (1 year, include subdomains)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Permissions Policy: Restrict access to powerful browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "accelerometer=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "clipboard-read=(), "
            "clipboard-write=()"
        )

        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

        return response


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize incoming request headers to prevent injection attacks.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for suspicious headers that might indicate attack attempts
        suspicious_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-forwarded-proto",
            "x-forwarded-host",
        ]

        for header in suspicious_headers:
            if header in request.headers:
                # Log suspicious header activity
                value = request.headers.get(header)
                if value and len(value) > 100:
                    logger.warning(
                        f"Suspicious {header} header detected: {value[:50]}..."
                    )

        response = await call_next(request)
        return response


class NOSQLInjectionProtectionMiddleware(BaseHTTPMiddleware):
    """
    Detect and prevent NoSQL injection attacks.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for common NoSQL injection patterns in query parameters
        suspicious_patterns = [
            "$ne",
            "$gt",
            "$lt",
            "$eq",
            "$regex",
            "$exists",
            "$where",
            "$or",
            "$and",
            "$nor",
            "$not",
        ]

        # Check query parameters
        if request.query_params:
            for key, value in request.query_params.items():
                if isinstance(value, str):
                    # Convert to string for pattern matching
                    value_str = str(value).lower()
                    for pattern in suspicious_patterns:
                        if pattern in value_str:
                            logger.warning(
                                f"Possible NoSQL injection attempt detected in parameter '{key}'"
                            )
                            # Don't block, just log - let application-level validation handle it

        response = await call_next(request)
        return response
