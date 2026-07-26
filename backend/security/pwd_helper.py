# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.

# -*- coding: utf-8 -*-
import bcrypt

# Ensure passlib compatibility with bcrypt 4.x
if not hasattr(bcrypt, "__about__"):
    bcrypt.__about__ = type("about", (), {"__version__": getattr(bcrypt, "__version__", "4.0.0")})()

_orig_hashpw = bcrypt.hashpw
def _safe_hashpw(password, salt):
    if len(password) > 72:
        password = password[:72]
    return _orig_hashpw(password, salt)
bcrypt.hashpw = _safe_hashpw

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password or not (
        hashed_password.startswith("$2b$") or hashed_password.startswith("$2a$")
    ):
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except (ValueError, TypeError):
        return False
