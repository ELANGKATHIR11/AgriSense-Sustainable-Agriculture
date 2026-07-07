# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# -*- coding: utf-8 -*-
"""
Database package initializer.
"""

from backend.database.base import Base
from backend.database.connection import sync_engine as engine, async_engine
from backend.database.session import (
    SessionLocalSync as SessionLocal,
    get_db,
    get_async_db,
)
