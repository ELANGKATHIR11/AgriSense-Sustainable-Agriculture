# -*- coding: utf-8 -*-
"""
Database bridge file to maintain backwards compatibility with existing imports.
Redirects to the new structured backend/database module.
"""
from backend.database.connection import sync_engine as engine
from backend.database.session import SessionLocalSync as SessionLocal, get_db
from backend.database.base import Base
