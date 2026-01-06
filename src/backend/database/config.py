"""
PocketDB Configuration Module
Manages database backend selection and initialization.
"""

import os
import logging
from enum import Enum
from typing import Optional, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DatabaseBackend(str, Enum):
    """Supported database backends."""

    SQLITE = "sqlite"
    POCKETDB = "pocketdb"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """Generic database configuration."""

    backend: DatabaseBackend = DatabaseBackend.SQLITE
    sqlite_path: Optional[str] = None
    pocketdb_url: Optional[str] = None
    pocketdb_data_dir: Optional[str] = None
    mongodb_url: Optional[str] = None
    auto_init: bool = True
    enable_migrations: bool = True

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        backend_str = os.getenv("AGRISENSE_DB_BACKEND", "sqlite").lower()

        try:
            backend = DatabaseBackend(backend_str)
        except ValueError:
            logger.warning(
                f"Unknown database backend '{backend_str}', defaulting to sqlite"
            )
            backend = DatabaseBackend.SQLITE

        return cls(
            backend=backend,
            sqlite_path=os.getenv("AGRISENSE_DB_PATH"),
            pocketdb_url=os.getenv("POCKETDB_URL", "http://localhost:8090"),
            pocketdb_data_dir=os.getenv("POCKETDB_DATA_DIR"),
            mongodb_url=os.getenv("MONGODB_URL"),
            auto_init=os.getenv("AGRISENSE_DB_AUTO_INIT", "1").lower() in ("1", "true"),
            enable_migrations=os.getenv("AGRISENSE_DB_MIGRATIONS", "1").lower()
            in ("1", "true"),
        )


def get_database_config() -> DatabaseConfig:
    """Get database configuration from environment."""
    return DatabaseConfig.from_env()
