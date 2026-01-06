"""
AgriSense Database Module
Multi-backend support: SQLite, PocketDB, MongoDB

Usage:
    # Initialize database
    from agrisense_app.backend.database import init_database, get_db_manager
    
    db = await init_database("pocketdb")
    
    # Use database
    readings = await db.get_readings(zone_id="field_1")
    
    # Migrate data
    from agrisense_app.backend.database import migrate_database
    result = await migrate_database("sqlite", "pocketdb")
"""

from .config import DatabaseBackend, DatabaseConfig, get_database_config
from .manager import DatabaseManager, get_database_manager, database_context
from .migration import MigrationManager, migrate_database
from .pocketdb_adapter import PocketDBAdapter, PocketDBConfig, init_pocketdb

__all__ = [
    # Config
    "DatabaseBackend",
    "DatabaseConfig",
    "get_database_config",
    # Manager
    "DatabaseManager",
    "get_database_manager",
    "database_context",
    # Migration
    "MigrationManager",
    "migrate_database",
    # PocketDB
    "PocketDBAdapter",
    "PocketDBConfig",
    "init_pocketdb",
]


async def init_database(backend: str = "sqlite", config: DatabaseConfig = None):
    """
    Initialize database with specified backend.

    Args:
        backend: Database backend ("sqlite", "pocketdb", "mongodb")
        config: Optional DatabaseConfig

    Returns:
        Initialized DatabaseManager instance

    Example:
        db = await init_database("pocketdb")
        readings = await db.get_readings()
    """
    if config is None:
        config = DatabaseConfig(backend=DatabaseBackend(backend))

    db = DatabaseManager(config)
    await db.init()
    return db
