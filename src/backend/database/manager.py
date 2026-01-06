"""
Database initialization and backend factory
Handles setup and switching between SQLite, PocketDB, and MongoDB
"""

import logging
from typing import Optional, Union
from contextlib import asynccontextmanager

from .config import DatabaseConfig, DatabaseBackend

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Unified database manager for switching between backends.
    Provides a consistent interface regardless of backend.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager."""
        self.config = config or DatabaseConfig.from_env()
        self.backend = self.config.backend
        self._adapter = None
        self._is_initialized = False

    async def init(self) -> None:
        """Initialize the selected database backend."""
        if self.backend == DatabaseBackend.POCKETDB:
            await self._init_pocketdb()
        elif self.backend == DatabaseBackend.SQLITE:
            await self._init_sqlite()
        elif self.backend == DatabaseBackend.MONGODB:
            await self._init_mongodb()
        else:
            raise ValueError(f"Unsupported database backend: {self.backend}")

        self._is_initialized = True
        logger.info(f"Database initialized: {self.backend.value}")

    async def close(self) -> None:
        """Close database connection."""
        if self._adapter and hasattr(self._adapter, "disconnect"):
            await self._adapter.disconnect()
        self._is_initialized = False
        logger.info("Database connection closed")

    async def _init_pocketdb(self) -> None:
        """Initialize PocketDB adapter."""
        try:
            from .pocketdb_adapter import init_pocketdb, PocketDBConfig

            pocketdb_config = PocketDBConfig(
                base_url=self.config.pocketdb_url,
                data_dir=self.config.pocketdb_data_dir,
                auto_init=self.config.auto_init,
            )

            self._adapter = await init_pocketdb(pocketdb_config)
            logger.info(
                f"PocketDB initialized at {self.config.pocketdb_url}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize PocketDB: {e}")
            raise

    async def _init_sqlite(self) -> None:
        """Initialize SQLite backend (legacy support)."""
        try:
            # Existing SQLite backend via core.data_store
            from ..core import data_store

            logger.info(f"SQLite database ready at {data_store.DB_PATH}")
            self._adapter = data_store
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise

    async def _init_mongodb(self) -> None:
        """Initialize MongoDB adapter."""
        try:
            from motor.motor_asyncio import AsyncClient

            client = AsyncClient(self.config.mongodb_url)
            db = client.agrisense
            self._adapter = db
            logger.info(f"MongoDB initialized at {self.config.mongodb_url}")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise

    def get_adapter(self):
        """Get the underlying database adapter."""
        if not self._is_initialized:
            raise RuntimeError(
                "Database not initialized. Call await db.init() first."
            )
        return self._adapter

    # ============= Convenience Methods =============

    async def insert_reading(self, data: dict) -> dict:
        """Insert a sensor reading."""
        if self.backend == DatabaseBackend.POCKETDB:
            return await self._adapter.insert_sensor_reading(data)
        elif self.backend == DatabaseBackend.SQLITE:
            # Use legacy SQLite interface
            conn = self._adapter.get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO readings (ts, zone_id, plant, soil_type, area_m2, 
                    ph, moisture_pct, temperature_c, ec_dS_m, n_ppm, p_ppm, k_ppm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("ts"),
                    data.get("zone_id"),
                    data.get("plant"),
                    data.get("soil_type"),
                    data.get("area_m2"),
                    data.get("ph"),
                    data.get("moisture_pct"),
                    data.get("temperature_c"),
                    data.get("ec_dS_m"),
                    data.get("n_ppm"),
                    data.get("p_ppm"),
                    data.get("k_ppm"),
                ),
            )
            conn.commit()
            return data
        elif self.backend == DatabaseBackend.MONGODB:
            result = await self._adapter.sensor_readings.insert_one(data)
            return {**data, "id": str(result.inserted_id)}

    async def get_readings(self, zone_id: Optional[str] = None, limit: int = 100):
        """Get sensor readings."""
        if self.backend == DatabaseBackend.POCKETDB:
            return await self._adapter.get_sensor_readings(zone_id, limit)
        elif self.backend == DatabaseBackend.SQLITE:
            conn = self._adapter.get_conn()
            cursor = conn.cursor()
            if zone_id:
                cursor.execute(
                    "SELECT * FROM readings WHERE zone_id = ? LIMIT ?",
                    (zone_id, limit),
                )
            else:
                cursor.execute("SELECT * FROM readings LIMIT ?", (limit,))
            return cursor.fetchall()
        elif self.backend == DatabaseBackend.MONGODB:
            cursor = self._adapter.sensor_readings.find()
            if zone_id:
                cursor = cursor.find({"zone_id": zone_id})
            return await cursor.limit(limit).to_list(length=None)

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if self.backend == DatabaseBackend.POCKETDB:
                return await self._adapter.health_check()
            elif self.backend == DatabaseBackend.SQLITE:
                conn = self._adapter.get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
            elif self.backend == DatabaseBackend.MONGODB:
                await self._adapter.client.admin.command("ping")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            if self.backend == DatabaseBackend.POCKETDB:
                return await self._adapter.get_stats()
            elif self.backend == DatabaseBackend.SQLITE:
                conn = self._adapter.get_conn()
                cursor = conn.cursor()
                stats = {}
                for table in [
                    "readings",
                    "reco_history",
                    "alerts",
                    "tank_levels",
                ]:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[table] = {"record_count": count}
                return stats
            elif self.backend == DatabaseBackend.MONGODB:
                stats = {}
                for collection_name in [
                    "sensor_readings",
                    "recommendations",
                    "alerts",
                ]:
                    count = await self._adapter[collection_name].count_documents({})
                    stats[collection_name] = {"record_count": count}
                return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Global instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get or create database manager instance."""
    global _db_manager

    if _db_manager is None:
        config = config or DatabaseConfig.from_env()
        _db_manager = DatabaseManager(config)

    return _db_manager


@asynccontextmanager
async def database_context(config: Optional[DatabaseConfig] = None):
    """Async context manager for database operations."""
    db = get_database_manager(config)
    await db.init()
    try:
        yield db
    finally:
        await db.close()
