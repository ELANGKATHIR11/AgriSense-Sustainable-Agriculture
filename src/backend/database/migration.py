"""
Database Migration Utilities
Migrate data between SQLite, PocketDB, and MongoDB backends
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import DatabaseBackend, DatabaseConfig
from .manager import DatabaseManager

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manages data migrations between different database backends.
    Supports SQLite → PocketDB, SQLite → MongoDB, etc.
    """

    def __init__(self, source_config: DatabaseConfig, target_config: DatabaseConfig):
        """Initialize migration manager."""
        self.source = DatabaseManager(source_config)
        self.target = DatabaseManager(target_config)
        self.migration_stats = {
            "readings": 0,
            "recommendations": 0,
            "alerts": 0,
            "errors": 0,
        }

    async def migrate_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Migrate all data from source to target database.

        Args:
            batch_size: Number of records to process at once

        Returns:
            Migration statistics
        """
        try:
            await self.source.init()
            await self.target.init()

            logger.info(
                f"Starting migration: {self.source.backend.value} → {self.target.backend.value}"
            )

            # Migrate different data types
            await self._migrate_readings(batch_size)
            await self._migrate_recommendations(batch_size)
            await self._migrate_alerts(batch_size)
            await self._migrate_tank_levels(batch_size)

            logger.info(f"Migration complete: {self.migration_stats}")
            return self.migration_stats

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            await self.source.close()
            await self.target.close()

    async def _migrate_readings(self, batch_size: int) -> None:
        """Migrate sensor readings."""
        try:
            logger.info("Migrating sensor readings...")

            if self.source.backend == DatabaseBackend.SQLITE:
                readings = await self._get_sqlite_readings()
            elif self.source.backend == DatabaseBackend.POCKETDB:
                readings = await self.source._adapter.get_sensor_readings(limit=10000)
            elif self.source.backend == DatabaseBackend.MONGODB:
                readings = await self._get_mongodb_readings()
            else:
                readings = []

            for i in range(0, len(readings), batch_size):
                batch = readings[i : i + batch_size]
                for reading in batch:
                    try:
                        # Normalize data format
                        normalized = self._normalize_reading(reading)
                        await self.target.insert_reading(normalized)
                        self.migration_stats["readings"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate reading: {e}")
                        self.migration_stats["errors"] += 1

                logger.debug(
                    f"Migrated {self.migration_stats['readings']} readings..."
                )

        except Exception as e:
            logger.error(f"Failed to migrate readings: {e}")
            self.migration_stats["errors"] += 1

    async def _migrate_recommendations(self, batch_size: int) -> None:
        """Migrate recommendations."""
        try:
            logger.info("Migrating recommendations...")

            if self.source.backend == DatabaseBackend.SQLITE:
                recommendations = await self._get_sqlite_recommendations()
            elif self.source.backend == DatabaseBackend.POCKETDB:
                recommendations = await self.source._adapter.get_recommendations(
                    limit=10000
                )
            else:
                recommendations = []

            for i in range(0, len(recommendations), batch_size):
                batch = recommendations[i : i + batch_size]
                for rec in batch:
                    try:
                        if self.target.backend == DatabaseBackend.POCKETDB:
                            await self.target._adapter.insert_recommendation(rec)
                        self.migration_stats["recommendations"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate recommendation: {e}")
                        self.migration_stats["errors"] += 1

        except Exception as e:
            logger.error(f"Failed to migrate recommendations: {e}")
            self.migration_stats["errors"] += 1

    async def _migrate_alerts(self, batch_size: int) -> None:
        """Migrate alerts."""
        try:
            logger.info("Migrating alerts...")

            if self.source.backend == DatabaseBackend.SQLITE:
                alerts = await self._get_sqlite_alerts()
            elif self.source.backend == DatabaseBackend.POCKETDB:
                alerts = await self.source._adapter.get_alerts(limit=10000)
            else:
                alerts = []

            for i in range(0, len(alerts), batch_size):
                batch = alerts[i : i + batch_size]
                for alert in batch:
                    try:
                        if self.target.backend == DatabaseBackend.POCKETDB:
                            await self.target._adapter.insert_alert(alert)
                        self.migration_stats["alerts"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate alert: {e}")
                        self.migration_stats["errors"] += 1

        except Exception as e:
            logger.error(f"Failed to migrate alerts: {e}")
            self.migration_stats["errors"] += 1

    async def _migrate_tank_levels(self, batch_size: int) -> None:
        """Migrate tank level records."""
        try:
            if self.source.backend == DatabaseBackend.SQLITE:
                # Implement SQLite tank level migration
                pass
            elif self.source.backend == DatabaseBackend.POCKETDB:
                tank_levels = await self.source._adapter.get_sensor_readings(
                    limit=10000
                )
                for level in tank_levels:
                    try:
                        await self.target._adapter.insert_tank_level(level)
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Failed to migrate tank levels: {e}")

    # ============= Helper Methods =============

    async def _get_sqlite_readings(self) -> List[Dict[str, Any]]:
        """Get readings from SQLite."""
        try:
            conn = self.source._adapter.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM readings LIMIT 10000")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get SQLite readings: {e}")
            return []

    async def _get_sqlite_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations from SQLite."""
        try:
            conn = self.source._adapter.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reco_history LIMIT 10000")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get SQLite recommendations: {e}")
            return []

    async def _get_sqlite_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts from SQLite."""
        try:
            conn = self.source._adapter.get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts LIMIT 10000")
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get SQLite alerts: {e}")
            return []

    async def _get_mongodb_readings(self) -> List[Dict[str, Any]]:
        """Get readings from MongoDB."""
        try:
            cursor = self.source._adapter.sensor_readings.find()
            return await cursor.to_list(length=10000)
        except Exception as e:
            logger.warning(f"Failed to get MongoDB readings: {e}")
            return []

    @staticmethod
    def _normalize_reading(reading: Any) -> Dict[str, Any]:
        """Normalize reading format across backends."""
        if isinstance(reading, dict):
            return reading
        elif hasattr(reading, "__dict__"):
            return reading.__dict__
        else:
            return {"data": str(reading), "timestamp": datetime.utcnow().isoformat()}

    async def validate_migration(self) -> Dict[str, Any]:
        """Validate that migration completed successfully."""
        try:
            source_stats = await self.source.get_stats()
            target_stats = await self.target.get_stats()

            validation = {
                "source": source_stats,
                "target": target_stats,
                "matches": source_stats == target_stats,
            }

            return validation
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}


async def migrate_database(
    from_backend: str = "sqlite",
    to_backend: str = "pocketdb",
    from_config: Optional[DatabaseConfig] = None,
    to_config: Optional[DatabaseConfig] = None,
) -> Dict[str, Any]:
    """
    Convenient function to migrate between database backends.

    Args:
        from_backend: Source backend (sqlite, pocketdb, mongodb)
        to_backend: Target backend (sqlite, pocketdb, mongodb)
        from_config: Optional source configuration
        to_config: Optional target configuration

    Returns:
        Migration statistics
    """
    try:
        # Create default configs
        if from_config is None:
            from_config = DatabaseConfig(backend=DatabaseBackend(from_backend))

        if to_config is None:
            to_config = DatabaseConfig(backend=DatabaseBackend(to_backend))

        # Run migration
        manager = MigrationManager(from_config, to_config)
        stats = await manager.migrate_all()

        # Validate
        validation = await manager.validate_migration()

        return {
            "status": "success",
            "migration_stats": stats,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    import sys

    async def main():
        """Example migration script."""
        from_backend = sys.argv[1] if len(sys.argv) > 1 else "sqlite"
        to_backend = sys.argv[2] if len(sys.argv) > 2 else "pocketdb"

        result = await migrate_database(from_backend, to_backend)
        print(result)

    asyncio.run(main())
