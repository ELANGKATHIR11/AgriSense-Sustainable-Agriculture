"""
PocketDB Adapter for AgriSense
Provides a lightweight, embedded database backend for sensor data and recommendations.

PocketDB is ideal for:
- Edge devices (Pi, edge servers)
- Offline-first applications
- Lightweight IoT deployments
- Local caching with sync capabilities
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# PocketBase Client (Python-friendly PocketDB interface)
try:
    import pocketbase
    from pocketbase.client import Client
    POCKETDB_AVAILABLE = True
except ImportError:
    POCKETDB_AVAILABLE = False
    logger.warning("PocketBase client not installed. Install with: pip install pocketbase-client")


class PocketDBConfig:
    """Configuration for PocketDB connection."""

    def __init__(
        self,
        base_url: str = "http://localhost:8090",
        data_dir: Optional[str] = None,
        admin_email: str = "admin@agrisense.local",
        admin_password: str = "AgriSense@2024!",
        auto_init: bool = True,
    ):
        """
        Initialize PocketDB configuration.

        Args:
            base_url: PocketDB server URL
            data_dir: Directory for database files (default: ./pb_data)
            admin_email: Admin email for authentication
            admin_password: Admin password
            auto_init: Automatically initialize database and collections
        """
        self.base_url = base_url
        self.data_dir = data_dir or os.getenv("POCKETDB_DATA_DIR", "./pb_data")
        self.admin_email = os.getenv("POCKETDB_ADMIN_EMAIL", admin_email)
        self.admin_password = os.getenv("POCKETDB_ADMIN_PASSWORD", admin_password)
        self.auto_init = auto_init

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)


class PocketDBAdapter:
    """
    PocketDB adapter for AgriSense.
    Provides async-friendly interface for sensor data persistence.
    """

    def __init__(self, config: PocketDBConfig):
        """Initialize PocketDB adapter."""
        if not POCKETDB_AVAILABLE:
            raise ImportError(
                "PocketBase client not installed. "
                "Install with: pip install pocketbase-client"
            )

        self.config = config
        self.client: Optional[Client] = None
        self._is_initialized = False

    async def connect(self) -> None:
        """Establish connection to PocketDB."""
        try:
            self.client = Client(self.config.base_url)
            logger.info(f"Connected to PocketDB at {self.config.base_url}")

            # Authenticate as admin
            await self._authenticate()

            if self.config.auto_init:
                await self._initialize_collections()

            self._is_initialized = True
        except Exception as e:
            logger.error(f"Failed to connect to PocketDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Close PocketDB connection."""
        if self.client:
            self.client = None
            logger.info("Disconnected from PocketDB")

    async def _authenticate(self) -> None:
        """Authenticate with PocketDB admin user."""
        try:
            # Try to authenticate with existing admin user
            auth_data = await self._run_async(
                lambda: self.client.admins.auth_with_password(
                    self.config.admin_email, self.config.admin_password
                )
            )
            logger.debug("Authenticated with PocketDB")
        except Exception as e:
            logger.warning(f"Admin authentication failed: {e}")

    async def _initialize_collections(self) -> None:
        """Initialize required collections."""
        collections = [
            {
                "name": "sensor_readings",
                "schema": {
                    "ts": {"type": "datetime"},
                    "zone_id": {"type": "text"},
                    "device_id": {"type": "text"},
                    "plant": {"type": "text"},
                    "soil_type": {"type": "text"},
                    "area_m2": {"type": "number"},
                    "ph": {"type": "number"},
                    "moisture_pct": {"type": "number"},
                    "temperature_c": {"type": "number"},
                    "ec_dS_m": {"type": "number"},
                    "n_ppm": {"type": "number"},
                    "p_ppm": {"type": "number"},
                    "k_ppm": {"type": "number"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "recommendations",
                "schema": {
                    "ts": {"type": "datetime"},
                    "zone_id": {"type": "text"},
                    "plant": {"type": "text"},
                    "water_liters": {"type": "number"},
                    "expected_savings_liters": {"type": "number"},
                    "fert_n_g": {"type": "number"},
                    "fert_p_g": {"type": "number"},
                    "fert_k_g": {"type": "number"},
                    "yield_potential": {"type": "number"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "recommendation_tips",
                "schema": {
                    "ts": {"type": "datetime"},
                    "zone_id": {"type": "text"},
                    "plant": {"type": "text"},
                    "tip": {"type": "text"},
                    "category": {"type": "text"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "tank_levels",
                "schema": {
                    "ts": {"type": "datetime"},
                    "tank_id": {"type": "text"},
                    "level_pct": {"type": "number"},
                    "volume_l": {"type": "number"},
                    "rainfall_mm": {"type": "number"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "rainwater_harvest",
                "schema": {
                    "ts": {"type": "datetime"},
                    "tank_id": {"type": "text"},
                    "collected_liters": {"type": "number"},
                    "used_liters": {"type": "number"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "valve_events",
                "schema": {
                    "ts": {"type": "datetime"},
                    "zone_id": {"type": "text"},
                    "action": {"type": "text"},
                    "duration_s": {"type": "number"},
                    "status": {"type": "text"},
                    "timestamp": {"type": "datetime"},
                },
            },
            {
                "name": "alerts",
                "schema": {
                    "ts": {"type": "datetime"},
                    "zone_id": {"type": "text"},
                    "category": {"type": "text"},
                    "message": {"type": "text"},
                    "sent": {"type": "bool"},
                    "timestamp": {"type": "datetime"},
                },
            },
        ]

        for collection in collections:
            try:
                await self._run_async(
                    lambda c=collection: self.client.collections.get_one(c["name"])
                )
            except Exception:
                # Collection doesn't exist, create it
                try:
                    await self._run_async(
                        lambda c=collection: self.client.collections.create(c)
                    )
                    logger.info(f"Created collection: {collection['name']}")
                except Exception as e:
                    logger.warning(f"Failed to create collection {collection['name']}: {e}")

    # ============= Data Operations =============

    async def insert_sensor_reading(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a sensor reading."""
        try:
            record = await self._run_async(
                lambda: self.client.collection("sensor_readings").create(data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to insert sensor reading: {e}")
            raise

    async def insert_recommendation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a recommendation record."""
        try:
            record = await self._run_async(
                lambda: self.client.collection("recommendations").create(data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to insert recommendation: {e}")
            raise

    async def insert_recommendation_tips(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert recommendation tips."""
        try:
            record = await self._run_async(
                lambda: self.client.collection("recommendation_tips").create(data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to insert recommendation tip: {e}")
            raise

    async def insert_tank_level(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a tank level record."""
        try:
            record = await self._run_async(
                lambda: self.client.collection("tank_levels").create(data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to insert tank level: {e}")
            raise

    async def insert_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert an alert record."""
        try:
            record = await self._run_async(
                lambda: self.client.collection("alerts").create(data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to insert alert: {e}")
            raise

    async def get_sensor_readings(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get sensor readings, optionally filtered by zone."""
        try:
            filter_str = f'zone_id = "{zone_id}"' if zone_id else ""
            records = await self._run_async(
                lambda: self.client.collection("sensor_readings").get_list(
                    1, limit, {"filter": filter_str} if filter_str else {}
                )
            )
            return records.items if hasattr(records, "items") else records
        except Exception as e:
            logger.error(f"Failed to get sensor readings: {e}")
            return []

    async def get_recommendations(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recommendations, optionally filtered by zone."""
        try:
            filter_str = f'zone_id = "{zone_id}"' if zone_id else ""
            records = await self._run_async(
                lambda: self.client.collection("recommendations").get_list(
                    1, limit, {"filter": filter_str} if filter_str else {}
                )
            )
            return records.items if hasattr(records, "items") else records
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []

    async def get_alerts(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by zone."""
        try:
            filter_str = f'zone_id = "{zone_id}"' if zone_id else ""
            records = await self._run_async(
                lambda: self.client.collection("alerts").get_list(
                    1, limit, {"filter": filter_str} if filter_str else {}
                )
            )
            return records.items if hasattr(records, "items") else records
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    async def update_record(
        self, collection: str, record_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a record in any collection."""
        try:
            record = await self._run_async(
                lambda: self.client.collection(collection).update(record_id, data)
            )
            return record
        except Exception as e:
            logger.error(f"Failed to update record in {collection}: {e}")
            raise

    async def delete_record(self, collection: str, record_id: str) -> None:
        """Delete a record from any collection."""
        try:
            await self._run_async(
                lambda: self.client.collection(collection).delete(record_id)
            )
        except Exception as e:
            logger.error(f"Failed to delete record from {collection}: {e}")
            raise

    async def clear_old_records(
        self, collection: str, days_to_keep: int = 90
    ) -> int:
        """Delete records older than specified days."""
        try:
            from datetime import datetime, timedelta

            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
            filter_str = f'timestamp < "{cutoff_date}"'

            records = await self._run_async(
                lambda: self.client.collection(collection).get_list(
                    1, 10000, {"filter": filter_str}
                )
            )

            deleted_count = 0
            if hasattr(records, "items"):
                for record in records.items:
                    await self.delete_record(collection, record.id)
                    deleted_count += 1
            else:
                for record in records:
                    await self.delete_record(collection, record["id"])
                    deleted_count += 1

            logger.info(f"Cleared {deleted_count} old records from {collection}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to clear old records from {collection}: {e}")
            return 0

    # ============= Utility Methods =============

    @staticmethod
    async def _run_async(func):
        """Run a blocking function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)

    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            if not self.client:
                return False

            # Try to get a collection list
            await self._run_async(lambda: self.client.collections.get_list())
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "connected": self._is_initialized,
                "collections": {},
            }

            for collection_name in [
                "sensor_readings",
                "recommendations",
                "alerts",
                "tank_levels",
            ]:
                try:
                    result = await self._run_async(
                        lambda cn=collection_name: self.client.collection(cn).get_list(
                            1, 1
                        )
                    )
                    count = result.total_items if hasattr(result, "total_items") else 0
                    stats["collections"][collection_name] = {"record_count": count}
                except Exception:
                    stats["collections"][collection_name] = {"record_count": 0}

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Global adapter instance
_pocketdb_adapter: Optional[PocketDBAdapter] = None


def get_pocketdb_adapter(config: Optional[PocketDBConfig] = None) -> PocketDBAdapter:
    """Get or create PocketDB adapter instance."""
    global _pocketdb_adapter

    if _pocketdb_adapter is None:
        config = config or PocketDBConfig()
        _pocketdb_adapter = PocketDBAdapter(config)

    return _pocketdb_adapter


async def init_pocketdb(config: Optional[PocketDBConfig] = None) -> PocketDBAdapter:
    """Initialize PocketDB adapter with connection."""
    adapter = get_pocketdb_adapter(config)
    if not adapter._is_initialized:
        await adapter.connect()
    return adapter


async def cleanup_pocketdb() -> None:
    """Cleanup PocketDB connection."""
    global _pocketdb_adapter
    if _pocketdb_adapter:
        await _pocketdb_adapter.disconnect()
        _pocketdb_adapter = None
