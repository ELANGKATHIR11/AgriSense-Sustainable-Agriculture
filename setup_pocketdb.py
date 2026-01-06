#!/usr/bin/env python
"""
PocketDB Setup Script for AgriSense
Handles initialization, migration, and configuration of PocketDB backend
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional


async def main():
    """Main setup flow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AgriSense PocketDB Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_pocketdb.py --mode init                    # Initialize PocketDB
  python setup_pocketdb.py --mode migrate --from sqlite   # Migrate from SQLite
  python setup_pocketdb.py --mode health-check            # Check database health
  python setup_pocketdb.py --mode stats                   # Get database statistics
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["init", "migrate", "health-check", "stats", "cleanup"],
        default="init",
        help="Operation mode",
    )

    parser.add_argument(
        "--from",
        dest="from_backend",
        default="sqlite",
        help="Source backend for migration (default: sqlite)",
    )

    parser.add_argument(
        "--to",
        dest="to_backend",
        default="pocketdb",
        help="Target backend for migration (default: pocketdb)",
    )

    parser.add_argument(
        "--pocketdb-url",
        default=os.getenv("POCKETDB_URL", "http://localhost:8090"),
        help="PocketDB server URL",
    )

    parser.add_argument(
        "--pocketdb-data-dir",
        default=os.getenv("POCKETDB_DATA_DIR", "./pb_data"),
        help="PocketDB data directory",
    )

    parser.add_argument(
        "--days-to-keep",
        type=int,
        default=90,
        help="Days of data to keep during cleanup",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger("agrisense.setup")

    # Import database modules
    try:
        from agrisense_app.backend.database import (
            init_database,
            migrate_database,
            DatabaseConfig,
            DatabaseBackend,
        )
    except ImportError as e:
        print(f"Error: Could not import database modules: {e}")
        print("Make sure you're running from the project root with proper PYTHONPATH")
        sys.exit(1)

    try:
        if args.mode == "init":
            await cmd_init(args, logger)
        elif args.mode == "migrate":
            await cmd_migrate(args, logger)
        elif args.mode == "health-check":
            await cmd_health_check(args, logger)
        elif args.mode == "stats":
            await cmd_stats(args, logger)
        elif args.mode == "cleanup":
            await cmd_cleanup(args, logger)

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)


async def cmd_init(args, logger):
    """Initialize PocketDB."""
    from agrisense_app.backend.database import init_database, PocketDBConfig

    logger.info("Initializing PocketDB...")
    logger.info(f"URL: {args.pocketdb_url}")
    logger.info(f"Data Dir: {args.pocketdb_data_dir}")

    # Create data directory
    os.makedirs(args.pocketdb_data_dir, exist_ok=True)

    try:
        db = await init_database("pocketdb")
        logger.info("✓ PocketDB initialized successfully")

        # Check health
        health = await db.health_check()
        if health:
            logger.info("✓ Database health check passed")
        else:
            logger.warning("⚠ Database health check failed")

        # Get stats
        stats = await db.get_stats()
        logger.info(f"Database statistics: {stats}")

        await db.close()
        logger.info("✓ Setup complete")

    except Exception as e:
        logger.error(f"Failed to initialize PocketDB: {e}")
        raise


async def cmd_migrate(args, logger):
    """Migrate data from one backend to another."""
    from agrisense_app.backend.database import migrate_database

    logger.info(f"Starting migration: {args.from_backend} → {args.to_backend}")

    try:
        result = await migrate_database(args.from_backend, args.to_backend)

        if result["status"] == "success":
            stats = result.get("migration_stats", {})
            logger.info(f"✓ Migration successful")
            logger.info(f"  - Readings: {stats.get('readings', 0)}")
            logger.info(f"  - Recommendations: {stats.get('recommendations', 0)}")
            logger.info(f"  - Alerts: {stats.get('alerts', 0)}")
            logger.info(f"  - Errors: {stats.get('errors', 0)}")

            # Validation
            validation = result.get("validation", {})
            if validation.get("matches"):
                logger.info("✓ Data validation passed")
            else:
                logger.warning("⚠ Data validation has differences")

        else:
            logger.error(f"✗ Migration failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration error: {e}")
        raise


async def cmd_health_check(args, logger):
    """Check database health."""
    from agrisense_app.backend.database import init_database

    logger.info("Checking database health...")

    try:
        db = await init_database("pocketdb")
        health = await db.health_check()

        if health:
            logger.info("✓ Database is healthy")
        else:
            logger.error("✗ Database is unhealthy")
            sys.exit(1)

        await db.close()

    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        sys.exit(1)


async def cmd_stats(args, logger):
    """Get database statistics."""
    from agrisense_app.backend.database import init_database

    logger.info("Fetching database statistics...")

    try:
        db = await init_database("pocketdb")
        stats = await db.get_stats()

        logger.info("Database Statistics:")
        logger.info(f"  Timestamp: {stats.get('timestamp', 'N/A')}")
        logger.info(f"  Connected: {stats.get('connected', 'N/A')}")
        logger.info("\n  Collections:")

        for collection, info in stats.get("collections", {}).items():
            count = info.get("record_count", 0)
            logger.info(f"    - {collection}: {count} records")

        await db.close()

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        sys.exit(1)


async def cmd_cleanup(args, logger):
    """Clean up old records."""
    from agrisense_app.backend.database import init_database

    logger.info(f"Cleaning up records older than {args.days_to_keep} days...")

    try:
        db = await init_database("pocketdb")

        collections = ["sensor_readings", "recommendations", "alerts"]

        total_deleted = 0
        for collection in collections:
            if hasattr(db._adapter, "clear_old_records"):
                deleted = await db._adapter.clear_old_records(
                    collection, args.days_to_keep
                )
                logger.info(f"  - {collection}: {deleted} deleted")
                total_deleted += deleted

        logger.info(f"✓ Total records deleted: {total_deleted}")
        await db.close()

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
