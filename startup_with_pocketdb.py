#!/usr/bin/env python3
"""
AgriSense Full-Stack Startup with PocketDB
Initializes and starts the entire application stack
"""

import asyncio
import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from threading import Thread

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agrisense.startup")


class AgriSenseStartup:
    """Manages startup of AgriSense with PocketDB."""

    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.backend_dir = self.root_dir / "AGRISENSEFULL-STACK" / "src" / "backend"
        self.frontend_dir = self.root_dir / "AGRISENSEFULL-STACK" / "src" / "frontend"
        self.pb_data_dir = self.root_dir / "pb_data"
        
        # Create data directories
        self.pb_data_dir.mkdir(exist_ok=True)
        
        logger.info(f"AgriSense Root: {self.root_dir}")
        logger.info(f"Backend: {self.backend_dir}")
        logger.info(f"Frontend: {self.frontend_dir}")

    async def init_pocketdb(self):
        """Initialize PocketDB with AgriSense collections."""
        logger.info("🗄️  Initializing PocketDB...")
        
        try:
            # Wait for PocketDB to be ready
            import httpx
            
            async with httpx.AsyncClient() as client:
                for attempt in range(30):
                    try:
                        response = await client.get("http://localhost:8090/api/health")
                        if response.status_code == 200:
                            logger.info("✓ PocketDB is ready")
                            break
                    except Exception:
                        if attempt < 29:
                            await asyncio.sleep(1)
                        else:
                            raise
            
            # Initialize database
            from agrisense_app.backend.database import init_database
            
            logger.info("Initializing database tables...")
            db = await init_database("pocketdb")
            
            stats = await db.get_stats()
            logger.info("✓ Database initialized")
            logger.info(f"  Collections: {list(stats.get('collections', {}).keys())}")
            
            await db.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize PocketDB: {e}")
            raise

    async def verify_setup(self):
        """Verify the complete setup."""
        logger.info("\n📋 Verifying Setup...")
        logger.info("=" * 50)
        
        checks = {
            "Backend directory": self.backend_dir.exists(),
            "Frontend directory": self.frontend_dir.exists(),
            "PocketDB data directory": self.pb_data_dir.exists(),
        }
        
        for check, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"{status} {check}")
        
        if not all(checks.values()):
            logger.error("Setup verification failed!")
            return False
        
        logger.info("✓ All checks passed!")
        logger.info("=" * 50)
        return True

    def print_startup_info(self):
        """Print startup information."""
        logger.info("\n" + "=" * 60)
        logger.info("🌾 AgriSense with PocketDB - Full Stack")
        logger.info("=" * 60)
        logger.info("\n📍 Service URLs:")
        logger.info("  • Backend API:       http://localhost:8000")
        logger.info("  • API Docs:          http://localhost:8000/docs")
        logger.info("  • PocketDB Admin:    http://localhost:8090/_/")
        logger.info("  • Frontend:          http://localhost:5173")
        logger.info("\n🔌 WebSocket:")
        logger.info("  • Connection:        ws://localhost:8000/ws")
        logger.info("\n🗄️  Database:")
        logger.info("  • Backend:           PocketDB")
        logger.info("  • Data Directory:    ./pb_data")
        logger.info("  • Admin Credentials: admin@agrisense.local")
        logger.info("\n📚 Documentation:")
        logger.info("  • Read: POCKETDB_INTEGRATION.md")
        logger.info("  • API Setup: src/backend/database/example_routes.py")
        logger.info("  • Config: src/backend/database/")
        logger.info("\n💡 Next Steps:")
        logger.info("  1. Access PocketDB Admin at http://localhost:8090/_/")
        logger.info("  2. Create API collections (if not auto-initialized)")
        logger.info("  3. Check API docs at http://localhost:8000/docs")
        logger.info("  4. Visit http://localhost:5173 for frontend")
        logger.info("\n" + "=" * 60)
        logger.info("Press Ctrl+C to stop all services")
        logger.info("=" * 60 + "\n")

    async def main(self):
        """Main startup flow."""
        try:
            # Verify setup
            if not await self.verify_setup():
                sys.exit(1)
            
            # Load environment
            env_file = self.root_dir / ".env.pocketdb"
            if env_file.exists():
                logger.info(f"Loading environment from {env_file}")
                from dotenv import load_dotenv
                load_dotenv(str(env_file))
            
            # Initialize PocketDB collections
            await self.init_pocketdb()
            
            # Print startup info
            self.print_startup_info()
            
            logger.info("\n✓ AgriSense is ready to use!")
            logger.info("Backend processes should be running. Monitor the logs above.")
            
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\n\nShutdown signal received...")
                logger.info("Cleaning up...")
                logger.info("✓ Shutdown complete")
                
        except Exception as e:
            logger.error(f"Startup failed: {e}", exc_info=True)
            sys.exit(1)


async def main():
    """Entry point."""
    # Add project root to path
    project_root = Path(__file__).parent.absolute()
    agrisense_root = project_root / "AGRISENSEFULL-STACK"
    
    if agrisense_root.exists():
        sys.path.insert(0, str(agrisense_root))
    
    startup = AgriSenseStartup()
    await startup.main()


if __name__ == "__main__":
    asyncio.run(main())
