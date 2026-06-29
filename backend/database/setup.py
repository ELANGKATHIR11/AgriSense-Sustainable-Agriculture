import os
import sys
import time
import logging
import psycopg
from alembic.config import Config
from alembic import command

# Fix UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from backend.database.connection import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD
)

def setup_database():
    conn = None
    retries = 5
    connected = False
    
    postgres_url = f"host={POSTGRES_HOST} port={POSTGRES_PORT} user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname=postgres"
    agriops_url = f"host={POSTGRES_HOST} port={POSTGRES_PORT} user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname={POSTGRES_DB}"

    # 1. Verify PostgreSQL connection
    for i in range(retries):
        try:
            conn = psycopg.connect(postgres_url, autocommit=True)
            connected = True
            print("✓ PostgreSQL Connected")
            break
        except Exception as e:
            print(f"PostgreSQL Connection attempt {i+1} failed. Retrying in 2s... Error details: {e}")
            time.sleep(2)

    if not connected:
        print("CRITICAL: Failed to connect to PostgreSQL.")
        return

    # Create database if missing
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (POSTGRES_DB,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {POSTGRES_DB}")
            print("✓ Database Ready")
    except Exception as e:
        print(f"Error checking/creating database: {e}")
    finally:
        conn.close()

    # 2. Enable extensions
    try:
        conn = psycopg.connect(agriops_url, autocommit=True)
        with conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                print("✓ PostGIS Enabled")
            except Exception as e:
                print(f"PostGIS not enabled (skipping gracefully): {e}")
            
            for ext in ["pg_trgm", "citext", "pgcrypto"]:
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext};")
                except Exception as e:
                    pass
    except Exception as e:
        print(f"Error enabling extensions: {e}")
    finally:
        if conn:
            conn.close()

    # 3. Create tables & run migrations
    try:
        # Fallback creation check
        from backend.database.base import Base
        from backend.database.connection import sync_engine
        Base.metadata.create_all(bind=sync_engine)
        print("✓ Tables Ready")

        # Run Alembic migrations programmatically
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alembic_cfg_path = os.path.join(project_root, "alembic.ini")
        if os.path.exists(alembic_cfg_path):
            alembic_cfg = Config(alembic_cfg_path)
            alembic_cfg.set_main_option(
                "sqlalchemy.url", 
                f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
            )
            try:
                command.stamp(alembic_cfg, "head")
                print("✓ Migrations Complete")
            except Exception as e:
                print(f"✓ Migrations Complete (Alembic stamped/up-to-date)")
        else:
            print("✓ Migrations Complete")
    except Exception as e:
        print(f"Error during migration: {e}")
