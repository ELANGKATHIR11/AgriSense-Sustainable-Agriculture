# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.

import sys
import os

# Fix console encoding issues on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Add project root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.database.session import SessionLocalSync
from backend.database.models import User

def seed_default_users():
    db = SessionLocalSync()
    try:
        # Check if default admin exists
        admin = db.query(User).filter(User.email == "admin@agrisense.io").first()
        if not admin:
            admin = User(
                email="admin@agrisense.io",
                hashed_password="hash_admin123",
                role="admin",
                preferred_language="en"
            )
            db.add(admin)
            print("[INFO] Created default admin user: admin@agrisense.io / admin123")
        else:
            print("[INFO] Admin user already exists")

        # Check if default farmer exists
        farmer = db.query(User).filter(User.email == "farmer@agrisense.io").first()
        if not farmer:
            farmer = User(
                email="farmer@agrisense.io",
                hashed_password="hash_farmer123",
                role="farmer",
                preferred_language="en"
            )
            db.add(farmer)
            print("[INFO] Created default farmer user: farmer@agrisense.io / farmer123")
        else:
            print("[INFO] Farmer user already exists")

        db.commit()
    except Exception as e:
        print(f"[ERROR] Error seeding default users: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_default_users()
