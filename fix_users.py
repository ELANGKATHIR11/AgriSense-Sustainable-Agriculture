from backend.database import SessionLocal, engine, Base
from backend.models import User
import sys

def main():
    db = SessionLocal()
    try:
        # Delete existing users
        db.query(User).delete()
        db.commit()
        print("Deleted existing users.")

        # Create new user
        new_user = User(
            email="kathir@1.io",
            hashed_password="hash_1234567890",
            role="farmer"
        )
        db.add(new_user)
        db.commit()
        print("Created new user: kathir@1.io")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
