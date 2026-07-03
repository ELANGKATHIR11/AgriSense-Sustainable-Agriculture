from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from .connection import sync_engine, async_engine

# Sync and Async Session factories
SessionLocalSync = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
SessionLocalAsync = async_sessionmaker(
    bind=async_engine, expire_on_commit=False, class_=AsyncSession
)


# Dependency Injection helper for synchronous operations
def get_db():
    db = SessionLocalSync()
    try:
        yield db
    finally:
        db.close()


# Dependency Injection helper for asynchronous operations
async def get_async_db():
    async with SessionLocalAsync() as session:
        try:
            yield session
        finally:
            await session.close()
