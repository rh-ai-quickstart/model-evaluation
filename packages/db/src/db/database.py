
"""
Database configuration and utilities
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import text


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://user:changeme@localhost:5432/model-evaluation"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=os.environ.get("DB_ECHO", "false").lower() == "true",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

logger = logging.getLogger(__name__)

# Capture service startup time
SERVICE_START_TIME = datetime.now(timezone.utc)


class DatabaseService:
    """Database service for handling database operations"""
    
    def __init__(self, engine=None):
        self.engine = engine or globals()['engine']
        self.start_time = SERVICE_START_TIME
    
    async def health_check(self) -> dict[str, Any]:
        """
        Perform database health check
        
        Returns:
            Dict containing health status information
        """
        try:
            # Test basic connectivity
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as health_check"))
                result.scalar()
            
            # Test if we can create a session
            async with SessionLocal() as session:
                await session.execute(text("SELECT version()"))
            
            return {
                "name": "Database",
                "status": "healthy",
                "message": "PostgreSQL connection successful",
                "version": "0.0.0",
                "start_time": self.start_time.isoformat()
            }
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return {
                "name": "Database", 
                "status": "down",
                "message": f"PostgreSQL connection failed: {str(e)[:100]}",
                "version": "0.0.0",
                "start_time": self.start_time.isoformat()
            }
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        return SessionLocal()


# Global database service instance
db_service = DatabaseService()


async def get_db():
    """Dependency to get database session"""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_db_service() -> DatabaseService:
    """Dependency to get database service"""
    return db_service
