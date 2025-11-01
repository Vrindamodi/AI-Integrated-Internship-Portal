#!/usr/bin/env python3
"""
Database initialization script for Smart Allocation Engine
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def create_database():
    """Create the database if it doesn't exist"""
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_0xFImLvK8sEh@ep-super-feather-adfslilb-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    
    if database_url.startswith("postgresql"):
        # For Neon/managed PostgreSQL services, the database already exists
        # We just need to verify the connection works
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                # Test connection
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                print("✅ Connected to PostgreSQL database successfully")
        except OperationalError as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            print("Please ensure PostgreSQL is running and credentials are correct")
            return False
    else:
        print("✅ Using SQLite database (no creation needed)")
    
    return True

def init_tables():
    """Initialize database tables"""
    try:
        from database import create_tables
        create_tables()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def main():
    """Main initialization function"""
    print("🚀 Initializing Smart Allocation Engine Database...")
    
    # Create database
    if not create_database():
        sys.exit(1)
    
    # Initialize tables
    if not init_tables():
        sys.exit(1)
    
    print("🎉 Database initialization completed successfully!")
    print("\nNext steps:")
    print("1. Start the API server: uvicorn app:app --reload")
    print("2. Visit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()
