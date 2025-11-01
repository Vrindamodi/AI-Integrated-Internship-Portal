#!/usr/bin/env python3
"""
Initialize database with default admin user
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import create_tables, get_db
from models import User, UserRole
from auth import create_user, get_password_hash
from sqlalchemy.orm import Session

def create_default_admin():
    """Create default admin user if it doesn't exist"""
    db = next(get_db())
    
    # Check if admin user already exists
    admin_user = db.query(User).filter(User.username == "admin").first()
    if admin_user:
        print("Admin user already exists")
        return admin_user
    
    # Create default admin user
    admin_user = create_user(
        db=db,
        username="admin",
        email="admin@smartallocation.com",
        password="admin123",  # Change this in production!
        role=UserRole.ADMIN
    )
    
    print(f"Created admin user: {admin_user.username}")
    return admin_user

def create_default_user():
    """Create default regular user if it doesn't exist"""
    db = next(get_db())
    
    # Check if user already exists
    regular_user = db.query(User).filter(User.username == "user").first()
    if regular_user:
        print("Regular user already exists")
        return regular_user
    
    # Create default regular user
    regular_user = create_user(
        db=db,
        username="user",
        email="user@smartallocation.com",
        password="user123",  # Change this in production!
        role=UserRole.USER
    )
    
    print(f"Created regular user: {regular_user.username}")
    return regular_user

if __name__ == "__main__":
    print("Initializing database with default users...")
    
    # Create tables
    create_tables()
    print("Database tables created")
    
    # Create default users
    create_default_admin()
    create_default_user()
    
    print("\nDefault users created:")
    print("Admin: username='admin', password='admin123'")
    print("User: username='user', password='user123'")
    print("\nIMPORTANT: Change these passwords in production!")
