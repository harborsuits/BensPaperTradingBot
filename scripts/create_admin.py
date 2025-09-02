#!/usr/bin/env python3
"""
Script to create an admin user for the trading bot.
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.auth.models import UserCreate
from trading_bot.auth.service import AuthService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("create_admin")

def create_admin_user(username, email, password):
    """Create an admin user with the given credentials."""
    try:
        # Create user
        user_create = UserCreate(
            username=username,
            email=email,
            password=password
        )
        
        # Check if user already exists
        existing_user = AuthService.get_user_by_username(username)
        if existing_user:
            logger.info(f"Admin user '{username}' already exists")
            return existing_user
        
        # Create new user
        user = AuthService.create_user(user_create)
        
        # Make user admin
        users = AuthService.get_users()
        users[user.id].is_admin = True
        AuthService.save_users(users)
        
        logger.info(f"Admin user '{username}' created successfully")
        return user
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        raise

def main():
    """Main entry point for the script."""
    # Get admin credentials from environment variables or use defaults
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "adminpassword")
    
    # Check if password is the default
    if admin_password == "adminpassword":
        logger.warning("Using default admin password. Consider setting ADMIN_PASSWORD environment variable.")
    
    # Create admin user
    create_admin_user(admin_username, admin_email, admin_password)

if __name__ == "__main__":
    main() 