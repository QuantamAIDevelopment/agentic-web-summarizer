"""
Database initialization script.

This script initializes the database tables for the application.
Run this script before starting the application for the first time.
"""

import os
import sys
import logging

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import init_db
from models.agent_decisions import Base as DecisionBase
from models.agent_message import Base as MessageBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database tables."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()