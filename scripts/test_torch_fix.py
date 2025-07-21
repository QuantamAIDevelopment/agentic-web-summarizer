"""
Test script to verify that the PyTorch warning issue is fixed.

This script:
1. Imports the necessary modules
2. Attempts to create an event loop
3. Checks if the torch._classes.__path__ issue is fixed
"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the PyTorch warning disabler
from scripts.disable_torch_warnings import *

print("Testing PyTorch fix...")

# Test asyncio event loop
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("✅ Asyncio event loop created successfully")
except Exception as e:
    print(f"❌ Asyncio event loop creation failed: {e}")

# Test torch._classes.__path__ access
try:
    import torch
    path = getattr(torch._classes, "__path__", None)
    print(f"✅ torch._classes.__path__ accessed successfully: {path}")
except Exception as e:
    print(f"❌ torch._classes.__path__ access failed: {e}")

print("Test complete.")

# Clean up
try:
    loop.close()
except:
    pass