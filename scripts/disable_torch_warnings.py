"""
Monkey patch to disable PyTorch warnings about torch.classes.

This script should be imported before any other imports in the main application.
"""

import warnings
import sys

# Disable PyTorch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch._classes")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*torch._classes.*")

# Monkey patch torch._classes.__getattr__ to prevent the warning
try:
    import torch
    
    # Store the original __getattr__
    if hasattr(torch._classes, "__getattr__"):
        original_getattr = torch._classes.__getattr__
        
        # Define a new __getattr__ that catches the specific error
        def safe_getattr(name):
            if name == "__path__":
                return []
            return original_getattr(name)
        
        # Replace the original __getattr__
        torch._classes.__getattr__ = safe_getattr
        
    print("PyTorch warnings disabled successfully")
except ImportError:
    print("PyTorch not found, no need to disable warnings")
except Exception as e:
    print(f"Failed to disable PyTorch warnings: {e}")