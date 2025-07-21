"""
Wrapper script to run any Python script with the PyTorch fix.

Usage:
    python scripts/run_with_torch_fix.py your_script.py [args...]
"""

import sys
import os
import importlib.util
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

# Set environment variables to avoid PyTorch issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# Fix torch._classes.__path__ issue
try:
    import torch
    
    # Create a dummy __path__ attribute
    class DummyPath:
        _path = []
    
    # Replace the problematic module
    if hasattr(torch, "_classes"):
        torch._classes.__path__ = DummyPath()
        
        # Also replace __getattr__ to prevent any other issues
        original_getattr = getattr(torch._classes, "__getattr__", None)
        
        def safe_getattr(name):
            if name == "__path__":
                return DummyPath()
            if original_getattr:
                return original_getattr(name)
            raise AttributeError(f"module 'torch._classes' has no attribute '{name}'")
        
        torch._classes.__getattr__ = safe_getattr
        
    print("PyTorch patched successfully")
except ImportError:
    print("PyTorch not found, no need to patch")
except Exception as e:
    print(f"Failed to patch PyTorch: {e}")

# Fix for asyncio error
import asyncio
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except Exception as e:
    print(f"Failed to set asyncio event loop: {e}")

def run_script(script_path, args):
    """Run a Python script with the given arguments."""
    # Add the script's directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(script_path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Load the script as a module
    spec = importlib.util.spec_from_file_location("__main__", script_path)
    module = importlib.util.module_from_spec(spec)
    
    # Set sys.argv to the script and its arguments
    sys.argv = [script_path] + args
    
    # Execute the module
    spec.loader.exec_module(module)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} script.py [args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found")
        sys.exit(1)
    
    run_script(script_path, script_args)