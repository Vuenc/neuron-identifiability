#!/usr/bin/env python3
"""
Debug mode setup for full stack traces and detailed error reporting.
"""

import sys
import traceback
import warnings

def setup_debug_mode():
    """Set up comprehensive debugging and error reporting."""
    
    # Enable all warnings
    warnings.filterwarnings("default")
    
    # Set up exception hook for uncaught exceptions
    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print("\n" + "="*80)
        print("❌ UNCAUGHT EXCEPTION")
        print("="*80)
        print(f"Exception Type: {exc_type.__name__}")
        print(f"Exception Value: {exc_value}")
        print("\n🔍 FULL STACK TRACE:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("="*80)
    
    sys.excepthook = exception_hook
    
    # Enable detailed error messages
    try:
        import torch
        torch.autograd.set_detect_anomaly(True)
        print("🐛 Debug mode enabled:")
        print("  - Full stack traces for all errors")
        print("  - All warnings enabled")
        print("  - PyTorch anomaly detection enabled")
        print("  - Detailed error reporting")
    except ImportError:
        print("🐛 Debug mode enabled (PyTorch not available):")
        print("  - Full stack traces for all errors")
        print("  - All warnings enabled")
        print("  - Detailed error reporting")

if __name__ == "__main__":
    setup_debug_mode()
