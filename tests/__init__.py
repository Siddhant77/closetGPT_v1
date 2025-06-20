"""
Test package for ClosetGPT v1
"""

# Version info
__version__ = "1.0.0"
__author__ = "Siddhant"

# Import main test classes for easy access
from .test_clip_embedder import TestCLIPEmbedder
from .test_data_manager import TestDataManager, TestDataManagerProduction
from .test_integration import TestIntegration

__all__ = [
    'TestCLIPEmbedder',
    'TestDataManager',
    'TestDataManagerProduction', 
    'TestIntegration'
]