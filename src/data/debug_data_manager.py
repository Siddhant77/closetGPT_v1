"""
Debug script for DataManager issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.data_manager import DataManager
import traceback

def test_basic_functionality():
    """Test basic DataManager functionality"""
    try:
        # Initialize manager
        manager = DataManager("debug_test.db")
        print("✅ DataManager initialized")
        
        # Test adding a simple item
        success = manager.add_item(
            item_id="debug_001",
            source="debug",
            filename="debug.jpg",
            category="top"
        )
        print(f"✅ Add item: {success}")
        
        # Test getting stats
        stats = manager.get_stats()
        print(f"✅ Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

def test_chunk_processing():
    """Test the chunk processing method specifically"""
    try:
        manager = DataManager("debug_test.db")
        
        # Create fake chunk data
        import numpy as np
        fake_embedding = np.random.rand(512).astype(np.float32)
        chunk_items = [
            ("data/images/polyvore/123.jpg", fake_embedding),
            ("data/images/polyvore/456.jpg", fake_embedding),
        ]
        
        print("Testing _process_polyvore_chunk...")
        result = manager._process_polyvore_chunk(chunk_items)
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        
        if result is not None:
            successful, failed = result
            print(f"✅ Chunk processing: {successful} successful, {failed} failed")
        else:
            print("❌ Chunk processing returned None")
            
    except Exception as e:
        print(f"❌ Error in chunk processing: {e}")
        traceback.print_exc()

def test_insert_chunk():
    """Test the insert chunk method"""
    try:
        manager = DataManager("debug_test.db")
        
        # Test data
        import numpy as np
        import pickle
        from datetime import datetime
        import json
        
        fake_embedding = np.random.rand(512).astype(np.float32)
        
        items_data = [
            ("debug_chunk_001", "debug", "test1.jpg", "top", None, None,
             "casual", "all", "/path/test1.jpg", 
             json.dumps({"test": True}), datetime.now().isoformat())
        ]
        
        embeddings_data = [
            ("debug_chunk_001", pickle.dumps(fake_embedding), "test-model", datetime.now().isoformat())
        ]
        
        print("Testing _insert_chunk_with_retry...")
        result = manager._insert_chunk_with_retry(items_data, embeddings_data)
        print(f"Insert result: {result}")
        print(f"Insert result type: {type(result)}")

        if result is not None:
            successful, failed = result
            print(f"✅ Insert Test: {successful} successful, {failed} failed")
        else:
            print("❌ Insert Test returned None")
        
    except Exception as e:
        print(f"❌ Error in insert chunk: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 DEBUGGING DATA MANAGER")
    print("=" * 40)
    
    print("\n1. Testing basic functionality...")
    test_basic_functionality()
    
    print("\n2. Testing chunk processing...")
    test_chunk_processing()
    
    print("\n3. Testing insert chunk...")
    test_insert_chunk()
    
    print("\n🔍 Debug complete")