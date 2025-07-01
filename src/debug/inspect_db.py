#!/usr/bin/env python3
"""
Quick database inspection script for closetgpt.db
"""
import sqlite3
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Adjust path to import DataManager

from src.data.data_manager import DataManager


def inspect_database(db_path="data/closetgpt.db"):
    """Inspect Polyvore database structure and content"""
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("📊 DATABASE INSPECTION REPORT")
    print("=" * 50)
    
    # 1. Database file size
    file_size = Path(db_path).stat().st_size
    print(f"💾 File Size: {file_size / (1024*1024):.2f} MB")
    
    # 2. Show all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\n📋 Tables ({len(tables)}):")
    for table in tables:
        print(f"  • {table[0]}")
    
    # 3. Schema for each table
    print(f"\n🏗️  SCHEMA:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        print(f"\n  {table_name}:")
        for col in columns:
            col_id, name, dtype, not_null, default, pk = col
            pk_str = " (PK)" if pk else ""
            null_str = " NOT NULL" if not_null else ""
            print(f"    {name}: {dtype}{null_str}{pk_str}")
    
    # 4. Record counts
    print(f"\n📈 RECORD COUNTS:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  {table_name}: {count:,} records")
    
    # 5. Sample data from main tables
    main_tables = ['items', 'polyvore_items', 'embeddings']
    for table_name in main_tables:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            rows = cursor.fetchall()
            if rows:
                print(f"\n🔍 SAMPLE DATA - {table_name}:")
                for i, row in enumerate(rows[:2]):
                    print(f"  Row {i+1}: {str(row)[:100]}...")
        except sqlite3.OperationalError:
            pass  # Table doesn't exist
    
    # 6. Check embedding dimensions
    try:
        cursor.execute("SELECT embedding FROM embeddings LIMIT 1;")
        result = cursor.fetchone()
        if result and result[0]:
            # Embeddings might be stored as TEXT (JSON) or BLOB
            embedding_data = result[0]
            if isinstance(embedding_data, str):
                # JSON format
                import json
                embedding = json.loads(embedding_data)
                if isinstance(embedding, list):
                    print(f"\n🧠 EMBEDDING INFO:")
                    print(f"  Format: JSON array")
                    print(f"  Dimensions: {len(embedding)}")
                    print(f"  Sample values: {embedding[:5]}...")
            elif isinstance(embedding_data, (bytes, memoryview)):
                # Binary format
                embedding_array = np.frombuffer(embedding_data, dtype=np.float32)
                print(f"\n🧠 EMBEDDING INFO:")
                print(f"  Format: Binary (numpy)")
                print(f"  Dimensions: {len(embedding_array)}")
                print(f"  Sample values: {embedding_array[:5]}")
    except Exception as e:
        print(f"\n🧠 EMBEDDING INFO: Could not inspect embeddings - {e}")
    
    # 7. Standard CLIP embedding size verification
    print(f"\n📏 CLIP STANDARD SIZES:")
    print(f"  CLIP ViT-B/32: 512 dimensions")
    print(f"  CLIP ViT-L/14: 768 dimensions")
    
    conn.close()

def quick_stats(db_path="data/closetgpt.db"):
    """Quick one-liner stats"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count items
    try:
        cursor.execute("SELECT COUNT(*) FROM items;")
        item_count = cursor.fetchone()[0]
    except:
        item_count = 0
    
    # Count embeddings
    try:
        cursor.execute("SELECT COUNT(*) FROM embeddings;")
        embedding_count = cursor.fetchone()[0]
    except:
        embedding_count = 0
    
    print(f"📊 Quick Stats: {item_count:,} items, {embedding_count:,} embeddings")
    conn.close()

if __name__ == "__main__":
    # Run inspection
    inspect_database()
    
    # Also provide SQL commands for manual inspection
    print(f"\n🔧 MANUAL SQL COMMANDS:")
    print(f"sqlite3 data/closetgpt.db")
    print(f".tables")
    print(f".schema items")
    print(f"SELECT COUNT(*) FROM items;")
    print(f"SELECT * FROM items LIMIT 5;")

    # cleanup database
    # Run once to clean database
    print(f"\n🧹 CLEANING UP DUPLICATE EMBEDDINGS:")
    manager = DataManager("data/closetgpt.db")
    manager.cleanup_duplicate_embeddings()