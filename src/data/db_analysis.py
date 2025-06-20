"""
Database Analysis Script. Sanity check.
Check what's actually in the ClosetGPT database
"""

import sqlite3
from pathlib import Path
import os

def analyze_database(db_path="data/closetgpt.db"):
    """Analyze the ClosetGPT database contents"""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    # Get database file size
    db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"Database file size: {db_size:.2f} MB")
    print("=" * 50)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Table sizes and row counts
        print("TABLE SIZES:")
        tables = ['items', 'embeddings', 'outfits', 'users']
        
        for table in tables:
            try:
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                print(f"  {table}: {row_count:,} rows")
                
                # Get approximate storage size for each table
                if table == 'items':
                    cursor.execute("SELECT COUNT(*), AVG(LENGTH(item_id) + LENGTH(filename) + LENGTH(COALESCE(metadata_json, ''))) FROM items")
                    count, avg_size = cursor.fetchone()
                    total_size = (avg_size or 0) * count
                    print(f"    Estimated size: {total_size/1024/1024:.1f} MB")
                elif table == 'embeddings':
                    cursor.execute("SELECT COUNT(*), AVG(LENGTH(embedding_vector)) FROM embeddings")
                    count, avg_size = cursor.fetchone()
                    total_size = (avg_size or 0) * count
                    print(f"    Estimated size: {total_size/1024/1024:.1f} MB")
                
            except Exception as e:
                print(f"  {table}: Error - {e}")
        
        print("\n" + "=" * 50)
        
        # 2. Items breakdown by source
        print("ITEMS BY SOURCE:")
        cursor.execute("SELECT source, COUNT(*) FROM items GROUP BY source ORDER BY COUNT(*) DESC")
        for source, count in cursor.fetchall():
            print(f"  {source}: {count:,} items")
        
        print("\n" + "=" * 50)
        
        # 3. Items breakdown by category
        print("ITEMS BY CATEGORY:")
        cursor.execute("SELECT category, COUNT(*) FROM items GROUP BY category ORDER BY COUNT(*) DESC")
        for category, count in cursor.fetchall():
            print(f"  {category}: {count:,} items")
        
        print("\n" + "=" * 50)
        
        # 4. Sample of actual items
        print("SAMPLE ITEMS (first 10):")
        cursor.execute("SELECT item_id, source, filename, category FROM items LIMIT 10")
        for item_id, source, filename, category in cursor.fetchall():
            print(f"  {item_id} | {source} | {filename} | {category}")
        
        print("\n" + "=" * 50)
        
        # 5. Check for embeddings
        print("EMBEDDING STATUS:")
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM items")
        total_items = cursor.fetchone()[0]
        
        print(f"  Items with embeddings: {embedding_count:,}")
        print(f"  Total items: {total_items:,}")
        print(f"  Coverage: {(embedding_count/total_items*100):.1f}%")
        
        # 6. Check actual image files
        print("\n" + "=" * 50)
        print("ACTUAL IMAGE FILES:")
        
        polyvore_dir = Path("data/images/polyvore")
        if polyvore_dir.exists():
            image_files = list(polyvore_dir.rglob("*.jpg")) + list(polyvore_dir.rglob("*.png"))
            print(f"  Polyvore image files on disk: {len(image_files):,}")
        else:
            print("  Polyvore directory not found")
        
        personal_dir = Path("data/images/personal")
        if personal_dir.exists():
            personal_files = list(personal_dir.rglob("*.jpg")) + list(personal_dir.rglob("*.png"))
            print(f"  Personal image files on disk: {len(personal_files):,}")
        else:
            print("  Personal directory not found")

def check_polyvore_structure():
    """Check the Polyvore dataset structure"""
    print("\n" + "=" * 50)
    print("POLYVORE DATASET STRUCTURE:")
    
    polyvore_dir = Path("data/images/polyvore")
    if polyvore_dir.exists():
        # Count files by subdirectory
        subdirs = [d for d in polyvore_dir.iterdir() if d.is_dir()]
        print(f"  Subdirectories: {len(subdirs)}")
        
        for subdir in sorted(subdirs)[:10]:  # Show first 10 subdirs
            file_count = len(list(subdir.rglob("*.jpg")) + list(subdir.rglob("*.png")))
            print(f"    {subdir.name}: {file_count} files")
        
        if len(subdirs) > 10:
            print(f"    ... and {len(subdirs) - 10} more subdirectories")
    
    # Check embeddings file
    embeddings_file = Path("data/embeddings/polyvore/clip_embeddings.pkl")
    if embeddings_file.exists():
        import pickle
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"  Embeddings file contains: {len(embeddings):,} items")
        
        # Show sample paths
        print("  Sample embedding paths:")
        for i, path in enumerate(list(embeddings.keys())[:5]):
            print(f"    {path}")
    else:
        print("  No embeddings file found")

if __name__ == "__main__":
    analyze_database()
    check_polyvore_structure()