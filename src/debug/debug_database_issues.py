#!/usr/bin/env python3
"""
Debug embedding loading issues
"""
import sqlite3
from pathlib import Path

def debug_embedding_issues(db_path="data/closetgpt.db"):
    """Debug the embedding mismatch issues"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔍 DEBUGGING EMBEDDING ISSUES")
    print("=" * 50)
    
    # 1. Check item_id mismatches
    print("\n1. Checking item_id consistency...")
    
    # Items without embeddings
    cursor.execute("""
        SELECT COUNT(*) FROM items i 
        LEFT JOIN embeddings e ON i.item_id = e.item_id 
        WHERE e.item_id IS NULL
    """)
    items_without_embeddings = cursor.fetchone()[0]
    
    # Embeddings without items
    cursor.execute("""
        SELECT COUNT(*) FROM embeddings e 
        LEFT JOIN items i ON e.item_id = i.item_id 
        WHERE i.item_id IS NULL
    """)
    embeddings_without_items = cursor.fetchone()[0]
    
    print(f"Items without embeddings: {items_without_embeddings:,}")
    print(f"Embeddings without items: {embeddings_without_items:,}")
    
    # 2. Sample problematic item IDs
    if items_without_embeddings > 0:
        print(f"\n2. Sample items without embeddings:")
        cursor.execute("""
            SELECT i.item_id FROM items i 
            LEFT JOIN embeddings e ON i.item_id = e.item_id 
            WHERE e.item_id IS NULL 
            LIMIT 10
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}")
    
    # 3. Check specific item from error
    test_item_id = "polyvore_136491077"
    print(f"\n3. Checking specific item: {test_item_id}")
    
    cursor.execute("SELECT COUNT(*) FROM items WHERE item_id = ?", (test_item_id,))
    item_exists = cursor.fetchone()[0] > 0
    
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE item_id = ?", (test_item_id,))
    embedding_exists = cursor.fetchone()[0] > 0
    
    print(f"  Item exists: {item_exists}")
    print(f"  Embedding exists: {embedding_exists}")
    
    # 4. Check for case sensitivity or whitespace issues
    cursor.execute("SELECT item_id FROM items WHERE item_id LIKE '%136491077%'")
    similar_items = cursor.fetchall()
    cursor.execute("SELECT item_id FROM embeddings WHERE item_id LIKE '%136491077%'")
    similar_embeddings = cursor.fetchall()
    
    print(f"  Similar item_ids in items: {[row[0] for row in similar_items]}")
    print(f"  Similar item_ids in embeddings: {[row[0] for row in similar_embeddings]}")
    
    # 5. Check the data_manager batch size issue
    print(f"\n4. SQL Variables Limit Analysis:")
    print(f"SQLite default max variables: 999")
    print(f"Recommendation: Batch embedding queries in chunks of 900")
    
    conn.close()

def fix_data_manager_batching():
    """Show how to fix the SQL variables issue"""
    
    print("\n🔧 FIX FOR 'TOO MANY SQL VARIABLES'")
    print("=" * 50)
    
    fix_code = '''
# In your DataManager class, modify the embedding retrieval method:

def get_embeddings_batch(self, item_ids, batch_size=900):
    """Retrieve embeddings in batches to avoid SQL variable limit"""
    embeddings = {}
    
    # Process in chunks of 900 (under 999 limit)
    for i in range(0, len(item_ids), batch_size):
        batch_ids = item_ids[i:i + batch_size]
        
        # Build query with correct number of placeholders
        placeholders = ','.join(['?'] * len(batch_ids))
        query = f"SELECT item_id, embedding_vector FROM embeddings WHERE item_id IN ({placeholders})"
        
        try:
            cursor.execute(query, batch_ids)
            for item_id, embedding_blob in cursor.fetchall():
                embeddings[item_id] = pickle.loads(embedding_blob)
        except Exception as e:
            self.logger.error(f"Error in batch {i//batch_size}: {e}")
            continue
    
    return embeddings
'''
    
    print(fix_code)

if __name__ == "__main__":
    debug_embedding_issues()
    fix_data_manager_batching()