"""
Data Manager for ClosetGPT v1
Handles SQLite database operations and data management
"""

import sqlite3
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, db_path: str = "data/closetgpt.db"):
        """
        Initialize data manager with SQLite database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"DataManager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Items table for both personal and dataset items
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT UNIQUE NOT NULL,  -- e.g., 'personal_0001', 'polyvore_12345'
                    source TEXT NOT NULL,          -- 'personal', 'polyvore', 'scraped'
                    filename TEXT NOT NULL,
                    category TEXT NOT NULL,        -- 'top', 'bottom', 'shoes', 'outerwear', 'accessory'
                    color TEXT,
                    style TEXT,
                    formality TEXT DEFAULT 'casual', -- 'casual', 'smart-casual', 'formal'
                    season TEXT DEFAULT 'all',      -- 'warm', 'cold', 'all'
                    embedding_path TEXT,            -- Path to embedding file
                    metadata_json TEXT,             -- Additional metadata as JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Outfits table for storing outfit combinations
                CREATE TABLE IF NOT EXISTS outfits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    outfit_id TEXT UNIQUE NOT NULL,
                    item_ids TEXT NOT NULL,         -- JSON array of item IDs
                    compatibility_score REAL,
                    context_json TEXT,              -- Weather, occasion context as JSON
                    source TEXT DEFAULT 'generated', -- 'manual', 'generated', 'polyvore'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Users table for future multi-user support
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    preferences_json TEXT,          -- User preferences as JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Embeddings table for efficient similarity searches
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    embedding_vector BLOB,          -- Pickled numpy array
                    model_name TEXT DEFAULT 'clip-vit-base-patch32',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (item_id) REFERENCES items (item_id)
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_items_category ON items (category);
                CREATE INDEX IF NOT EXISTS idx_items_source ON items (source);
                CREATE INDEX IF NOT EXISTS idx_items_color ON items (color);
                CREATE INDEX IF NOT EXISTS idx_items_formality ON items (formality);
                CREATE INDEX IF NOT EXISTS idx_outfits_score ON outfits (compatibility_score);
                CREATE INDEX IF NOT EXISTS idx_embeddings_item_id ON embeddings (item_id);
            """)
        
        logger.info("Database tables initialized")

    #
    # CRUD helper functions for 'items' object/table
    # 
    
    def add_item(self, 
                 item_id: str,
                 source: str,
                 filename: str,
                 category: str,
                 color: Optional[str] = None,
                 style: Optional[str] = None,
                 formality: str = "casual",
                 season: str = "all",
                 embedding_path: Optional[str] = None,
                 metadata: Optional[Dict] = None) -> bool:
        """
        Add a clothing item to the database
        
        Args:
            item_id: Unique identifier for the item
            source: Data source ('personal', 'polyvore', 'scraped')
            filename: Image filename
            category: Clothing category
            color: Primary color
            style: Style description
            formality: Formality level
            season: Season appropriateness
            embedding_path: Path to embedding file
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO items 
                        (item_id, source, filename, category, color, style, formality, season, 
                         embedding_path, metadata_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        item_id, source, filename, category, color, style, formality, season,
                        embedding_path, json.dumps(metadata) if metadata else None,
                        datetime.now().isoformat()
                    ))
                return True
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Error adding item {item_id}: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error adding item {item_id}: {e}")
                return False
            
    def get_items(self, 
                  source: Optional[str] = None,
                  category: Optional[str] = None,
                  color: Optional[str] = None,
                  formality: Optional[str] = None,
                  season: Optional[str] = None) -> List[Dict]:
        """
        Retrieve items with optional filtering
        
        Args:
            source: Filter by data source
            category: Filter by category
            color: Filter by color
            formality: Filter by formality
            season: Filter by season
            
        Returns:
            List of item dictionaries
        """
        query = "SELECT * FROM items WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        if category:
            query += " AND category = ?"
            params.append(category)
        if color:
            query += " AND color = ?"
            params.append(color)
        if formality:
            query += " AND formality = ?"
            params.append(formality)
        if season:
            query += " AND (season = ? OR season = 'all')"
            params.append(season)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                items = []
                for row in cursor.fetchall():
                    item = dict(row)
                    # Parse JSON metadata
                    if item['metadata_json']:
                        item['metadata'] = json.loads(item['metadata_json'])
                    else:
                        item['metadata'] = {}
                    del item['metadata_json']
                    items.append(item)
                return items
        except Exception as e:
            logger.error(f"Error retrieving items: {e}")
            return []

    def update_item(self, 
                    item_id: str,
                    **kwargs) -> bool:
        """
        Update an existing item's metadata
        
        Args:
            item_id: Item identifier
            **kwargs: Fields to update (category, color, style, formality, season, metadata, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not kwargs:
            logger.warning(f"No fields provided for updating item {item_id}")
            return False
        
        # Build dynamic update query
        valid_fields = {'source', 'filename', 'category', 'color', 'style', 'formality', 'season', 'embedding_path', 'metadata'}
        update_fields = []
        params = []
        
        for field, value in kwargs.items():
            if field in valid_fields:
                if field == 'metadata':
                    update_fields.append("metadata_json = ?")
                    params.append(json.dumps(value) if value else None)
                else:
                    update_fields.append(f"{field} = ?")
                    params.append(value)
        
        if not update_fields:
            logger.warning(f"No valid fields provided for updating item {item_id}")
            return False
        
        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(item_id)  # For WHERE clause
        
        query = f"UPDATE items SET {', '.join(update_fields)} WHERE item_id = ?"
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute(query, params)
                if cursor.rowcount > 0:
                    logger.info(f"Successfully updated item {item_id}")
                    return True
                else:
                    logger.warning(f"Item {item_id} not found for update")
                    return False
        except Exception as e:
            logger.error(f"Error updating item {item_id}: {e}")
            return False
    
    def delete_item(self, item_id: str, delete_embedding: bool = True) -> bool:
        """
        Delete an item and optionally its embedding
        
        Args:
            item_id: Item identifier
            delete_embedding: Whether to also delete the embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Check if item exists
                cursor.execute("SELECT COUNT(*) FROM items WHERE item_id = ?", (item_id,))
                if cursor.fetchone()[0] == 0:
                    logger.warning(f"Item {item_id} not found for deletion")
                    return False
                
                # Delete embedding first (if requested)
                if delete_embedding:
                    cursor.execute("DELETE FROM embeddings WHERE item_id = ?", (item_id,))
                    embedding_deleted = cursor.rowcount
                    logger.info(f"Deleted {embedding_deleted} embedding(s) for item {item_id}")
                
                # Delete from outfits (remove item from any outfit combinations)
                cursor.execute("SELECT outfit_id, item_ids FROM outfits")
                outfits_to_update = []
                outfits_to_delete = []
                
                for outfit_id, item_ids_json in cursor.fetchall():
                    item_ids = json.loads(item_ids_json)
                    if item_id in item_ids:
                        item_ids.remove(item_id)
                        if len(item_ids) < 2:  # Outfit needs at least 2 items
                            outfits_to_delete.append(outfit_id)
                        else:
                            outfits_to_update.append((json.dumps(item_ids), outfit_id))
                
                # Update/delete affected outfits
                for outfit_id in outfits_to_delete:
                    cursor.execute("DELETE FROM outfits WHERE outfit_id = ?", (outfit_id,))
                    logger.info(f"Deleted outfit {outfit_id} (insufficient items after removing {item_id})")
                
                for item_ids_json, outfit_id in outfits_to_update:
                    cursor.execute("UPDATE outfits SET item_ids = ? WHERE outfit_id = ?", (item_ids_json, outfit_id))
                    logger.info(f"Updated outfit {outfit_id} after removing item {item_id}")
                
                # Delete the item
                cursor.execute("DELETE FROM items WHERE item_id = ?", (item_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Successfully deleted item {item_id}")
                    return True
                else:
                    logger.error(f"Failed to delete item {item_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            return False
        
    #
    # CRUD helper functions for 'embeddings' object/table
    # 

    def add_embedding(self, item_id: str, embedding: np.ndarray, model_name: str = "clip-vit-base-patch32") -> bool:
        """
        Add or update embedding for an item
        
        Args:
            item_id: Item identifier
            embedding: Numpy array embedding
            model_name: Name of the model used
            
        Returns:
            True if successful, False otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Serialize numpy array
                embedding_blob = pickle.dumps(embedding)
                
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO embeddings 
                        (item_id, embedding_vector, model_name, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (item_id, embedding_blob, model_name, datetime.now().isoformat()))
                return True
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Error adding embedding for {item_id}: {e}")
                    return False
            except Exception as e:
                logger.error(f"Error adding embedding for {item_id}: {e}")
                return False
    
    def get_embeddings_batch(self, item_ids: Optional[List[str]] = None, batch_size=900) -> Dict[str, np.ndarray]:
        """
        Get embeddings for items in batches. batch size is 900 to avoid SQLite's 999 variable limit.
        
        Args:
            item_ids: List of item IDs, or None for all embeddings
            
        Returns:
            Dictionary mapping item_id to embedding array
        """

        embeddings = {}
        
        if not item_ids:
            return embeddings
        
        # Process in chunks to avoid SQL variable limit
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i + batch_size]
            
            # Build query with correct number of placeholders
            placeholders = ','.join(['?'] * len(batch_ids))
            
            # Order by id DESC to get most recent embedding for duplicates
            query = f"""
                SELECT item_id, embedding_vector 
                FROM embeddings 
                WHERE item_id IN ({placeholders})
                ORDER BY id DESC
            """
            
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    cursor = conn.execute(query, batch_ids)
                    
                    for item_id, embedding_blob in cursor.fetchall():
                        # Only store first occurrence (most recent due to ORDER BY)
                        if item_id not in embeddings:
                            try:
                                import pickle
                                embeddings[item_id] = pickle.loads(embedding_blob)
                            except Exception as e:
                                logger.warning(f"Failed to deserialize embedding for {item_id}: {e}")
                            
            except Exception as e:
                logger.error(f"Error retrieving embeddings batch {i//batch_size}: {e}")
                continue
        
        return embeddings
    
    # def get_embeddings(self, item_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    #     """
    #     Get embeddings for items
        
    #     Args:
    #         item_ids: List of item IDs, or None for all embeddings
            
    #     Returns:
    #         Dictionary mapping item_id to embedding array
    #     """
    #     query = "SELECT item_id, embedding_vector FROM embeddings"
    #     params = []
        
    #     if item_ids:
    #         placeholders = ",".join(["?"] * len(item_ids))
    #         query += f" WHERE item_id IN ({placeholders})"
    #         params = item_ids
        
    #     try:
    #         with sqlite3.connect(self.db_path) as conn:
    #             cursor = conn.execute(query, params)
    #             embeddings = {}
    #             for item_id, embedding_blob in cursor.fetchall():
    #                 embeddings[item_id] = pickle.loads(embedding_blob)
    #             return embeddings
    #     except Exception as e:
    #         logger.error(f"Error retrieving embeddings: {e}")
    #         return {}


    def get_embeddings(self, item_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple items (batched)"""
        if isinstance(item_ids, str):
            item_ids = [item_ids]
        
        return self.get_embeddings_batch(item_ids)

    def cleanup_duplicate_embeddings(self) -> None:
        """
        Remove duplicate embeddings (keep most recent)
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                query = """
                        SELECT item_id, COUNT(*) as count 
                        FROM embeddings 
                        GROUP BY item_id 
                        HAVING COUNT(*) > 1
                        """
                cursor = conn.execute(query)
                
                duplicates = cursor.fetchall()
                logger.info(f"Found {len(duplicates)} items with duplicate embeddings")
            
                for item_id, count in duplicates:
                    # Keep only the most recent (highest id)
                    cursor.execute("""
                        DELETE FROM embeddings 
                        WHERE item_id = ? AND id NOT IN (
                            SELECT MAX(id) FROM embeddings WHERE item_id = ?
                        )
                    """, (item_id, item_id))
                
                conn.commit()
                logger.info(f"Cleaned up duplicate embeddings")
            
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {e}")
            conn.rollback()
        
    def update_embedding(self, item_id: str, embedding: np.ndarray, model_name: str = "clip-vit-base-patch32") -> bool:
        """
        Update an existing embedding (same as add_embedding with OR REPLACE)
        
        Args:
            item_id: Item identifier
            embedding: New embedding vector
            model_name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        success = self.add_embedding(item_id, embedding, model_name)
        if success:
            logger.info(f"Successfully updated embedding for item {item_id}")
        return success
    
    def delete_embedding(self, item_id: str) -> bool:
        """
        Delete an embedding for an item
        
        Args:
            item_id: Item identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("DELETE FROM embeddings WHERE item_id = ?", (item_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Successfully deleted embedding for item {item_id}")
                    return True
                else:
                    logger.warning(f"No embedding found for item {item_id}")
                    return False
        except Exception as e:
            logger.error(f"Error deleting embedding for {item_id}: {e}")
            return False

        
    def get_item_with_embedding(self, item_id: str) -> Optional[Dict]:
        """
        Get item data along with its embedding
        
        Args:
            item_id: Item identifier
            
        Returns:
            Dictionary with item data and embedding, or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT i.*, e.embedding_vector, e.model_name
                    FROM items i
                    LEFT JOIN embeddings e ON i.item_id = e.item_id
                    WHERE i.item_id = ?
                """, (item_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                item = dict(row)
                
                # Parse JSON metadata
                if item['metadata_json']:
                    item['metadata'] = json.loads(item['metadata_json'])
                else:
                    item['metadata'] = {}
                del item['metadata_json']
                
                # Deserialize embedding
                if item['embedding_vector']:
                    item['embedding'] = pickle.loads(item['embedding_vector'])
                else:
                    item['embedding'] = None
                del item['embedding_vector']
                
                return item
        except Exception as e:
            logger.error(f"Error retrieving item {item_id}: {e}")
            return None
        
    #
    # CRUD helper functions for 'outfits' object/table
    # 

    def add_outfit(self, 
                   outfit_id: str,
                   item_ids: List[str],
                   compatibility_score: Optional[float] = None,
                   context: Optional[Dict] = None,
                   source: str = "generated") -> bool:
        """
        Add an outfit combination
        
        Args:
            outfit_id: Unique outfit identifier
            item_ids: List of item IDs in the outfit
            compatibility_score: Compatibility score (0-1)
            context: Context information (weather, occasion, etc.)
            source: Source of the outfit
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO outfits 
                    (outfit_id, item_ids, compatibility_score, context_json, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    outfit_id, 
                    json.dumps(item_ids),
                    compatibility_score,
                    json.dumps(context) if context else None,
                    source,
                    datetime.now().isoformat()
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding outfit {outfit_id}: {e}")
            return False
    
    def get_outfits(self, min_score: Optional[float] = None) -> List[Dict]:
        """
        Retrieve outfits with optional score filtering
        
        Args:
            min_score: Minimum compatibility score
            
        Returns:
            List of outfit dictionaries
        """
        query = "SELECT * FROM outfits WHERE 1=1"
        params = []
        
        if min_score is not None:
            query += " AND compatibility_score >= ?"
            params.append(min_score)
        
        query += " ORDER BY compatibility_score DESC"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                outfits = []
                for row in cursor.fetchall():
                    outfit = dict(row)
                    # Parse JSON fields
                    outfit['item_ids'] = json.loads(outfit['item_ids'])
                    if outfit['context_json']:
                        outfit['context'] = json.loads(outfit['context_json'])
                    else:
                        outfit['context'] = {}
                    del outfit['context_json']
                    outfits.append(outfit)
                return outfits
        except Exception as e:
            logger.error(f"Error retrieving outfits: {e}")
            return []

    def update_outfit(self, 
                      outfit_id: str,
                      item_ids: Optional[List[str]] = None,
                      compatibility_score: Optional[float] = None,
                      context: Optional[Dict] = None) -> bool:
        """
        Update an existing outfit
        
        Args:
            outfit_id: Outfit identifier
            item_ids: New list of item IDs (optional)
            compatibility_score: New compatibility score (optional)
            context: New context information (optional)
            
        Returns:
            True if successful, False otherwise
        """
        update_fields = []
        params = []
        
        if item_ids is not None:
            update_fields.append("item_ids = ?")
            params.append(json.dumps(item_ids))
        
        if compatibility_score is not None:
            update_fields.append("compatibility_score = ?")
            params.append(compatibility_score)
        
        if context is not None:
            update_fields.append("context_json = ?")
            params.append(json.dumps(context))
        
        if not update_fields:
            logger.warning(f"No fields provided for updating outfit {outfit_id}")
            return False
        
        query = f"UPDATE outfits SET {', '.join(update_fields)} WHERE outfit_id = ?"
        params.append(outfit_id)
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute(query, params)
                if cursor.rowcount > 0:
                    logger.info(f"Successfully updated outfit {outfit_id}")
                    return True
                else:
                    logger.warning(f"Outfit {outfit_id} not found for update")
                    return False
        except Exception as e:
            logger.error(f"Error updating outfit {outfit_id}: {e}")
            return False
    
    def delete_outfit(self, outfit_id: str) -> bool:
        """
        Delete an outfit
        
        Args:
            outfit_id: Outfit identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("DELETE FROM outfits WHERE outfit_id = ?", (outfit_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Successfully deleted outfit {outfit_id}")
                    return True
                else:
                    logger.warning(f"Outfit {outfit_id} not found for deletion")
                    return False
        except Exception as e:
            logger.error(f"Error deleting outfit {outfit_id}: {e}")
            return False
        
    #
    # CRUD helper functions for 'users' object/table
    # 

    def add_user(self, user_id: str, name: Optional[str] = None, preferences: Optional[Dict] = None) -> bool:
        """
        Add a new user
        
        Args:
            user_id: Unique user identifier
            name: User's name
            preferences: User preferences as dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO users (user_id, name, preferences_json, created_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, name, json.dumps(preferences) if preferences else None, datetime.now().isoformat()))
                logger.info(f"Successfully added user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding user {user_id}: {e}")
            return False
        
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user information
        
        Args:
            user_id: User identifier
            
        Returns:
            User dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    user = dict(row)
                    if user['preferences_json']:
                        user['preferences'] = json.loads(user['preferences_json'])
                    else:
                        user['preferences'] = {}
                    del user['preferences_json']
                    return user
                return None
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None

    def update_user(self, user_id: str, name: Optional[str] = None, preferences: Optional[Dict] = None) -> bool:
        """
        Update user information
        
        Args:
            user_id: User identifier
            name: New name (optional)
            preferences: New preferences (optional)
            
        Returns:
            True if successful, False otherwise
        """
        update_fields = []
        params = []
        
        if name is not None:
            update_fields.append("name = ?")
            params.append(name)
        
        if preferences is not None:
            update_fields.append("preferences_json = ?")
            params.append(json.dumps(preferences))
        
        if not update_fields:
            logger.warning(f"No fields provided for updating user {user_id}")
            return False
        
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE user_id = ?"
        params.append(user_id)
        
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute(query, params)
                if cursor.rowcount > 0:
                    logger.info(f"Successfully updated user {user_id}")
                    return True
                else:
                    logger.warning(f"User {user_id} not found for update")
                    return False
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Successfully deleted user {user_id}")
                    return True
                else:
                    logger.warning(f"User {user_id} not found for deletion")
                    return False
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            return False
        
    #
    # Database statistics and import functions
    #
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Item counts by source
                cursor = conn.execute("SELECT source, COUNT(*) FROM items GROUP BY source")
                stats['items_by_source'] = dict(cursor.fetchall())
                
                # Item counts by category
                cursor = conn.execute("SELECT category, COUNT(*) FROM items GROUP BY category")
                stats['items_by_category'] = dict(cursor.fetchall())
                
                # Total counts
                cursor = conn.execute("SELECT COUNT(*) FROM items")
                stats['total_items'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                stats['total_embeddings'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM outfits")
                stats['total_outfits'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
        
    #
    # Build mock database using Polyvore dataset
    # 
    
    def import_polyvore_data(self, 
                            embeddings_file: str = "data/embeddings/polyvore/clip_embeddings.pkl",
                            polyvore_json: str = "data/datasets/polyvore_outfits.json",
                            chunk_size: int = 5000) -> int:
        """
        Import Polyvore dataset into database using chunked processing
        
        Args:
            embeddings_file: Path to embeddings pickle file
            polyvore_json: Path to Polyvore outfit JSON file
            chunk_size: Number of items to process in each chunk
            
        Returns:
            Number of items imported
        """
        try:
            # Load embeddings
            logger.info("Loading Polyvore embeddings...")
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            total_items = len(embeddings)
            logger.info(f"Found {total_items:,} Polyvore items to import")
            
            # Process in chunks to avoid resource exhaustion
            imported_count = 0
            failed_count = 0
            
            # Convert to list for chunking
            embedding_items = list(embeddings.items())
            
            for chunk_start in range(0, total_items, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_items)
                chunk_items = embedding_items[chunk_start:chunk_end]
                
                logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_items-1)//chunk_size + 1} "
                           f"(items {chunk_start+1}-{chunk_end})")
                
                # Process this chunk
                chunk_imported, chunk_failed = self._process_polyvore_chunk(chunk_items)
                imported_count += chunk_imported
                failed_count += chunk_failed
                
                print(f"Chunk complete: {chunk_imported}/{len(chunk_items)} successful")
                print(f"Total progress: {imported_count:,}/{total_items:,} ({imported_count/total_items*100:.1f}%)")
                
                # Force garbage collection between chunks
                import gc
                gc.collect()
            
            logger.info(f"Import complete: {imported_count:,} successful, {failed_count:,} failed")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing Polyvore data: {e}")
            return 0
    
    #
    # Private helper functions for processing Polyvore data in a chunk
    #
    def _process_polyvore_chunk(self, chunk_items: List[Tuple[str, np.ndarray]]) -> Tuple[int, int]:
        """
        Process a chunk of Polyvore items
        
        Args:
            chunk_items: List of (image_path, embedding) tuples
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        items_data = []
        embeddings_data = []
        
        # Prepare data for batch insert
        for image_path, embedding in chunk_items:
            path_obj = Path(image_path)
            filename = path_obj.name
            item_id = f"polyvore_{path_obj.stem}"
            category = self._detect_category_from_path(image_path)
            
            items_data.append((
                item_id, "polyvore", filename, category, None, None,
                "casual", "all", str(path_obj),
                json.dumps({"original_path": image_path}),
                datetime.now().isoformat()
            ))
            
            embeddings_data.append((
                item_id, pickle.dumps(embedding), "clip-vit-base-patch32",
                datetime.now().isoformat()
            ))
        
        # Insert with retry logic
        return self._insert_chunk_with_retry(items_data, embeddings_data)
    
    #
    # Private helper function for inserting a chunk with robust retry logic
    #
    def _insert_chunk_with_retry(self, items_data: List, embeddings_data: List) -> Tuple[int, int]:
        """
        Insert a chunk with robust retry logic
        
        Args:
            items_data: List of item tuples
            embeddings_data: List of embedding tuples
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        max_retries = 5
        successful = 0
        failed = 0
        
        for attempt in range(max_retries):
            try:
                # Use a single transaction for the entire chunk
                with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                    # Set WAL mode for better concurrency
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    
                    # Begin transaction
                    conn.execute("BEGIN TRANSACTION")
                    
                    try:
                        # Insert items
                        conn.executemany("""
                            INSERT OR IGNORE INTO items 
                            (item_id, source, filename, category, color, style, formality, season, 
                             embedding_path, metadata_json, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, items_data)
                        
                        # Insert embeddings
                        conn.executemany("""
                            INSERT OR IGNORE INTO embeddings 
                            (item_id, embedding_vector, model_name, created_at)
                            VALUES (?, ?, ?, ?)
                        """, embeddings_data)
                        
                        # Commit transaction
                        conn.execute("COMMIT")
                        successful = len(items_data)
                        break
                        
                    except Exception as e:
                        conn.execute("ROLLBACK")
                        raise e
                        
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Database busy, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to insert chunk after {max_retries} attempts: {e}")
                    failed = len(items_data)
                    break
            except Exception as e:
                logger.error(f"Unexpected error inserting chunk: {e}")
                failed = len(items_data)
                break
        
        return successful, failed
    

    #
    # Private helper function for detecting category from image path. 
    # (Not used as image path does not contain category info.)
    #            
    def _detect_category_from_path(self, image_path: str) -> str:
        """
        Simple category detection from image path/filename
        
        Args:
            image_path: Path to image
            
        Returns:
            Detected category
        """
        path_lower = image_path.lower()
        
        # For Polyvore dataset, filenames are just numbers, so we can't detect category
        # We'll need to use the Polyvore metadata JSON if available, or leave as 'unknown'
        # This will be improved when we add the outfit compatibility model
        
        # Simple keyword matching for personal photos and other datasets
        if any(word in path_lower for word in ['shirt', 'top', 'blouse', 'tee', 'sweater', 'tank']):
            return 'top'
        elif any(word in path_lower for word in ['pant', 'jean', 'trouser', 'skirt', 'short', 'bottom']):
            return 'bottom'
        elif any(word in path_lower for word in ['shoe', 'boot', 'sneaker', 'sandal', 'heel']):
            return 'shoes'
        elif any(word in path_lower for word in ['jacket', 'coat', 'cardigan', 'blazer']):
            return 'outerwear'
        elif any(word in path_lower for word in ['bag', 'purse', 'hat', 'belt', 'accessory', 'jewelry', 'watch']):
            return 'accessory'
        else:
            # For Polyvore numeric filenames, we'll categorize as 'unknown' for now
            # This can be improved later with ML-based category detection
            return 'unknown'

def setup_database_with_polyvore():
    """Convenience function to set up database with Polyvore data"""
    manager = DataManager()
    imported = manager.import_polyvore_data()
    stats = manager.get_stats()
    
    print("Database setup complete!")
    print(f"Imported {imported} items")
    print("Stats:", stats)
    
    return manager

if __name__ == "__main__":
    # Set up database with Polyvore data
    manager = setup_database_with_polyvore()