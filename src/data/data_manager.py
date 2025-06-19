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
        try:
            with sqlite3.connect(self.db_path) as conn:
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
        except Exception as e:
            logger.error(f"Error adding item {item_id}: {e}")
            return False
    
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
        try:
            # Serialize numpy array
            embedding_blob = pickle.dumps(embedding)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (item_id, embedding_vector, model_name, created_at)
                    VALUES (?, ?, ?, ?)
                """, (item_id, embedding_blob, model_name, datetime.now().isoformat()))
            return True
        except Exception as e:
            logger.error(f"Error adding embedding for {item_id}: {e}")
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
    
    def get_embeddings(self, item_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get embeddings for items
        
        Args:
            item_ids: List of item IDs, or None for all embeddings
            
        Returns:
            Dictionary mapping item_id to embedding array
        """
        query = "SELECT item_id, embedding_vector FROM embeddings"
        params = []
        
        if item_ids:
            placeholders = ",".join(["?"] * len(item_ids))
            query += f" WHERE item_id IN ({placeholders})"
            params = item_ids
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                embeddings = {}
                for item_id, embedding_blob in cursor.fetchall():
                    embeddings[item_id] = pickle.loads(embedding_blob)
                return embeddings
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return {}
    
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
    
    def import_polyvore_data(self, 
                            embeddings_file: str = "data/embeddings/polyvore/clip_embeddings.pkl",
                            polyvore_json: str = "data/datasets/polyvore_outfits.json") -> int:
        """
        Import Polyvore dataset into database
        
        Args:
            embeddings_file: Path to embeddings pickle file
            polyvore_json: Path to Polyvore outfit JSON file
            
        Returns:
            Number of items imported
        """
        try:
            # Load embeddings
            logger.info("Loading Polyvore embeddings...")
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load Polyvore metadata if available
            polyvore_metadata = {}
            if Path(polyvore_json).exists():
                logger.info("Loading Polyvore metadata...")
                with open(polyvore_json, 'r') as f:
                    polyvore_data = json.load(f)
                    # Process Polyvore structure (this depends on the actual format)
                    # For now, we'll create basic metadata from filenames
            
            imported_count = 0
            logger.info(f"Importing {len(embeddings)} Polyvore items...")
            
            for image_path, embedding in embeddings.items():
                # Extract item info from path
                path_obj = Path(image_path)
                filename = path_obj.name
                
                # Generate item_id from filename
                item_id = f"polyvore_{path_obj.stem}"
                
                # Basic category detection from filename/path
                category = self._detect_category_from_path(image_path)
                
                # Add item to database
                success = self.add_item(
                    item_id=item_id,
                    source="polyvore",
                    filename=filename,
                    category=category,
                    embedding_path=str(path_obj),
                    metadata={"original_path": image_path}
                )
                
                if success:
                    # Add embedding
                    self.add_embedding(item_id, embedding)
                    imported_count += 1
                
                if imported_count % 1000 == 0:
                    print(f"\rImported {imported_count} items...", end="", flush=True)
            
            print()  # New line
            logger.info(f"Successfully imported {imported_count} Polyvore items")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing Polyvore data: {e}")
            return 0
    
    def _detect_category_from_path(self, image_path: str) -> str:
        """
        Simple category detection from image path/filename
        
        Args:
            image_path: Path to image
            
        Returns:
            Detected category
        """
        path_lower = image_path.lower()
        
        # Simple keyword matching
        if any(word in path_lower for word in ['shirt', 'top', 'blouse', 'tee', 'sweater']):
            return 'top'
        elif any(word in path_lower for word in ['pant', 'jean', 'trouser', 'skirt', 'short']):
            return 'bottom'
        elif any(word in path_lower for word in ['shoe', 'boot', 'sneaker', 'sandal']):
            return 'shoes'
        elif any(word in path_lower for word in ['jacket', 'coat', 'cardigan']):
            return 'outerwear'
        elif any(word in path_lower for word in ['bag', 'hat', 'belt', 'accessory']):
            return 'accessory'
        else:
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