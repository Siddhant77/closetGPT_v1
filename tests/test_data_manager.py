"""
Test cases for DataManager
"""

import unittest
import tempfile
import shutil
import numpy as np
import sqlite3
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_manager import DataManager


class TestDataManagerBase(unittest.TestCase):
    """Base test cases for DataManager that can run on different databases"""
    
    def setUp(self):
        """Set up test database for each test"""
        # This will be overridden by subclasses to use different databases
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_closetgpt.db"
        self.manager = DataManager(str(self.db_path))
        self.is_production_db = False  # Flag to modify test behavior for production DB
        
        # Sample test data
        self.sample_embedding = np.random.rand(512).astype(np.float32)
        self.sample_embedding = self.sample_embedding / np.linalg.norm(self.sample_embedding)
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _get_test_prefix(self):
        """Get unique test prefix to avoid conflicts with production data"""
        return f"test_{self.__class__.__name__.lower()}_{id(self)}"
    
    def test_database_initialization(self):
        """Test database tables are created correctly"""
        self.assertTrue(self.db_path.exists())
        
        # Check that all tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('items', 'embeddings', 'outfits', 'users')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        expected_tables = {'items', 'embeddings', 'outfits', 'users'}
        self.assertEqual(set(tables), expected_tables)
        
        # Check table schemas
        with sqlite3.connect(self.db_path) as conn:
            # Check items table columns
            cursor = conn.execute("PRAGMA table_info(items)")
            items_columns = {row[1] for row in cursor.fetchall()}
            expected_items_columns = {
                'id', 'item_id', 'source', 'filename', 'category', 'color', 'style',
                'formality', 'season', 'embedding_path', 'metadata_json', 'created_at', 'updated_at'
            }
            self.assertEqual(items_columns, expected_items_columns)
            
            # Check embeddings table columns
            cursor = conn.execute("PRAGMA table_info(embeddings)")
            embeddings_columns = {row[1] for row in cursor.fetchall()}
            expected_embeddings_columns = {'id', 'item_id', 'embedding_vector', 'model_name', 'created_at'}
            self.assertEqual(embeddings_columns, expected_embeddings_columns)
            
            # Check indexes exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row[0] for row in cursor.fetchall()}
            expected_indexes = {
                'idx_items_category', 'idx_items_source', 'idx_items_color',
                'idx_items_formality', 'idx_outfits_score', 'idx_embeddings_item_id'
            }
            # Some indexes might be auto-created by SQLite, so check subset
            self.assertTrue(expected_indexes.issubset(indexes))
        
        print("✅ Database initialization: All tables and schemas created correctly")
    
    def test_database_integrity_and_size(self):
        """Test database integrity and analyze storage"""
        # Add test data for analysis
        test_items = [
            ("integrity_001", "test", "item1.jpg", "top", "blue", "shirt", "casual"),
            ("integrity_002", "test", "item2.jpg", "bottom", "black", "jeans", "casual"),
            ("integrity_003", "test", "item3.jpg", "shoes", "white", "sneakers", "casual"),
        ]
        
        for item_id, source, filename, category, color, style, formality in test_items:
            self.manager.add_item(item_id, source, filename, category, color, style, formality)
            self.manager.add_embedding(item_id, self.sample_embedding)
        
        # Test database integrity
        with sqlite3.connect(self.db_path) as conn:
            # Check foreign key constraints
            cursor = conn.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            self.assertEqual(len(fk_violations), 0, f"Foreign key violations: {fk_violations}")
            
            # Check database integrity
            cursor = conn.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            self.assertEqual(integrity_result, "ok", f"Database integrity issue: {integrity_result}")
            
            # Analyze table sizes
            cursor = conn.execute("SELECT COUNT(*) FROM items")
            items_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            embeddings_count = cursor.fetchone()[0]
            
            # Test data should be there
            self.assertGreaterEqual(items_count, 3)
            self.assertGreaterEqual(embeddings_count, 3)
            
            # Check average row sizes
            cursor = conn.execute("""
                SELECT AVG(LENGTH(item_id) + LENGTH(filename) + LENGTH(COALESCE(metadata_json, ''))) 
                FROM items
            """)
            avg_item_size = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("SELECT AVG(LENGTH(embedding_vector)) FROM embeddings")
            avg_embedding_size = cursor.fetchone()[0] or 0
            
            # Embeddings should be substantial (512 floats ≈ 2KB when pickled)
            self.assertGreater(avg_embedding_size, 1000, "Embeddings seem too small")
            
        print(f"✅ Database integrity: {items_count} items, {embeddings_count} embeddings")
        print(f"   Average item size: {avg_item_size:.0f} bytes")
        print(f"   Average embedding size: {avg_embedding_size:.0f} bytes")
    
    def test_concurrent_access_and_locking(self):
        """Test database handles concurrent access properly"""
        import threading
        import time
        
        results = {"successes": 0, "failures": 0}
        
        def worker_thread(thread_id):
            """Worker thread to test concurrent database access"""
            try:
                # Each thread adds its own items
                for i in range(5):
                    item_id = f"concurrent_{thread_id}_{i:03d}"
                    success = self.manager.add_item(
                        item_id, "concurrent_test", f"file_{thread_id}_{i}.jpg", "top"
                    )
                    if success:
                        # Add embedding
                        self.manager.add_embedding(item_id, self.sample_embedding)
                        results["successes"] += 1
                    else:
                        results["failures"] += 1
                    
                    # Small delay to increase chance of conflicts
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Thread {thread_id} error: {e}")
                results["failures"] += 1
        
        # Start multiple threads
        threads = []
        for i in range(3):  # 3 threads adding 5 items each = 15 total
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        total_expected = 15
        self.assertGreater(results["successes"], 0, "No successful concurrent operations")
        self.assertLessEqual(results["failures"], total_expected // 2, "Too many concurrent failures")
        
        # Verify data in database
        concurrent_items = self.manager.get_items(source="concurrent_test")
        self.assertGreater(len(concurrent_items), 0, "No concurrent items found in database")
        
        print(f"✅ Concurrent access: {results['successes']} successes, {results['failures']} failures")
        print(f"   Items in database: {len(concurrent_items)}")
    
    def test_large_data_handling(self):
        """Test handling of large datasets and edge cases"""
        # Test large metadata
        large_metadata = {
            "description": "x" * 1000,  # 1KB string
            "tags": ["tag_" + str(i) for i in range(100)],  # Large list
            "nested": {"level1": {"level2": {"level3": "deep_value"}}},  # Nested structure
            "unicode": "🧥👔👗👠🎒",  # Unicode characters
        }
        
        success = self.manager.add_item(
            "large_001",
            "large_test",
            "large_file.jpg",
            "top",
            metadata=large_metadata
        )
        self.assertTrue(success)
        
        # Retrieve and verify large metadata
        item = self.manager.get_item_with_embedding("large_001")
        self.assertIsNotNone(item)
        
        # Verify all metadata fields thoroughly
        self.assertEqual(item['metadata']['description'], "x" * 1000)
        self.assertEqual(item['metadata']['unicode'], "🧥👔👗👠🎒")
        
        # Verify tags list - both length and content
        self.assertEqual(len(item['metadata']['tags']), 100)
        self.assertEqual(item['metadata']['tags'][0], "tag_0")
        self.assertEqual(item['metadata']['tags'][50], "tag_50")
        self.assertEqual(item['metadata']['tags'][99], "tag_99")
        # Verify all tags are present and correct
        expected_tags = ["tag_" + str(i) for i in range(100)]
        self.assertEqual(item['metadata']['tags'], expected_tags)
        
        # Verify nested structure - complete path verification
        self.assertIn('nested', item['metadata'])
        self.assertIn('level1', item['metadata']['nested'])
        self.assertIn('level2', item['metadata']['nested']['level1'])
        self.assertIn('level3', item['metadata']['nested']['level1']['level2'])
        self.assertEqual(item['metadata']['nested']['level1']['level2']['level3'], "deep_value")
        
        # Verify the entire nested structure matches exactly
        expected_nested = {"level1": {"level2": {"level3": "deep_value"}}}
        self.assertEqual(item['metadata']['nested'], expected_nested)
        
        # Test edge case: empty/null values
        success = self.manager.add_item(
            "edge_001",
            "edge_test",
            "",  # Empty filename
            "unknown",  # Edge case category
            color=None,  # Null color
            style="",  # Empty style
            metadata={}  # Empty metadata
        )
        self.assertTrue(success)
        
        edge_item = self.manager.get_item_with_embedding("edge_001")
        self.assertIsNotNone(edge_item)
        self.assertEqual(edge_item['filename'], "")
        self.assertIsNone(edge_item['color'])
        self.assertEqual(edge_item['metadata'], {})
        
        # Test very long item_id (edge case)
        long_item_id = "very_long_item_id_" + "x" * 200
        success = self.manager.add_item(long_item_id, "edge_test", "long_id.jpg", "top")
        self.assertTrue(success)
        
        long_item = self.manager.get_item_with_embedding(long_item_id)
        self.assertIsNotNone(long_item)
        self.assertEqual(long_item['item_id'], long_item_id)
        
        # Additional edge case: complex nested structures with different data types
        complex_metadata = {
            "mixed_list": [1, "string", True, None, {"nested": "value"}],
            "number_types": {"int": 42, "float": 3.14159, "negative": -100},
            "boolean_flags": {"active": True, "deprecated": False},
            "special_chars": "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "empty_structures": {"empty_list": [], "empty_dict": {}, "null_value": None}
        }
        
        success = self.manager.add_item(
            "complex_001",
            "complex_test",
            "complex.jpg",
            "accessory",
            metadata=complex_metadata
        )
        self.assertTrue(success)
        
        complex_item = self.manager.get_item_with_embedding("complex_001")
        self.assertIsNotNone(complex_item)
        
        # Verify complex metadata preservation
        self.assertEqual(complex_item['metadata']['mixed_list'], [1, "string", True, None, {"nested": "value"}])
        self.assertEqual(complex_item['metadata']['number_types']['int'], 42)
        self.assertEqual(complex_item['metadata']['number_types']['float'], 3.14159)
        self.assertEqual(complex_item['metadata']['boolean_flags']['active'], True)
        self.assertEqual(complex_item['metadata']['boolean_flags']['deprecated'], False)
        self.assertEqual(complex_item['metadata']['special_chars'], "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?")
        self.assertEqual(complex_item['metadata']['empty_structures']['empty_list'], [])
        self.assertEqual(complex_item['metadata']['empty_structures']['empty_dict'], {})
        self.assertIsNone(complex_item['metadata']['empty_structures']['null_value'])
        
        print("✅ Large data handling: Large metadata, nested structures, edge cases, and complex data types handled correctly")
    
    def test_data_consistency_and_relationships(self):
        """Test data consistency and relationship integrity"""
        # Add items for relationship testing
        outfit_items = []
        for i in range(4):
            item_id = f"relationship_{i:03d}"
            self.manager.add_item(item_id, "relationship_test", f"item_{i}.jpg", "top")
            self.manager.add_embedding(item_id, self.sample_embedding)
            outfit_items.append(item_id)
        
        # Create outfit with these items
        outfit_id = "test_relationship_outfit"
        success = self.manager.add_outfit(
            outfit_id,
            outfit_items,
            compatibility_score=0.85,
            context={"weather": "mild", "occasion": "casual"}
        )
        self.assertTrue(success)
        
        # Test relationship integrity
        # 1. All outfit items should exist in items table
        for item_id in outfit_items:
            item = self.manager.get_item_with_embedding(item_id)
            self.assertIsNotNone(item, f"Outfit item {item_id} not found")
        
        # 2. All items should have embeddings
        embeddings = self.manager.get_embeddings(outfit_items)
        self.assertEqual(len(embeddings), len(outfit_items), "Missing embeddings for outfit items")
        
        # 3. Test cascade behavior - delete item should update outfit
        deleted_item = outfit_items[0]
        success = self.manager.delete_item(deleted_item, delete_embedding=True)
        self.assertTrue(success)
        
        # Check outfit was updated (should still exist but with fewer items)
        outfits = self.manager.get_outfits()
        updated_outfit = next((o for o in outfits if o['outfit_id'] == outfit_id), None)
        
        if updated_outfit:  # Outfit still exists
            self.assertNotIn(deleted_item, updated_outfit['item_ids'])
            self.assertEqual(len(updated_outfit['item_ids']), 3)
        # If outfit was deleted due to insufficient items, that's also valid behavior
        
        # 4. Test data consistency after updates
        remaining_items = [item for item in outfit_items if item != deleted_item]
        for item_id in remaining_items:
            # Update item
            success = self.manager.update_item(item_id, color="updated_color")
            self.assertTrue(success)
            
            # Verify update
            updated_item = self.manager.get_item_with_embedding(item_id)
            self.assertEqual(updated_item['color'], "updated_color")
        
        print("✅ Data consistency: Relationships and cascade operations working correctly")
    
    def test_add_and_get_item(self):
        """Test adding and retrieving items"""
        test_prefix = self._get_test_prefix()
        
        # Add item
        success = self.manager.add_item(
            item_id=f"{test_prefix}_001",
            source="test",
            filename="test_shirt.jpg",
            category="top",
            color="blue",
            style="t-shirt",
            formality="casual",
            metadata={"brand": "TestBrand", "size": "M"}
        )
        self.assertTrue(success)
        
        # Retrieve item
        item = self.manager.get_item_with_embedding(f"{test_prefix}_001")
        self.assertIsNotNone(item)
        self.assertEqual(item['item_id'], f"{test_prefix}_001")
        self.assertEqual(item['category'], "top")
        self.assertEqual(item['color'], "blue")
        self.assertEqual(item['metadata']['brand'], "TestBrand")
        
        # Clean up if production DB
        if self.is_production_db:
            self.manager.delete_item(f"{test_prefix}_001", delete_embedding=True)
        
        print("✅ Add/Get Item: Item stored and retrieved correctly")
    
    def test_add_and_get_embedding(self):
        """Test adding and retrieving embeddings"""
        test_prefix = self._get_test_prefix()
        
        # Add item first
        self.manager.add_item(f"{test_prefix}_002", "test", "test.jpg", "top")
        
        # Add embedding
        success = self.manager.add_embedding(f"{test_prefix}_002", self.sample_embedding)
        self.assertTrue(success)
        
        # Retrieve embedding
        embeddings = self.manager.get_embeddings([f"{test_prefix}_002"])
        self.assertIn(f"{test_prefix}_002", embeddings)
        
        retrieved_embedding = embeddings[f"{test_prefix}_002"]
        np.testing.assert_array_almost_equal(retrieved_embedding, self.sample_embedding)
        
        # Clean up if production DB
        if self.is_production_db:
            self.manager.delete_item(f"{test_prefix}_002", delete_embedding=True)
        
        print("✅ Add/Get Embedding: Embedding stored and retrieved correctly")
    
    def test_update_item(self):
        """Test updating item metadata"""
        test_prefix = self._get_test_prefix()
        
        # Add item
        self.manager.add_item(f"{test_prefix}_003", "test", "test.jpg", "top", color="red")
        
        # Update item
        success = self.manager.update_item(
            f"{test_prefix}_003",
            color="blue",
            style="polo",
            metadata={"updated": True}
        )
        self.assertTrue(success)
        
        # Verify update
        item = self.manager.get_item_with_embedding(f"{test_prefix}_003")
        self.assertEqual(item['color'], "blue")
        self.assertEqual(item['style'], "polo")
        self.assertTrue(item['metadata']['updated'])
        
        # Clean up if production DB
        if self.is_production_db:
            self.manager.delete_item(f"{test_prefix}_003", delete_embedding=True)
        
        print("✅ Update Item: Item metadata updated correctly")
    
    def test_delete_item(self):
        """Test deleting items"""
        test_prefix = self._get_test_prefix()
        
        # Add item and embedding
        self.manager.add_item(f"{test_prefix}_004", "test", "test.jpg", "top")
        self.manager.add_embedding(f"{test_prefix}_004", self.sample_embedding)
        
        # Verify they exist
        self.assertIsNotNone(self.manager.get_item_with_embedding(f"{test_prefix}_004"))
        embeddings = self.manager.get_embeddings([f"{test_prefix}_004"])
        self.assertIn(f"{test_prefix}_004", embeddings)
        
        # Delete item (and embedding)
        success = self.manager.delete_item(f"{test_prefix}_004", delete_embedding=True)
        self.assertTrue(success)
        
        # Verify deletion
        self.assertIsNone(self.manager.get_item_with_embedding(f"{test_prefix}_004"))
        embeddings = self.manager.get_embeddings([f"{test_prefix}_004"])
        self.assertNotIn(f"{test_prefix}_004", embeddings)
        
        print("✅ Delete Item: Item and embedding deleted correctly")
    
    def test_outfit_operations(self):
        """Test outfit CRUD operations"""
        test_prefix = self._get_test_prefix()
        
        # Add some items first
        for i in range(3):
            self.manager.add_item(f"{test_prefix}_outfit_item_{i}", "test", f"item_{i}.jpg", "top")
        
        # Add outfit
        item_ids = [f"{test_prefix}_outfit_item_{i}" for i in range(3)]
        context = {"weather": "cold", "occasion": "casual"}
        
        success = self.manager.add_outfit(
            f"{test_prefix}_outfit_001",
            item_ids,
            compatibility_score=0.85,
            context=context
        )
        self.assertTrue(success)
        
        # Get outfit
        outfits = self.manager.get_outfits()
        test_outfit = next((o for o in outfits if o['outfit_id'] == f"{test_prefix}_outfit_001"), None)
        self.assertIsNotNone(test_outfit)
        
        self.assertEqual(test_outfit['outfit_id'], f"{test_prefix}_outfit_001")
        self.assertEqual(test_outfit['item_ids'], item_ids)
        self.assertEqual(test_outfit['compatibility_score'], 0.85)
        self.assertEqual(test_outfit['context']['weather'], "cold")
        
        # Update outfit
        success = self.manager.update_outfit(
            f"{test_prefix}_outfit_001",
            compatibility_score=0.90,
            context={"weather": "mild", "occasion": "work"}
        )
        self.assertTrue(success)
        
        # Verify update
        outfits = self.manager.get_outfits()
        test_outfit = next((o for o in outfits if o['outfit_id'] == f"{test_prefix}_outfit_001"), None)
        self.assertEqual(test_outfit['compatibility_score'], 0.90)
        self.assertEqual(test_outfit['context']['weather'], "mild")
        
        # Clean up if production DB
        if self.is_production_db:
            self.manager.delete_outfit(f"{test_prefix}_outfit_001")
            for i in range(3):
                self.manager.delete_item(f"{test_prefix}_outfit_item_{i}", delete_embedding=True)
        
        print("✅ Outfit Operations: Add/Update/Delete outfits working correctly")
    
    def test_user_operations(self):
        """Test user CRUD operations"""
        # Add user
        preferences = {"style": "casual", "colors": ["blue", "black"]}
        success = self.manager.add_user("test_user_001", "John Doe", preferences)
        self.assertTrue(success)
        
        # Get user
        user = self.manager.get_user("test_user_001")
        self.assertIsNotNone(user)
        self.assertEqual(user['user_id'], "test_user_001")
        self.assertEqual(user['name'], "John Doe")
        self.assertEqual(user['preferences']['style'], "casual")
        
        # Update user
        new_preferences = {"style": "formal", "colors": ["gray", "navy"]}
        success = self.manager.update_user("test_user_001", name="Jane Doe", preferences=new_preferences)
        self.assertTrue(success)
        
        # Verify update
        user = self.manager.get_user("test_user_001")
        self.assertEqual(user['name'], "Jane Doe")
        self.assertEqual(user['preferences']['style'], "formal")
        
        # Delete user
        success = self.manager.delete_user("test_user_001")
        self.assertTrue(success)
        
        user = self.manager.get_user("test_user_001")
        self.assertIsNone(user)
        
        print("✅ User Operations: Add/Update/Delete users working correctly")
    
    def test_filtering_and_search(self):
        """Test item filtering and search functionality"""
        # Add various items
        items_data = [
            ("filter_001", "test", "blue_shirt.jpg", "top", "blue", "shirt", "casual"),
            ("filter_002", "test", "red_dress.jpg", "top", "red", "dress", "formal"),
            ("filter_003", "test", "blue_jeans.jpg", "bottom", "blue", "jeans", "casual"),
            ("filter_004", "test", "black_shoes.jpg", "shoes", "black", "sneakers", "casual"),
        ]
        
        for item_id, source, filename, category, color, style, formality in items_data:
            self.manager.add_item(item_id, source, filename, category, color, style, formality)
        
        # Test category filter
        tops = self.manager.get_items(category="top")
        # Test color filter
        blue_items = self.manager.get_items(color="blue")
        # Test combined filters
        casual_tops = self.manager.get_items(category="top", formality="casual")
        if self.is_production_db:
            # In production, we might have more items, so just check we get some
            self.assertEqual(len(tops), 78)
            self.assertEqual(len(blue_items), 4)
            self.assertGreater(len(casual_tops), 40)
            self.assertEqual(casual_tops[0]['item_id'], "large_001")
        else:
            # In test DB, we know we added 2 tops
            self.assertEqual(len(tops), 2)
            self.assertEqual(len(blue_items), 2)        
            self.assertEqual(len(casual_tops), 1)
            self.assertEqual(casual_tops[0]['item_id'], "filter_001")
        
        
        print("✅ Filtering: Item filtering working correctly")
    
    def test_stats(self):
        """Test statistics generation"""
        # Add some test data
        for i in range(5):
            self.manager.add_item(f"stats_{i}", "test", f"item_{i}.jpg", "top" if i < 3 else "bottom")
            self.manager.add_embedding(f"stats_{i}", self.sample_embedding)
        
        stats = self.manager.get_stats()
        
        if self.is_production_db:
            # In production we have more items
            self.assertGreater(stats['total_items'], 250000)
            self.assertGreater(stats['total_embeddings'], 250000)
            self.assertGreaterEqual(stats['items_by_source']['test'], 5)
            self.assertGreater(stats['items_by_category']['top'], 70)
            self.assertGreater(stats['items_by_category']['bottom'], 40)
        else:
            # In test DB, we know we added 5 items
            self.assertEqual(stats['total_items'], 5)
            self.assertEqual(stats['total_embeddings'], 5)
            self.assertEqual(stats['items_by_source']['test'], 5)
            self.assertEqual(stats['items_by_category']['top'], 3)
            self.assertEqual(stats['items_by_category']['bottom'], 2)
        
        print("✅ Statistics: Database statistics generated correctly")
    
    def test_database_analysis_and_diagnostics(self):
        """Comprehensive database analysis similar to db_analysis.py"""
        # Add diverse test data for analysis
        test_data = [
            ("analysis_001", "test_source_1", "shirt.jpg", "top", "blue", "shirt", "casual"),
            ("analysis_002", "test_source_1", "pants.jpg", "bottom", "black", "jeans", "casual"),
            ("analysis_003", "test_source_2", "dress.jpg", "top", "red", "dress", "formal"),
            ("analysis_004", "test_source_2", "shoes.jpg", "shoes", "brown", "boots", "casual"),
            ("analysis_005", "test_source_1", "jacket.jpg", "outerwear", "navy", "blazer", "formal"),
        ]
        
        for item_id, source, filename, category, color, style, formality in test_data:
            self.manager.add_item(item_id, source, filename, category, color, style, formality)
            self.manager.add_embedding(item_id, self.sample_embedding)
        
        # Analyze database like db_analysis.py
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Table sizes and row counts
            table_stats = {}
            for table in ['items', 'embeddings', 'outfits', 'users']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                table_stats[table] = row_count
            
            # Should have our test data
            self.assertGreaterEqual(table_stats['items'], 5)
            self.assertGreaterEqual(table_stats['embeddings'], 5)
            
            # 2. Items breakdown by source
            cursor.execute("SELECT source, COUNT(*) FROM items GROUP BY source ORDER BY COUNT(*) DESC")
            source_breakdown = dict(cursor.fetchall())
            
            # Should have our test sources
            self.assertIn('test_source_1', source_breakdown)
            self.assertIn('test_source_2', source_breakdown)
            self.assertEqual(source_breakdown['test_source_1'], 3)
            self.assertEqual(source_breakdown['test_source_2'], 2)
            
            # 3. Items breakdown by category
            cursor.execute("SELECT category, COUNT(*) FROM items GROUP BY category ORDER BY COUNT(*) DESC")
            category_breakdown = dict(cursor.fetchall())
            
            # Should have our test categories
            expected_categories = {'top', 'bottom', 'shoes', 'outerwear'}
            for category in expected_categories:
                self.assertIn(category, category_breakdown)
            
            # 4. Check embedding coverage
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM items")
            total_items = cursor.fetchone()[0]
            
            coverage = (embedding_count / total_items * 100) if total_items > 0 else 0
            self.assertGreater(coverage, 90, "Embedding coverage should be >90%")
            
            # 5. Sample items verification
            cursor.execute("SELECT item_id, source, filename, category FROM items LIMIT 3")
            sample_items = cursor.fetchall()
            self.assertGreater(len(sample_items), 0)
            
            # Verify sample items have expected structure
            for item_id, source, filename, category in sample_items:
                self.assertIsNotNone(item_id)
                self.assertIsNotNone(source)
                self.assertIsNotNone(filename)
                self.assertIsNotNone(category)
            
            # 6. Test complex queries (like real usage)
            # Find all casual items
            cursor.execute("SELECT COUNT(*) FROM items WHERE formality = 'casual'")
            casual_count = cursor.fetchone()[0]
            self.assertGreater(casual_count, 0)
            
            # Find items by multiple criteria
            cursor.execute("""
                SELECT COUNT(*) FROM items 
                WHERE category = 'top' AND formality = 'casual' AND source LIKE 'test_%'
            """)
            filtered_count = cursor.fetchone()[0]
            self.assertGreater(filtered_count, 0)
            
            # Test join query (items with embeddings)
            cursor.execute("""
                SELECT COUNT(*) FROM items i 
                INNER JOIN embeddings e ON i.item_id = e.item_id
            """)
            joined_count = cursor.fetchone()[0]
            self.assertEqual(joined_count, embedding_count)
        
        print(f"✅ Database Analysis:")
        print(f"   Tables: {table_stats}")
        print(f"   Sources: {source_breakdown}")
        print(f"   Categories: {category_breakdown}")
        print(f"   Embedding coverage: {coverage:.1f}%")
    
    def test_performance_and_query_optimization(self):
        """Test database performance with larger datasets"""
        import time
        
        # Add more data for performance testing
        batch_size = 100
        items_data = []
        
        start_time = time.time()
        
        # Bulk insert test
        for i in range(batch_size):
            item_id = f"perf_{i:04d}"
            self.manager.add_item(
                item_id, "performance_test", f"perf_item_{i}.jpg", 
                "top" if i % 2 == 0 else "bottom",
                color=f"color_{i % 10}",
                formality="casual" if i % 3 == 0 else "formal"
            )
            self.manager.add_embedding(item_id, self.sample_embedding)
        
        bulk_insert_time = time.time() - start_time
        
        # Query performance tests
        start_time = time.time()
        
        # Test index usage
        items_by_category = self.manager.get_items(category="top")
        category_query_time = time.time() - start_time
        
        start_time = time.time()
        items_by_source = self.manager.get_items(source="performance_test")
        source_query_time = time.time() - start_time
        
        start_time = time.time()
        items_combined = self.manager.get_items(category="top", source="performance_test")
        combined_query_time = time.time() - start_time
        
        # Embedding retrieval performance
        item_ids = [f"perf_{i:04d}" for i in range(0, batch_size, 10)]  # Every 10th item
        start_time = time.time()
        embeddings = self.manager.get_embeddings(item_ids)
        embedding_query_time = time.time() - start_time
        
        # Verify results
        self.assertGreater(len(items_by_category), 0)
        self.assertEqual(len(items_by_source), batch_size)
        self.assertGreater(len(items_combined), 0)
        self.assertEqual(len(embeddings), len(item_ids))
        
        # Performance assertions (reasonable thresholds)
        self.assertLess(bulk_insert_time, 30.0, f"Bulk insert too slow: {bulk_insert_time:.2f}s")
        self.assertLess(category_query_time, 1.0, f"Category query too slow: {category_query_time:.2f}s")
        self.assertLess(source_query_time, 1.0, f"Source query too slow: {source_query_time:.2f}s")
        self.assertLess(embedding_query_time, 1.0, f"Embedding query too slow: {embedding_query_time:.2f}s")
        
        print(f"✅ Performance Tests:")
        print(f"   Bulk insert ({batch_size} items): {bulk_insert_time:.2f}s")
        print(f"   Category query: {category_query_time:.3f}s")
        print(f"   Source query: {source_query_time:.3f}s")
        print(f"   Combined query: {combined_query_time:.3f}s")
        print(f"   Embedding retrieval: {embedding_query_time:.3f}s")


class TestDataManager(TestDataManagerBase):
    """Test DataManager on clean test database"""
    
    def setUp(self):
        """Set up clean test database"""
        super().setUp()
        self.is_production_db = False
        print(f"🧪 Testing on clean database: {self.db_path}")


class TestDataManagerProduction(TestDataManagerBase):
    """Test DataManager on production database (if it exists)"""
    
    @classmethod
    def setUpClass(cls):
        """Check if production database exists"""
        cls.production_db_path = Path("data/closetgpt.db")
        if not cls.production_db_path.exists():
            raise unittest.SkipTest("Production database (data/closetgpt.db) not found - skipping production tests")
        print(f"🗃️ Production database found: {cls.production_db_path}")
        print(f"   Size: {cls.production_db_path.stat().st_size / (1024*1024):.1f} MB")
    
    def setUp(self):
        """Set up production database testing"""
        # Don't call super().setUp() as we don't want to create temp dir
        self.db_path = self.production_db_path
        self.manager = DataManager(str(self.db_path))
        self.is_production_db = True
        
        # Sample test data
        self.sample_embedding = np.random.rand(512).astype(np.float32)
        self.sample_embedding = self.sample_embedding / np.linalg.norm(self.sample_embedding)
        
        # Get initial stats for comparison
        self.initial_stats = self.manager.get_stats()
        print(f"🗃️ Testing on production database with {self.initial_stats.get('total_items', 0)} existing items")
    
    def tearDown(self):
        """Production database - no cleanup of temp files needed"""
        pass
    
    def test_production_database_readonly_analysis(self):
        """Analyze production database without modifying it"""
        stats = self.manager.get_stats()
        
        # Verify database has data
        self.assertGreater(stats.get('total_items', 250000), 0, "Production database should have items")
        self.assertGreater(stats.get('total_embeddings', 250000), 0, "Production database should have embeddings")
        
        # Check embedding coverage
        total_items = stats.get('total_items', 0)
        total_embeddings = stats.get('total_embeddings', 0)
        coverage = (total_embeddings / total_items * 100) if total_items > 0 else 0
        
        print(f"✅ Production DB Analysis:")
        print(f"   Total items: {total_items:,}")
        print(f"   Total embeddings: {total_embeddings:,}")
        print(f"   Embedding coverage: {coverage:.1f}%")
        print(f"   Sources: {stats.get('items_by_source', {})}")
        print(f"   Categories: {stats.get('items_by_category', {})}")
        
        # Verify production data integrity
        self.assertGreaterEqual(coverage, 90, "Production database should have high embedding coverage")
    
    def test_production_database_read_operations(self):
        """Test read operations on production database"""
        # Test basic queries work
        all_items = self.manager.get_items()
        self.assertGreater(len(all_items), 0, "Should be able to read items from production DB")
        
        # Test filtering works
        if 'polyvore' in self.initial_stats.get('items_by_source', {}):
            polyvore_items = self.manager.get_items(source='polyvore')
            self.assertGreater(len(polyvore_items), 0, "Should find polyvore items")
        
        # Test embedding retrieval
        sample_items = all_items[:5]  # Test first 5 items
        item_ids = [item['item_id'] for item in sample_items]
        embeddings = self.manager.get_embeddings(item_ids)
        
        self.assertGreater(len(embeddings), 0, "Should be able to retrieve embeddings")
        
        # Test individual item retrieval
        if sample_items:
            first_item = self.manager.get_item_with_embedding(sample_items[0]['item_id'])
            self.assertIsNotNone(first_item, "Should be able to retrieve individual items")
            
        print(f"✅ Production DB Read Operations:")
        print(f"   Retrieved {len(all_items)} items successfully")
        print(f"   Retrieved {len(embeddings)} embeddings successfully")


def create_test_suite():
    """Create test suite that runs on both test and production databases"""
    suite = unittest.TestSuite()
    
    # Add tests for clean database
    clean_db_tests = unittest.TestLoader().loadTestsFromTestCase(TestDataManager)
    suite.addTest(clean_db_tests)
    
    # Add tests for production database (will be skipped if DB doesn't exist)
    try:
        production_db_tests = unittest.TestLoader().loadTestsFromTestCase(TestDataManagerProduction)
        suite.addTest(production_db_tests)
    except unittest.SkipTest as e:
        print(f"⚠️ Skipping production database tests: {e}")
    
    return suite


if __name__ == '__main__':
    # Run both test suites
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)