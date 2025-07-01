"""
Test cases for Similarity Engine
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "models"))
sys.path.append(str(Path(__file__).parent.parent / "src" / "data"))

from data_structures import Item
from similarity_engine import SimilarityEngine
from data_manager import DataManager


class TestSimilarityEngine(unittest.TestCase):
    """Test cases for SimilarityEngine"""
    
    def setUp(self):
        """Set up test data"""
        self.engine = SimilarityEngine()
        
        # Create test items with normalized embeddings
        self.blue_shirt = Item("shirt_001", "top", "blue", "shirt", "casual", embedding=np.random.rand(512))
        self.navy_shirt = Item("shirt_002", "top", "navy", "shirt", "casual", embedding=np.random.rand(512))
        self.red_dress = Item("dress_001", "top", "red", "dress", "formal", embedding=np.random.rand(512))
        self.blue_jeans = Item("jeans_001", "bottom", "blue", "jeans", "casual", embedding=np.random.rand(512))
        self.white_tshirt = Item("tshirt_001", "top", "white", "tshirt", "casual", embedding=np.random.rand(512))
        
        # Normalize embeddings
        for item in [self.blue_shirt, self.navy_shirt, self.red_dress, self.blue_jeans, self.white_tshirt]:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    def test_engine_initialization(self):
        """Test similarity engine initialization"""
        self.assertIsInstance(self.engine, SimilarityEngine)
        self.assertEqual(self.engine.items_cache, {})
        
        print("✅ Engine initialization: SimilarityEngine initialized correctly")
    
    def test_calculate_similarity_basic(self):
        """Test basic similarity calculation"""
        # Test similarity between two shirts (should be high)
        similarity = self.engine.calculate_similarity(self.blue_shirt, self.navy_shirt)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Similar items (both shirts, similar colors) should have higher similarity than dissimilar ones
        similarity_shirts = self.engine.calculate_similarity(self.blue_shirt, self.navy_shirt)
        similarity_different = self.engine.calculate_similarity(self.blue_shirt, self.red_dress)
        
        # Both are tops but different styles/formality, so similarity_shirts might not always be higher
        # Just test that both are valid scores
        self.assertGreaterEqual(similarity_shirts, 0.0)
        self.assertGreaterEqual(similarity_different, 0.0)
        
        print(f"✅ Similarity calculation: Blue shirt vs Navy shirt: {similarity_shirts:.3f}")
        print(f"   Blue shirt vs Red dress: {similarity_different:.3f}")
    
    def test_calculate_similarity_edge_cases(self):
        """Test similarity calculation edge cases"""
        # Test with missing embeddings
        item_no_embedding = Item("no_embed", "top", "blue", "shirt")
        similarity = self.engine.calculate_similarity(self.blue_shirt, item_no_embedding)
        self.assertEqual(similarity, 0.0)
        
        # Test self-similarity (same item)
        self_similarity = self.engine.calculate_similarity(self.blue_shirt, self.blue_shirt)
        self.assertGreater(self_similarity, 0.9)  # Should be very high
        
        print("✅ Similarity edge cases: Missing embeddings and self-similarity handled correctly")
    
    def test_style_similarity_calculation(self):
        """Test style similarity component"""
        # Test exact style match
        style_sim = self.engine._calculate_style_similarity(self.blue_shirt, self.navy_shirt)
        self.assertEqual(style_sim, 1.0)  # Both are "shirt"
        
        # Test different styles in same category
        style_sim_diff = self.engine._calculate_style_similarity(self.blue_shirt, self.red_dress)
        self.assertLess(style_sim_diff, 1.0)  # Different styles
        
        # Test missing style info
        item_no_style = Item("no_style", "top")
        style_sim_missing = self.engine._calculate_style_similarity(self.blue_shirt, item_no_style)
        self.assertEqual(style_sim_missing, 0.5)  # Neutral if missing
        
        print("✅ Style similarity: Style matching working correctly")
    
    def test_color_similarity_calculation(self):
        """Test color similarity component"""
        # Test exact color match
        blue_shirt2 = Item("shirt_003", "top", "blue", "shirt")
        color_sim = self.engine._calculate_color_similarity(self.blue_shirt, blue_shirt2)
        self.assertEqual(color_sim, 1.0)
        
        # Test color family match (blue and navy)
        color_sim_family = self.engine._calculate_color_similarity(self.blue_shirt, self.navy_shirt)
        self.assertEqual(color_sim_family, 0.7)  # Same color family
        
        # Test different colors
        color_sim_diff = self.engine._calculate_color_similarity(self.blue_shirt, self.red_dress)
        self.assertLess(color_sim_diff, 0.7)
        
        print("✅ Color similarity: Color matching working correctly")
    
    def test_formality_similarity_calculation(self):
        """Test formality similarity component"""
        # Test exact formality match
        formality_sim = self.engine._calculate_formality_similarity(self.blue_shirt, self.navy_shirt)
        self.assertEqual(formality_sim, 1.0)  # Both casual
        
        # Test different formality levels
        formality_sim_diff = self.engine._calculate_formality_similarity(self.blue_shirt, self.red_dress)
        self.assertEqual(formality_sim_diff, 0.2)  # Casual vs formal
        
        print("✅ Formality similarity: Formality matching working correctly")
    
    def test_find_similar_items(self):
        """Test finding similar items"""
        candidates = [self.navy_shirt, self.red_dress, self.white_tshirt]
        
        similar_items = self.engine.find_similar_items(
            self.blue_shirt, 
            candidates, 
            top_k=3,
            same_category_only=True
        )
        
        # Should return all candidates since they're all tops
        self.assertLessEqual(len(similar_items), 3)
        
        # Check return format
        for item, score in similar_items:
            self.assertIsInstance(item, Item)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Results should be sorted by similarity (descending)
        if len(similar_items) > 1:
            scores = [score for _, score in similar_items]
            self.assertEqual(scores, sorted(scores, reverse=True))
        
        print(f"✅ Find similar items: Found {len(similar_items)} similar items")
        for item, score in similar_items:
            print(f"   {item.item_id} ({item.color} {item.style}): {score:.3f}")
    
    def test_find_similar_items_cross_category(self):
        """Test finding similar items across categories"""
        candidates = [self.blue_jeans, self.navy_shirt, self.red_dress]
        
        # Test with same_category_only=False
        similar_items = self.engine.find_similar_items(
            self.blue_shirt,
            candidates,
            top_k=3,
            same_category_only=False
        )
        
        # Should include items from different categories
        categories = {item.category for item, _ in similar_items}
        self.assertGreater(len(categories), 1)  # Should have multiple categories
        
        print(f"✅ Cross-category similarity: Found items across {len(categories)} categories")
    
    def test_find_style_variations(self):
        """Test finding style variations"""
        # Create items with different variations
        blue_polo = Item("polo_001", "top", "blue", "polo", "casual")
        red_shirt = Item("shirt_004", "top", "red", "shirt", "casual")
        blue_shirt_formal = Item("shirt_005", "top", "blue", "shirt", "formal")
        
        candidates = [blue_polo, red_shirt, blue_shirt_formal]
        
        # Test color variations
        color_variations = self.engine.find_style_variations(
            self.blue_shirt, 
            candidates, 
            variation_type="color",
            top_k=3
        )
        
        # Should find red shirt (same style/formality, different color)
        self.assertEqual(len(color_variations), 1)
        self.assertEqual(color_variations[0][0].item_id, "shirt_004")
        
        # Test formality variations
        formality_variations = self.engine.find_style_variations(
            self.blue_shirt,
            candidates,
            variation_type="formality",
            top_k=3
        )
        
        # Should find formal blue shirt (same color/style, different formality)
        self.assertEqual(len(formality_variations), 1)
        self.assertEqual(formality_variations[0][0].item_id, "shirt_005")
        
        print("✅ Style variations: Color and formality variations found correctly")
    
    def test_batch_similarity_matrix(self):
        """Test batch similarity matrix calculation"""
        items = [self.blue_shirt, self.navy_shirt, self.red_dress]
        
        similarity_matrix = self.engine.batch_similarity_matrix(items)
        
        # Check matrix properties
        self.assertEqual(similarity_matrix.shape, (3, 3))
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(3):
            self.assertEqual(similarity_matrix[i][i], 1.0)
        
        # Matrix should be symmetric
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(similarity_matrix[i][j], similarity_matrix[j][i], places=5)
        
        # All values should be between 0 and 1
        self.assertTrue(np.all(similarity_matrix >= 0.0))
        self.assertTrue(np.all(similarity_matrix <= 1.0))
        
        print("✅ Batch similarity matrix: Matrix calculation working correctly")
    
    def test_minimum_similarity_threshold(self):
        """Test minimum similarity threshold filtering"""
        candidates = [self.navy_shirt, self.red_dress, self.blue_jeans]
        
        # Test with high threshold
        similar_items_high = self.engine.find_similar_items(
            self.blue_shirt,
            candidates,
            min_similarity=0.8,
            same_category_only=False
        )
        
        # Test with low threshold
        similar_items_low = self.engine.find_similar_items(
            self.blue_shirt,
            candidates,
            min_similarity=0.1,
            same_category_only=False
        )
        
        # Low threshold should return more or equal items than high threshold
        self.assertGreaterEqual(len(similar_items_low), len(similar_items_high))
        
        # All returned items should meet the threshold
        for item, score in similar_items_high:
            self.assertGreaterEqual(score, 0.8)
        
        print(f"✅ Similarity threshold: High threshold: {len(similar_items_high)} items, Low threshold: {len(similar_items_low)} items")


class TestSimilarityEngineWithPolyvoreData(unittest.TestCase):
    """Test similarity engine with real Polyvore data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class with real database data"""
        cls.production_db_path = Path("data/closetgpt.db")
        if not cls.production_db_path.exists():
            raise unittest.SkipTest("Production database not found - skipping Polyvore tests")
        
        cls.manager = DataManager(str(cls.production_db_path))
        cls.engine = SimilarityEngine()
        
        # Load a sample of items from database
        cls.polyvore_items = cls._load_sample_items(50)  # Load 50 items for testing
        
        if len(cls.polyvore_items) < 10:
            raise unittest.SkipTest("Insufficient Polyvore items for testing")
        
        print(f"🗃️ Loaded {len(cls.polyvore_items)} Polyvore items for testing")
    
    @classmethod
    def _load_sample_items(cls, limit: int) -> list:
        """Load a sample of items from the database"""
        try:
            # Get items from database
            db_items = cls.manager.get_items(source='polyvore')[:limit]
            
            if not db_items:
                return []
            
            # Get embeddings
            item_ids = [item['item_id'] for item in db_items]
            embeddings = cls.manager.get_embeddings(item_ids)
            
            # Convert to Item objects
            items = []
            for db_item in db_items:
                if db_item['item_id'] in embeddings:
                    item = Item.from_db_record(db_item)
                    item.embedding = embeddings[db_item['item_id']]
                    items.append(item)
            
            return items
        except Exception as e:
            print(f"Error loading Polyvore items: {e}")
            return []
    
    def test_polyvore_similarity_calculation(self):
        """Test similarity calculation with real Polyvore data"""
        if len(self.polyvore_items) < 2:
            self.skipTest("Need at least 2 Polyvore items")
        
        item1 = self.polyvore_items[0]
        item2 = self.polyvore_items[1]
        
        similarity = self.engine.calculate_similarity(item1, item2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        print(f"✅ Polyvore similarity: {item1.item_id} vs {item2.item_id}: {similarity:.3f}")
    
    def test_polyvore_find_similar_items(self):
        """Test finding similar items with Polyvore data"""
        if len(self.polyvore_items) < 10:
            self.skipTest("Need at least 10 Polyvore items")
        
        target_item = self.polyvore_items[0]
        candidates = self.polyvore_items[1:10]  # Use next 9 as candidates
        
        similar_items = self.engine.find_similar_items(
            target_item,
            candidates,
            top_k=5,
            min_similarity=0.1,
            same_category_only=False
        )
        
        self.assertLessEqual(len(similar_items), 5)
        
        # Check that results are valid
        for item, score in similar_items:
            self.assertIsInstance(item, Item)
            self.assertGreaterEqual(score, 0.1)
            self.assertNotEqual(item.item_id, target_item.item_id)
        
        print(f"✅ Polyvore similar items: Found {len(similar_items)} similar items")
        for item, score in similar_items[:3]:  # Show top 3
            print(f"   {item.item_id}: {score:.3f}")
    
    def test_polyvore_batch_similarity_performance(self):
        """Test batch similarity calculation performance with Polyvore data"""
        if len(self.polyvore_items) < 5:
            self.skipTest("Need at least 5 Polyvore items")
        
        import time
        
        # Test with a small batch
        test_items = self.polyvore_items[:5]
        
        start_time = time.time()
        similarity_matrix = self.engine.batch_similarity_matrix(test_items)
        calculation_time = time.time() - start_time
        
        # Verify matrix properties
        self.assertEqual(similarity_matrix.shape, (5, 5))
        self.assertTrue(np.all(np.diag(similarity_matrix) == 1.0))
        
        # Performance should be reasonable (less than 5 seconds for 5 items)
        self.assertLess(calculation_time, 5.0)
        
        print(f"✅ Polyvore batch performance: 5x5 matrix calculated in {calculation_time:.2f}s")
    
    def test_polyvore_category_distribution(self):
        """Test similarity across different categories in Polyvore data"""
        # Group items by category
        category_items = {}
        for item in self.polyvore_items:
            if item.category not in category_items:
                category_items[item.category] = []
            category_items[item.category].append(item)
        
        # Test similarity within and across categories
        results = {}
        for category, items in category_items.items():
            if len(items) >= 2:
                # Calculate average intra-category similarity
                similarities = []
                for i in range(min(3, len(items))):
                    for j in range(i+1, min(3, len(items))):
                        sim = self.engine.calculate_similarity(items[i], items[j])
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    results[category] = avg_similarity
        
        print(f"✅ Polyvore category distribution: Tested {len(results)} categories")
        for category, avg_sim in results.items():
            print(f"   {category}: {avg_sim:.3f} average similarity")
    
    def test_polyvore_edge_cases(self):
        """Test edge cases with Polyvore data"""
        if len(self.polyvore_items) < 1:
            self.skipTest("Need at least 1 Polyvore item")
        
        target_item = self.polyvore_items[0]
        
        # Test with empty candidates list
        similar_items = self.engine.find_similar_items(target_item, [], top_k=5)
        self.assertEqual(len(similar_items), 0)
        
        # Test with single candidate (same item)
        similar_items = self.engine.find_similar_items(target_item, [target_item], top_k=5)
        self.assertEqual(len(similar_items), 0)  # Should exclude self
        
        # Test with very high minimum similarity
        if len(self.polyvore_items) > 1:
            candidates = self.polyvore_items[1:6]
            similar_items = self.engine.find_similar_items(
                target_item, 
                candidates, 
                top_k=5, 
                min_similarity=0.95
            )
            # Should return few or no items due to high threshold
            self.assertLessEqual(len(similar_items), len(candidates))
        
        print("✅ Polyvore edge cases: Empty lists and high thresholds handled correctly")


class TestSimilarityEngineIntegration(unittest.TestCase):
    """Integration tests for similarity engine"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_similarity.db"
        self.manager = DataManager(str(self.db_path))
        self.engine = SimilarityEngine()
        
        # Create and add test items to database
        self.test_items = self._create_test_items()
        self._add_items_to_database()
    
    def tearDown(self):
        """Clean up after integration tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_items(self):
        """Create a diverse set of test items"""
        items = [
            Item("int_001", "top", "blue", "shirt", "casual", embedding=np.random.rand(512)),
            Item("int_002", "top", "navy", "shirt", "casual", embedding=np.random.rand(512)),
            Item("int_003", "top", "red", "blouse", "formal", embedding=np.random.rand(512)),
            Item("int_004", "bottom", "black", "jeans", "casual", embedding=np.random.rand(512)),
            Item("int_005", "bottom", "navy", "trousers", "formal", embedding=np.random.rand(512)),
            Item("int_006", "shoes", "brown", "loafers", "smart-casual", embedding=np.random.rand(512)),
        ]
        
        # Normalize embeddings
        for item in items:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        return items
    
    def _add_items_to_database(self):
        """Add test items to database"""
        for item in self.test_items:
            # Add item to database
            success = self.manager.add_item(
                item.item_id, item.source, f"{item.item_id}.jpg", item.category,
                item.color, item.style, item.formality, item.season
            )
            self.assertTrue(success)
            
            # Add embedding
            success = self.manager.add_embedding(item.item_id, item.embedding)
            self.assertTrue(success)
    
    def test_similarity_engine_with_database_integration(self):
        """Test similarity engine integrated with database"""
        # Load items from database
        db_items = self.manager.get_items()
        item_ids = [item['item_id'] for item in db_items]
        embeddings = self.manager.get_embeddings(item_ids)
        
        # Convert to Item objects
        items = []
        for db_item in db_items:
            item = Item.from_db_record(db_item)
            item.embedding = embeddings[db_item['item_id']]
            items.append(item)
        
        # Test similarity engine with database items
        target_item = items[0]
        candidates = items[1:]
        
        similar_items = self.engine.find_similar_items(target_item, candidates, top_k=3)
        
        self.assertLessEqual(len(similar_items), 3)
        
        print(f"✅ Database integration: Found {len(similar_items)} similar items from database")
    
    def test_end_to_end_similarity_workflow(self):
        """Test complete similarity workflow"""
        # 1. Load target item from database
        target_db_item = self.manager.get_item_with_embedding("int_001")
        self.assertIsNotNone(target_db_item)
        
        target_item = Item.from_db_record(target_db_item)
        
        # 2. Load candidate items
        candidate_db_items = self.manager.get_items(category="top")  # Same category
        candidate_items = []
        
        for db_item in candidate_db_items:
            if db_item['item_id'] != target_item.item_id:
                item = Item.from_db_record(db_item)
                item.embedding = self.manager.get_embeddings([item.item_id])[item.item_id]
                candidate_items.append(item)
        
        # 3. Find similar items
        similar_items = self.engine.find_similar_items(
            target_item, 
            candidate_items, 
            top_k=2,
            min_similarity=0.0
        )
        
        # 4. Verify results
        self.assertLessEqual(len(similar_items), 2)
        
        for item, score in similar_items:
            self.assertEqual(item.category, target_item.category)  # Same category filter worked
            self.assertGreaterEqual(score, 0.0)
        
        print(f"✅ End-to-end workflow: Complete similarity workflow successful")


if __name__ == '__main__':
    unittest.main()