"""
Test cases for Unified Recommendation Engine
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

from data_structures import Item, RecommendationContext, Recommendation
from recommendation_engine import RecommendationEngine
from data_manager import DataManager


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine"""
    
    def setUp(self):
        """Set up test data"""
        # Create temp database for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_recommendations.db"
        self.manager = DataManager(str(self.db_path))
        
        # Initialize recommendation engine
        self.engine = RecommendationEngine(data_manager=self.manager)
        
        # Create test items
        self.test_items = self._create_test_items()
        self._add_items_to_database()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_items(self):
        """Create diverse test items"""
        items = [
            # Similar items (for similarity testing)
            Item("blue_shirt_1", "top", "blue", "shirt", "casual", embedding=np.random.rand(512)),
            Item("navy_shirt_1", "top", "navy", "shirt", "casual", embedding=np.random.rand(512)),
            Item("blue_polo_1", "top", "blue", "polo", "casual", embedding=np.random.rand(512)),
            
            # Complementary items
            Item("dark_jeans_1", "bottom", "dark_blue", "jeans", "casual", embedding=np.random.rand(512)),
            Item("white_sneakers_1", "shoes", "white", "sneakers", "casual", embedding=np.random.rand(512)),
            Item("brown_belt_1", "accessory", "brown", "belt", "casual", embedding=np.random.rand(512)),
            
            # Formal items
            Item("white_dress_shirt", "top", "white", "dress_shirt", "formal", embedding=np.random.rand(512)),
            Item("black_suit_pants", "bottom", "black", "suit_pants", "formal", embedding=np.random.rand(512)),
            Item("black_dress_shoes", "shoes", "black", "oxfords", "formal", embedding=np.random.rand(512)),
            
            # Seasonal items
            Item("winter_coat", "outerwear", "black", "winter_coat", "casual", season="cold", embedding=np.random.rand(512)),
            Item("summer_shorts", "bottom", "khaki", "shorts", "casual", season="warm", embedding=np.random.rand(512)),
        ]
        
        # Normalize embeddings
        for item in items:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        return items
    
    def _add_items_to_database(self):
        """Add test items to database"""
        for item in self.test_items:
            success = self.manager.add_item(
                item.item_id, item.source, f"{item.item_id}.jpg", item.category,
                item.color, item.style, item.formality, item.season
            )
            self.assertTrue(success)
            
            success = self.manager.add_embedding(item.item_id, item.embedding)
            self.assertTrue(success)
    
    def test_engine_initialization(self):
        """Test recommendation engine initialization"""
        self.assertIsNotNone(self.engine.similarity_engine)
        self.assertIsNotNone(self.engine.compatibility_engine)
        self.assertEqual(self.engine.data_manager, self.manager)
        
        print("✅ Engine initialization: RecommendationEngine initialized correctly")
    
    def test_load_items_from_database(self):
        """Test loading items from database"""
        # Load all items
        loaded_items = self.engine.load_items_from_database()
        self.assertEqual(len(loaded_items), len(self.test_items))
        
        # Load items by category
        top_items = self.engine.load_items_from_database(category="top")
        expected_tops = len([item for item in self.test_items if item.category == "top"])
        self.assertEqual(len(top_items), expected_tops)
        
        # Load items with limit
        limited_items = self.engine.load_items_from_database(limit=3)
        self.assertEqual(len(limited_items), 3)
        
        # Check that items have embeddings
        for item in loaded_items:
            self.assertIsNotNone(item.embedding)
        
        print(f"✅ Load items from database: Loaded {len(loaded_items)} items successfully")
    
    def test_find_similar_items(self):
        """Test finding similar items"""
        target_item = next(item for item in self.test_items if item.item_id == "blue_shirt_1")
        
        similar_recommendations = self.engine.find_similar_items(
            target_item,
            top_k=3
        )
        
        self.assertLessEqual(len(similar_recommendations), 3)
        
        # Check recommendation format
        for rec in similar_recommendations:
            self.assertIsInstance(rec, Recommendation)
            self.assertEqual(rec.recommendation_type, "similar")
            self.assertEqual(rec.item.category, target_item.category)  # Same category
            self.assertGreater(rec.score, 0.0)
        
        # Results should be sorted by score
        if len(similar_recommendations) > 1:
            scores = [rec.score for rec in similar_recommendations]
            self.assertEqual(scores, sorted(scores, reverse=True))
        
        print(f"✅ Find similar items: Found {len(similar_recommendations)} similar items")
        for rec in similar_recommendations:
            print(f"   {rec.item.item_id}: {rec.score:.3f} - {rec.reason}")
    
    def test_find_complementary_items(self):
        """Test finding complementary items"""
        target_item = next(item for item in self.test_items if item.item_id == "blue_shirt_1")
        
        complementary_recommendations = self.engine.find_complementary_items(
            target_item,
            top_k=5
        )
        
        self.assertLessEqual(len(complementary_recommendations), 5)
        
        # Check recommendation format
        for rec in complementary_recommendations:
            self.assertIsInstance(rec, Recommendation)
            self.assertEqual(rec.recommendation_type, "complementary")
            self.assertNotEqual(rec.item.category, target_item.category)  # Different category
            self.assertGreater(rec.score, 0.0)
        
        print(f"✅ Find complementary items: Found {len(complementary_recommendations)} complementary items")
        for rec in complementary_recommendations:
            print(f"   {rec.item.item_id} ({rec.item.category}): {rec.score:.3f}")
    
    def test_complete_outfit(self):
        """Test outfit completion"""
        base_item = next(item for item in self.test_items if item.item_id == "blue_shirt_1")
        partial_outfit = [base_item]
        
        completion_recommendations = self.engine.complete_outfit(
            partial_outfit,
            target_categories=["bottom", "shoes"]
        )
        
        # Should suggest items from target categories
        suggested_categories = {rec.item.category for rec in completion_recommendations}
        self.assertTrue(suggested_categories.issubset({"bottom", "shoes"}))
        
        # Check recommendation format
        for rec in completion_recommendations:
            self.assertIsInstance(rec, Recommendation)
            self.assertEqual(rec.recommendation_type, "outfit_completion")
            self.assertGreater(rec.score, 0.0)
        
        print(f"✅ Complete outfit: Found {len(completion_recommendations)} completion suggestions")
    
    def test_generate_complete_outfit(self):
        """Test generating complete outfit from base item"""
        base_item = next(item for item in self.test_items if item.item_id == "blue_shirt_1")
        
        outfit, score = self.engine.generate_complete_outfit(base_item)
        
        self.assertGreaterEqual(len(outfit), 1)  # At least the base item
        self.assertIn(base_item, outfit)  # Should include base item
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Check outfit has items from different categories
        categories = {item.category for item in outfit}
        self.assertGreaterEqual(len(categories), 1)
        
        print(f"✅ Generate complete outfit: Created {len(outfit)}-item outfit with score {score:.3f}")
        for item in outfit:
            print(f"   {item.item_id} ({item.category})")
    
    def test_context_filtering(self):
        """Test context-aware filtering"""
        # Create contexts
        cold_context = RecommendationContext(weather="cold", season="cold")
        warm_context = RecommendationContext(weather="hot", season="warm")
        formal_context = RecommendationContext(occasion="formal")
        
        base_item = next(item for item in self.test_items if item.item_id == "blue_shirt_1")
        
        # Test cold weather filtering
        cold_recommendations = self.engine.find_complementary_items(
            base_item,
            context=cold_context,
            top_k=10
        )
        
        # Should not include summer items
        for rec in cold_recommendations:
            self.assertNotEqual(rec.item.season, "warm")
        
        # Test formal occasion filtering
        formal_recommendations = self.engine.find_complementary_items(
            base_item,
            context=formal_context,
            top_k=10
        )
        
        # Should prefer formal/smart-casual items
        for rec in formal_recommendations:
            self.assertIn(rec.item.formality, ["formal", "smart-casual"])
        
        print(f"✅ Context filtering: Cold context: {len(cold_recommendations)} items, Formal context: {len(formal_recommendations)} items")
    
    def test_score_outfit(self):
        """Test outfit scoring"""
        # Create test outfits
        good_outfit = [
            next(item for item in self.test_items if item.item_id == "blue_shirt_1"),
            next(item for item in self.test_items if item.item_id == "dark_jeans_1"),
            next(item for item in self.test_items if item.item_id == "white_sneakers_1")
        ]
        
        poor_outfit = [
            next(item for item in self.test_items if item.item_id == "blue_shirt_1"),
            next(item for item in self.test_items if item.item_id == "navy_shirt_1")  # Two tops
        ]
        
        good_score = self.engine.score_outfit(good_outfit)
        poor_score = self.engine.score_outfit(poor_outfit)
        
        self.assertGreaterEqual(good_score, 0.0)
        self.assertLessEqual(good_score, 1.0)
        self.assertGreaterEqual(poor_score, 0.0)
        self.assertLessEqual(poor_score, 1.0)
        
        # Good outfit should score higher than poor outfit
        self.assertGreater(good_score, poor_score)
        
        print(f"✅ Score outfit: Good outfit: {good_score:.3f}, Poor outfit: {poor_score:.3f}")
    
    def test_get_recommendations_for_user(self):
        """Test getting personalized recommendations for user"""
        recommendations = self.engine.get_recommendations_for_user(
            user_id="test_user",
            recommendation_type="daily",
            limit=5
        )
        
        # Check structure
        self.assertIn('similar_items', recommendations)
        self.assertIn('complementary_items', recommendations)
        self.assertIn('complete_outfits', recommendations)
        
        # Check that recommendations are lists of Recommendation objects
        for category, recs in recommendations.items():
            self.assertIsInstance(recs, list)
            for rec in recs:
                if rec:  # Some categories might be empty
                    self.assertIsInstance(rec, Recommendation)
        
        total_recommendations = sum(len(recs) for recs in recommendations.values())
        print(f"✅ User recommendations: Generated {total_recommendations} total recommendations")


class TestRecommendationEngineWithPolyvoreData(unittest.TestCase):
    """Test recommendation engine with real Polyvore data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class with real database data"""
        cls.production_db_path = Path("data/closetgpt.db")
        if not cls.production_db_path.exists():
            raise unittest.SkipTest("Production database not found - skipping Polyvore tests")
        
        cls.manager = DataManager(str(cls.production_db_path))
        cls.engine = RecommendationEngine(data_manager=cls.manager)
        
        print("🗃️ Testing recommendation engine with Polyvore database")
    
    def test_polyvore_similar_items_recommendation(self):
        """Test similar items recommendation with Polyvore data"""
        # Load a small sample for testing
        sample_items = self.engine.load_items_from_database(limit=10)
        
        if len(sample_items) < 2:
            self.skipTest("Need at least 2 Polyvore items")
        
        target_item = sample_items[0]
        
        similar_recommendations = self.engine.find_similar_items(
            target_item,
            candidate_items=sample_items[1:],
            top_k=3
        )
        
        self.assertLessEqual(len(similar_recommendations), 3)
        
        for rec in similar_recommendations:
            self.assertEqual(rec.recommendation_type, "similar")
            self.assertNotEqual(rec.item.item_id, target_item.item_id)
        
        print(f"✅ Polyvore similar items: Found {len(similar_recommendations)} recommendations")
    
    def test_polyvore_complementary_items_recommendation(self):
        """Test complementary items recommendation with Polyvore data"""
        # Load a sample for testing
        sample_items = self.engine.load_items_from_database(limit=20)
        
        if len(sample_items) < 5:
            self.skipTest("Need at least 5 Polyvore items")
        
        target_item = sample_items[0]
        
        complementary_recommendations = self.engine.find_complementary_items(
            target_item,
            candidate_items=sample_items[1:],
            top_k=5
        )
        
        self.assertLessEqual(len(complementary_recommendations), 5)
        
        for rec in complementary_recommendations:
            self.assertEqual(rec.recommendation_type, "complementary")
            # Most complementary items should be different category
            # (unless target is outerwear/accessory which can layer)
            if target_item.category not in ["outerwear", "accessory"]:
                self.assertNotEqual(rec.item.category, target_item.category)
        
        print(f"✅ Polyvore complementary items: Found {len(complementary_recommendations)} recommendations")
    
    def test_polyvore_complete_outfit_generation(self):
        """Test complete outfit generation with Polyvore data"""
        # Load a sample for testing
        sample_items = self.engine.load_items_from_database(limit=30)
        
        if len(sample_items) < 10:
            self.skipTest("Need at least 10 Polyvore items")
        
        base_item = sample_items[0]
        
        outfit, score = self.engine.generate_complete_outfit(
            base_item,
            candidate_items=sample_items[1:]
        )
        
        self.assertGreaterEqual(len(outfit), 1)
        self.assertIn(base_item, outfit)
        self.assertGreaterEqual(score, 0.0)
        
        print(f"✅ Polyvore complete outfit: Generated {len(outfit)}-item outfit with score {score:.3f}")
    
    def test_polyvore_recommendation_performance(self):
        """Test recommendation performance with Polyvore data"""
        import time
        
        # Load a moderate sample
        sample_items = self.engine.load_items_from_database(limit=100)
        
        if len(sample_items) < 50:
            self.skipTest("Need at least 50 Polyvore items for performance testing")
        
        target_item = sample_items[0]
        candidates = sample_items[1:50]  # 49 candidates
        
        # Test similar items performance
        start_time = time.time()
        similar_recs = self.engine.find_similar_items(target_item, candidates, top_k=10)
        similar_time = time.time() - start_time
        
        # Test complementary items performance
        start_time = time.time()
        comp_recs = self.engine.find_complementary_items(target_item, candidates, top_k=10)
        comp_time = time.time() - start_time
        
        # Test outfit generation performance
        start_time = time.time()
        outfit, score = self.engine.generate_complete_outfit(target_item, candidates)
        outfit_time = time.time() - start_time
        
        # Performance should be reasonable
        self.assertLess(similar_time, 3.0)
        self.assertLess(comp_time, 3.0)
        self.assertLess(outfit_time, 5.0)
        
        print(f"✅ Polyvore performance: Similar: {similar_time:.2f}s, Complementary: {comp_time:.2f}s, Outfit: {outfit_time:.2f}s")


class TestRecommendationEngineEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = RecommendationEngine()  # No database manager
    
    def test_no_database_manager(self):
        """Test engine behavior without database manager"""
        target_item = Item("test_001", "top", "blue", "shirt", embedding=np.random.rand(512))
        target_item.embedding = target_item.embedding / np.linalg.norm(target_item.embedding)
        
        # Should return empty list when trying to load from database
        items = self.engine.load_items_from_database()
        self.assertEqual(len(items), 0)
        
        # Should work with provided candidate items
        candidates = [
            Item("test_002", "bottom", "black", "jeans", embedding=np.random.rand(512)),
            Item("test_003", "shoes", "white", "sneakers", embedding=np.random.rand(512))
        ]
        
        for item in candidates:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        similar_recs = self.engine.find_similar_items(target_item, candidates)
        comp_recs = self.engine.find_complementary_items(target_item, candidates)
        
        # Should work with provided candidates
        self.assertIsInstance(similar_recs, list)
        self.assertIsInstance(comp_recs, list)
        
        print("✅ No database manager: Engine handles missing database gracefully")
    
    def test_empty_candidate_lists(self):
        """Test behavior with empty candidate lists"""
        target_item = Item("test_001", "top", "blue", "shirt", embedding=np.random.rand(512))
        target_item.embedding = target_item.embedding / np.linalg.norm(target_item.embedding)
        
        # Test with empty candidates
        similar_recs = self.engine.find_similar_items(target_item, [])
        comp_recs = self.engine.find_complementary_items(target_item, [])
        outfit_recs = self.engine.complete_outfit([target_item], [])
        
        self.assertEqual(len(similar_recs), 0)
        self.assertEqual(len(comp_recs), 0)
        self.assertEqual(len(outfit_recs), 0)
        
        print("✅ Empty candidates: Engine handles empty lists correctly")
    
    def test_invalid_context(self):
        """Test behavior with invalid context"""
        target_item = Item("test_001", "top", "blue", "shirt", embedding=np.random.rand(512))
        candidate = Item("test_002", "bottom", "black", "jeans", embedding=np.random.rand(512))
        
        for item in [target_item, candidate]:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        # Context with invalid values (should be normalized in RecommendationContext)
        context = RecommendationContext(
            weather="invalid_weather",
            occasion="invalid_occasion"
        )
        
        # Should still work (invalid values normalized to None)
        comp_recs = self.engine.find_complementary_items(
            target_item, 
            [candidate], 
            context=context
        )
        
        self.assertIsInstance(comp_recs, list)
        
        print("✅ Invalid context: Engine handles invalid context values")
    
    def test_items_without_embeddings(self):
        """Test behavior with items missing embeddings"""
        target_item = Item("test_001", "top", "blue", "shirt")  # No embedding
        candidate = Item("test_002", "bottom", "black", "jeans")  # No embedding
        
        # Should handle missing embeddings gracefully
        similar_recs = self.engine.find_similar_items(target_item, [candidate])
        comp_recs = self.engine.find_complementary_items(target_item, [candidate])
        
        # Might return empty or low-scored results
        self.assertIsInstance(similar_recs, list)
        self.assertIsInstance(comp_recs, list)
        
        print("✅ Missing embeddings: Engine handles missing embeddings gracefully")
    
    def test_single_item_operations(self):
        """Test operations with single items"""
        single_item = Item("single_001", "top", "blue", "shirt", embedding=np.random.rand(512))
        single_item.embedding = single_item.embedding / np.linalg.norm(single_item.embedding)
        
        # Test outfit scoring with single item
        score = self.engine.score_outfit([single_item])
        self.assertEqual(score, 0.0)  # Single item should score 0
        
        # Test outfit generation with single base item and no candidates
        outfit, score = self.engine.generate_complete_outfit(single_item, [])
        self.assertEqual(len(outfit), 1)  # Should return just the base item
        self.assertEqual(outfit[0], single_item)
        
        print("✅ Single item operations: Engine handles single items correctly")


if __name__ == '__main__':
    unittest.main()