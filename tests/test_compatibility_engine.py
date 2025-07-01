"""
Test cases for Compatibility Engine
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
from compatibility_engine import CompatibilityEngine
from data_manager import DataManager


class TestCompatibilityEngine(unittest.TestCase):
    """Test cases for CompatibilityEngine"""
    
    def setUp(self):
        """Set up test data"""
        self.engine = CompatibilityEngine()
        
        # Create test items with normalized embeddings
        self.white_shirt = Item("shirt_001", "top", "white", "shirt", "smart-casual", embedding=np.random.rand(512))
        self.blue_jeans = Item("jeans_001", "bottom", "blue", "jeans", "casual", embedding=np.random.rand(512))
        self.brown_shoes = Item("shoes_001", "shoes", "brown", "loafers", "smart-casual", embedding=np.random.rand(512))
        self.red_shirt = Item("shirt_002", "top", "red", "shirt", "casual", embedding=np.random.rand(512))
        self.black_dress = Item("dress_001", "top", "black", "dress", "formal", embedding=np.random.rand(512))
        self.navy_blazer = Item("blazer_001", "outerwear", "navy", "blazer", "formal", embedding=np.random.rand(512))
        
        # Normalize embeddings
        for item in [self.white_shirt, self.blue_jeans, self.brown_shoes, self.red_shirt, self.black_dress, self.navy_blazer]:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    def test_engine_initialization(self):
        """Test compatibility engine initialization"""
        self.assertIsInstance(self.engine, CompatibilityEngine)
        self.assertIsInstance(self.engine.color_rules, dict)
        self.assertIsInstance(self.engine.style_rules, dict)
        self.assertIsInstance(self.engine.category_rules, dict)
        
        # Check that rules are loaded
        self.assertIn('white', self.engine.color_rules)
        self.assertIn('casual', self.engine.style_rules)
        self.assertIn('top', self.engine.category_rules)
        
        print("✅ Engine initialization: CompatibilityEngine initialized with rules")
    
    def test_calculate_compatibility_cross_category(self):
        """Test compatibility calculation between different categories"""
        # Test classic combination: white shirt + blue jeans
        compatibility = self.engine.calculate_compatibility(self.white_shirt, self.blue_jeans)
        
        self.assertIsInstance(compatibility, float)
        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
        
        # White and blue should have good compatibility
        self.assertGreater(compatibility, 0.5)
        
        print(f"✅ Cross-category compatibility: White shirt + Blue jeans: {compatibility:.3f}")
    
    def test_calculate_compatibility_same_category(self):
        """Test compatibility calculation within same category"""
        # Two shirts should have low compatibility (usually don't wear two tops together)
        compatibility = self.engine.calculate_compatibility(self.white_shirt, self.red_shirt)
        
        self.assertLessEqual(compatibility, 0.2)  # Should be low for same category
        
        print(f"✅ Same-category compatibility: White shirt + Red shirt: {compatibility:.3f}")
    
    def test_calculate_compatibility_same_item(self):
        """Test compatibility calculation for same item"""
        # Same item should have 0 compatibility (can't wear same item twice)
        compatibility = self.engine.calculate_compatibility(self.white_shirt, self.white_shirt)
        self.assertEqual(compatibility, 0.0)
        
        print("✅ Same item compatibility: Returns 0.0 as expected")
    
    def test_color_compatibility_rules(self):
        """Test color compatibility calculation"""
        # Test exact color match (should be high)
        white_item1 = Item("white_001", "top", "white", "shirt")
        white_item2 = Item("white_002", "bottom", "white", "pants")
        color_comp = self.engine._calculate_color_compatibility(white_item1, white_item2)
        self.assertGreater(color_comp, 0.8)
        
        # Test known good combination (white + blue)
        white_item = Item("white_003", "top", "white", "shirt")
        blue_item = Item("blue_001", "bottom", "blue", "jeans")
        color_comp_good = self.engine._calculate_color_compatibility(white_item, blue_item)
        self.assertGreater(color_comp_good, 0.8)
        
        # Test missing color (should return neutral)
        no_color_item = Item("no_color", "top")
        color_comp_missing = self.engine._calculate_color_compatibility(white_item, no_color_item)
        self.assertEqual(color_comp_missing, 0.6)
        
        print("✅ Color compatibility: Color rules working correctly")
    
    def test_style_compatibility_rules(self):
        """Test style/formality compatibility calculation"""
        # Test exact formality match
        casual_item1 = Item("casual_001", "top", formality="casual")
        casual_item2 = Item("casual_002", "bottom", formality="casual")
        style_comp = self.engine._calculate_style_compatibility(casual_item1, casual_item2)
        self.assertEqual(style_comp, 0.9)  # Exact match
        
        # Test compatible formality levels (smart-casual with formal)
        smart_casual_item = Item("smart_001", "top", formality="smart-casual")
        formal_item = Item("formal_001", "bottom", formality="formal")
        style_comp_compat = self.engine._calculate_style_compatibility(smart_casual_item, formal_item)
        self.assertEqual(style_comp_compat, 0.8)
        
        # Test incompatible formality levels (casual with formal)
        casual_item = Item("casual_003", "top", formality="casual")
        style_comp_incompat = self.engine._calculate_style_compatibility(casual_item, formal_item)
        self.assertEqual(style_comp_incompat, 0.3)
        
        print("✅ Style compatibility: Formality rules working correctly")
    
    def test_season_compatibility_rules(self):
        """Test season compatibility calculation"""
        # Test 'all' season compatibility
        all_season_item = Item("all_001", "top", season="all")
        warm_item = Item("warm_001", "bottom", season="warm")
        season_comp = self.engine._calculate_season_compatibility(all_season_item, warm_item)
        self.assertEqual(season_comp, 1.0)
        
        # Test same season compatibility
        warm_item1 = Item("warm_002", "top", season="warm")
        warm_item2 = Item("warm_003", "bottom", season="warm")
        season_comp_same = self.engine._calculate_season_compatibility(warm_item1, warm_item2)
        self.assertEqual(season_comp_same, 1.0)
        
        # Test different season compatibility
        cold_item = Item("cold_001", "top", season="cold")
        season_comp_diff = self.engine._calculate_season_compatibility(warm_item, cold_item)
        self.assertEqual(season_comp_diff, 0.3)
        
        print("✅ Season compatibility: Season rules working correctly")
    
    def test_pattern_compatibility_rules(self):
        """Test pattern compatibility calculation"""
        # Test both items with patterns (should be lower compatibility)
        striped_item = Item("striped_001", "top", style="striped_shirt")
        plaid_item = Item("plaid_001", "bottom", style="plaid_pants")
        pattern_comp = self.engine._calculate_pattern_compatibility(striped_item, plaid_item)
        self.assertEqual(pattern_comp, 0.4)
        
        # Test one patterned, one solid (should be higher)
        solid_item = Item("solid_001", "bottom", style="pants")
        pattern_comp_mixed = self.engine._calculate_pattern_compatibility(striped_item, solid_item)
        self.assertEqual(pattern_comp_mixed, 0.8)
        
        # Test both solid (should be neutral)
        solid_item2 = Item("solid_002", "top", style="shirt")
        pattern_comp_solid = self.engine._calculate_pattern_compatibility(solid_item, solid_item2)
        self.assertEqual(pattern_comp_solid, 0.7)
        
        print("✅ Pattern compatibility: Pattern rules working correctly")
    
    def test_find_complementary_items(self):
        """Test finding complementary items"""
        candidates = [self.blue_jeans, self.brown_shoes, self.red_shirt, self.navy_blazer]
        
        complementary_items = self.engine.find_complementary_items(
            self.white_shirt,
            candidates,
            top_k=3,
            min_compatibility=0.4
        )
        
        # Should find items from different categories
        self.assertLessEqual(len(complementary_items), 3)
        
        # Check return format
        for item, score in complementary_items:
            self.assertIsInstance(item, Item)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.4)
            self.assertNotEqual(item.category, self.white_shirt.category)  # Different category
        
        # Results should be sorted by compatibility (descending)
        if len(complementary_items) > 1:
            scores = [score for _, score in complementary_items]
            self.assertEqual(scores, sorted(scores, reverse=True))
        
        print(f"✅ Find complementary items: Found {len(complementary_items)} complementary items")
        for item, score in complementary_items:
            print(f"   {item.item_id} ({item.category}): {score:.3f}")
    
    def test_find_complementary_items_with_exclusions(self):
        """Test finding complementary items with category exclusions"""
        candidates = [self.blue_jeans, self.brown_shoes, self.red_shirt, self.navy_blazer]
        
        # Exclude shoes category
        complementary_items = self.engine.find_complementary_items(
            self.white_shirt,
            candidates,
            top_k=5,
            exclude_categories={'top', 'shoes'}
        )
        
        # Should not include any shoes or tops
        for item, score in complementary_items:
            self.assertNotIn(item.category, {'top', 'shoes'})
        
        print(f"✅ Complementary items with exclusions: Found {len(complementary_items)} items (no tops/shoes)")
    
    def test_score_outfit(self):
        """Test outfit scoring"""
        # Test good outfit
        good_outfit = [self.white_shirt, self.blue_jeans, self.brown_shoes]
        good_score = self.engine.score_outfit(good_outfit)
        
        self.assertGreaterEqual(good_score, 0.0)
        self.assertLessEqual(good_score, 1.0)
        self.assertGreater(good_score, 0.4)  # Should be reasonably good
        
        # Test poor outfit (same category items)
        poor_outfit = [self.white_shirt, self.red_shirt]
        poor_score = self.engine.score_outfit(poor_outfit)
        
        self.assertLess(poor_score, good_score)  # Should be worse than good outfit
        
        # Test single item outfit
        single_score = self.engine.score_outfit([self.white_shirt])
        self.assertEqual(single_score, 0.0)
        
        # Test empty outfit
        empty_score = self.engine.score_outfit([])
        self.assertEqual(empty_score, 0.0)
        
        print(f"✅ Outfit scoring: Good outfit: {good_score:.3f}, Poor outfit: {poor_score:.3f}")
    
    def test_outfit_completeness_bonus(self):
        """Test outfit completeness bonus in scoring"""
        # Outfit with top + bottom should get bonus
        complete_outfit = [self.white_shirt, self.blue_jeans]
        complete_score = self.engine.score_outfit(complete_outfit)
        
        # Same items but calculate base score manually
        base_compatibility = self.engine.calculate_compatibility(self.white_shirt, self.blue_jeans)
        
        # Complete outfit should have bonus applied
        self.assertGreaterEqual(complete_score, base_compatibility)
        
        print(f"✅ Outfit completeness bonus: Base compatibility: {base_compatibility:.3f}, Outfit score: {complete_score:.3f}")
    
    def test_suggest_outfit_completion(self):
        """Test outfit completion suggestions"""
        partial_outfit = [self.white_shirt]
        candidates = [self.blue_jeans, self.brown_shoes, self.red_shirt, self.navy_blazer]
        
        suggestions = self.engine.suggest_outfit_completion(
            partial_outfit,
            candidates,
            target_categories=['bottom', 'shoes']
        )
        
        # Should suggest items from target categories only
        suggested_categories = {item.category for item, _ in suggestions}
        self.assertTrue(suggested_categories.issubset({'bottom', 'shoes'}))
        
        # Should have compatibility scores
        for item, score in suggestions:
            self.assertGreater(score, 0.4)  # Above threshold
        
        print(f"✅ Outfit completion: {len(suggestions)} suggestions for completing white shirt")
        for item, score in suggestions:
            print(f"   {item.item_id} ({item.category}): {score:.3f}")
    
    def test_suggest_outfit_completion_auto_categories(self):
        """Test outfit completion with automatic category detection"""
        partial_outfit = [self.white_shirt]  # top
        candidates = [self.blue_jeans, self.brown_shoes, self.navy_blazer]
        
        # Don't specify target_categories, should auto-detect missing ones
        suggestions = self.engine.suggest_outfit_completion(partial_outfit, candidates)
        
        # Should suggest bottom, shoes, outerwear (missing categories)
        suggested_categories = {item.category for item, _ in suggestions}
        expected_categories = {'bottom', 'shoes', 'outerwear'}
        self.assertTrue(suggested_categories.issubset(expected_categories))
        
        print(f"✅ Auto-category completion: Suggested {len(suggestions)} items from missing categories")


class TestCompatibilityEngineWithPolyvoreData(unittest.TestCase):
    """Test compatibility engine with real Polyvore data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class with real database data"""
        cls.production_db_path = Path("data/closetgpt.db")
        if not cls.production_db_path.exists():
            raise unittest.SkipTest("Production database not found - skipping Polyvore tests")
        
        cls.manager = DataManager(str(cls.production_db_path))
        cls.engine = CompatibilityEngine()
        
        # Load a sample of items from database
        cls.polyvore_items = cls._load_sample_items(100)  # Load 100 items for testing
        
        if len(cls.polyvore_items) < 20:
            raise unittest.SkipTest("Insufficient Polyvore items for testing")
        
        print(f"🗃️ Loaded {len(cls.polyvore_items)} Polyvore items for compatibility testing")
    
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
    
    def test_polyvore_compatibility_calculation(self):
        """Test compatibility calculation with real Polyvore data"""
        if len(self.polyvore_items) < 2:
            self.skipTest("Need at least 2 Polyvore items")
        
        item1 = self.polyvore_items[0]
        item2 = self.polyvore_items[1]
        
        compatibility = self.engine.calculate_compatibility(item1, item2)
        
        self.assertIsInstance(compatibility, float)
        self.assertGreaterEqual(compatibility, 0.0)
        self.assertLessEqual(compatibility, 1.0)
        
        print(f"✅ Polyvore compatibility: {item1.item_id} vs {item2.item_id}: {compatibility:.3f}")
    
    def test_polyvore_find_complementary_items(self):
        """Test finding complementary items with Polyvore data"""
        if len(self.polyvore_items) < 10:
            self.skipTest("Need at least 10 Polyvore items")
        
        target_item = self.polyvore_items[0]
        candidates = self.polyvore_items[1:20]  # Use next 19 as candidates
        
        complementary_items = self.engine.find_complementary_items(
            target_item,
            candidates,
            top_k=5,
            min_compatibility=0.3
        )
        
        self.assertLessEqual(len(complementary_items), 5)
        
        # Check that results exclude same category (unless it's outerwear/accessories)
        for item, score in complementary_items:
            if target_item.category not in {'outerwear', 'accessory'}:
                self.assertNotEqual(item.category, target_item.category)
            self.assertGreaterEqual(score, 0.3)
        
        print(f"✅ Polyvore complementary items: Found {len(complementary_items)} complementary items")
        for item, score in complementary_items[:3]:  # Show top 3
            print(f"   {item.item_id} ({item.category}): {score:.3f}")
    
    def test_polyvore_outfit_scoring(self):
        """Test outfit scoring with Polyvore combinations"""
        if len(self.polyvore_items) < 3:
            self.skipTest("Need at least 3 Polyvore items")
        
        # Create test outfits
        outfit1 = self.polyvore_items[:2]
        outfit2 = self.polyvore_items[:3]
        outfit3 = self.polyvore_items[:4]
        
        score1 = self.engine.score_outfit(outfit1)
        score2 = self.engine.score_outfit(outfit2)
        score3 = self.engine.score_outfit(outfit3)
        
        # All scores should be valid
        for score in [score1, score2, score3]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        print(f"✅ Polyvore outfit scoring: 2-item: {score1:.3f}, 3-item: {score2:.3f}, 4-item: {score3:.3f}")
    
    def test_polyvore_category_compatibility_patterns(self):
        """Test compatibility patterns across different categories"""
        # Group items by category
        category_items = {}
        for item in self.polyvore_items:
            if item.category not in category_items:
                category_items[item.category] = []
            category_items[item.category].append(item)
        
        # Test cross-category compatibility
        results = {}
        categories = list(category_items.keys())
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                if len(category_items[cat1]) > 0 and len(category_items[cat2]) > 0:
                    item1 = category_items[cat1][0]
                    item2 = category_items[cat2][0]
                    
                    compatibility = self.engine.calculate_compatibility(item1, item2)
                    results[f"{cat1}-{cat2}"] = compatibility
        
        print(f"✅ Polyvore category patterns: Tested {len(results)} category combinations")
        for combo, comp in sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {combo}: {comp:.3f}")
    
    def test_polyvore_performance_with_large_dataset(self):
        """Test performance with larger Polyvore dataset"""
        if len(self.polyvore_items) < 50:
            self.skipTest("Need at least 50 Polyvore items")
        
        import time
        
        target_item = self.polyvore_items[0]
        candidates = self.polyvore_items[1:50]  # 49 candidates
        
        # Test performance of finding complementary items
        start_time = time.time()
        complementary_items = self.engine.find_complementary_items(
            target_item,
            candidates,
            top_k=10,
            min_compatibility=0.2
        )
        search_time = time.time() - start_time
        
        # Test performance of outfit scoring
        start_time = time.time()
        outfit_scores = []
        for i in range(min(10, len(candidates))):
            outfit = [target_item, candidates[i]]
            score = self.engine.score_outfit(outfit)
            outfit_scores.append(score)
        scoring_time = time.time() - start_time
        
        # Performance should be reasonable
        self.assertLess(search_time, 5.0)  # Less than 5 seconds for 49 items
        self.assertLess(scoring_time, 2.0)  # Less than 2 seconds for 10 outfits
        
        print(f"✅ Polyvore performance: Search 49 items: {search_time:.2f}s, Score 10 outfits: {scoring_time:.2f}s")


class TestCompatibilityEngineIntegration(unittest.TestCase):
    """Integration tests for compatibility engine"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_compatibility.db"
        self.manager = DataManager(str(self.db_path))
        self.engine = CompatibilityEngine()
        
        # Create and add test items to database
        self.test_items = self._create_test_items()
        self._add_items_to_database()
    
    def tearDown(self):
        """Clean up after integration tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_items(self):
        """Create a diverse set of test items for compatibility testing"""
        items = [
            # Casual outfit components
            Item("cas_shirt", "top", "white", "tshirt", "casual", season="all", embedding=np.random.rand(512)),
            Item("cas_jeans", "bottom", "blue", "jeans", "casual", season="all", embedding=np.random.rand(512)),
            Item("cas_sneakers", "shoes", "white", "sneakers", "casual", season="all", embedding=np.random.rand(512)),
            
            # Formal outfit components
            Item("for_shirt", "top", "white", "dress_shirt", "formal", season="all", embedding=np.random.rand(512)),
            Item("for_pants", "bottom", "black", "dress_pants", "formal", season="all", embedding=np.random.rand(512)),
            Item("for_shoes", "shoes", "black", "dress_shoes", "formal", season="all", embedding=np.random.rand(512)),
            
            # Smart-casual components
            Item("sc_shirt", "top", "blue", "button_down", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("sc_chinos", "bottom", "khaki", "chinos", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("sc_loafers", "shoes", "brown", "loafers", "smart-casual", season="all", embedding=np.random.rand(512)),
            
            # Outerwear
            Item("blazer", "outerwear", "navy", "blazer", "formal", season="all", embedding=np.random.rand(512)),
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
    
    def test_compatibility_engine_with_database_integration(self):
        """Test compatibility engine integrated with database"""
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
        
        # Test compatibility engine with database items
        target_item = next(item for item in items if item.item_id == "cas_shirt")
        candidates = [item for item in items if item.item_id != "cas_shirt"]
        
        complementary_items = self.engine.find_complementary_items(target_item, candidates, top_k=5)
        
        self.assertLessEqual(len(complementary_items), 5)
        
        print(f"✅ Database integration: Found {len(complementary_items)} complementary items from database")
    
    def test_formality_matching_integration(self):
        """Test that formality matching works correctly in integration"""
        # Load items from database
        db_items = self.manager.get_items()
        item_ids = [item['item_id'] for item in db_items]
        embeddings = self.manager.get_embeddings(item_ids)
        
        items = []
        for db_item in db_items:
            item = Item.from_db_record(db_item)
            item.embedding = embeddings[db_item['item_id']]
            items.append(item)
        
        # Test formal item with all candidates
        formal_shirt = next(item for item in items if item.item_id == "for_shirt")
        candidates = [item for item in items if item.item_id != "for_shirt"]
        
        complementary_items = self.engine.find_complementary_items(
            formal_shirt, 
            candidates, 
            top_k=10,
            min_compatibility=0.3
        )
        
        # Should prefer formal and smart-casual items
        formality_scores = {}
        for item, score in complementary_items:
            if item.formality not in formality_scores:
                formality_scores[item.formality] = []
            formality_scores[item.formality].append(score)
        
        # Calculate average scores by formality
        avg_scores = {}
        for formality, scores in formality_scores.items():
            avg_scores[formality] = np.mean(scores) if scores else 0.0
        
        print(f"✅ Formality matching: Average compatibility scores by formality:")
        for formality, avg_score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {formality}: {avg_score:.3f}")
    
    def test_complete_outfit_generation_integration(self):
        """Test complete outfit generation workflow"""
        # Load all items
        db_items = self.manager.get_items()
        item_ids = [item['item_id'] for item in db_items]
        embeddings = self.manager.get_embeddings(item_ids)
        
        items = []
        for db_item in db_items:
            item = Item.from_db_record(db_item)
            item.embedding = embeddings[db_item['item_id']]
            items.append(item)
        
        # Start with a base item
        base_item = next(item for item in items if item.item_id == "cas_shirt")
        
        # Build a complete casual outfit
        outfit = [base_item]
        candidates = [item for item in items if item.item_id != base_item.item_id]
        
        # Add bottom
        bottom_suggestions = self.engine.suggest_outfit_completion(
            outfit, candidates, target_categories=['bottom']
        )
        if bottom_suggestions:
            outfit.append(bottom_suggestions[0][0])
        
        # Add shoes
        shoe_suggestions = self.engine.suggest_outfit_completion(
            outfit, candidates, target_categories=['shoes']
        )
        if shoe_suggestions:
            outfit.append(shoe_suggestions[0][0])
        
        # Score the complete outfit
        final_score = self.engine.score_outfit(outfit)
        
        self.assertGreaterEqual(len(outfit), 2)  # At least shirt + one other item
        self.assertGreater(final_score, 0.0)
        
        print(f"✅ Complete outfit generation: Generated {len(outfit)}-item outfit with score {final_score:.3f}")
        for item in outfit:
            print(f"   {item.item_id} ({item.category}, {item.formality})")


if __name__ == '__main__':
    unittest.main()