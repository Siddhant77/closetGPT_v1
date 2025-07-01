"""
Test cases for data structures
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "models"))

from data_structures import Item, RecommendationContext, Recommendation, Outfit, create_item_from_db_record, items_to_outfit, filter_items_by_context


class TestItem(unittest.TestCase):
    """Test cases for Item class"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_embedding = np.random.rand(512).astype(np.float32)
        self.sample_embedding = self.sample_embedding / np.linalg.norm(self.sample_embedding)
    
    def test_item_creation(self):
        """Test basic item creation"""
        item = Item(
            item_id="test_001",
            category="top",
            color="blue",
            style="shirt",
            formality="casual"
        )
        
        self.assertEqual(item.item_id, "test_001")
        self.assertEqual(item.category, "top")
        self.assertEqual(item.color, "blue")
        self.assertEqual(item.style, "shirt")
        self.assertEqual(item.formality, "casual")
        self.assertEqual(item.season, "all")  # Default value
        self.assertEqual(item.source, "unknown")  # Default value
        self.assertEqual(item.metadata, {})  # Default value
        
        print("✅ Item creation: Basic item created successfully")
    
    def test_item_validation(self):
        """Test item field validation"""
        # Test invalid category normalization
        item = Item("test_002", "invalid_category")
        self.assertEqual(item.category, "unknown")
        
        # Test invalid formality normalization
        item = Item("test_003", "top", formality="invalid_formality")
        self.assertEqual(item.formality, "casual")
        
        # Test invalid season normalization
        item = Item("test_004", "top", season="invalid_season")
        self.assertEqual(item.season, "all")
        
        print("✅ Item validation: Invalid fields normalized correctly")
    
    def test_item_with_embedding(self):
        """Test item creation with embedding"""
        item = Item(
            item_id="test_005",
            category="top",
            embedding=self.sample_embedding
        )
        
        self.assertIsNotNone(item.embedding)
        np.testing.assert_array_equal(item.embedding, self.sample_embedding)
        
        print("✅ Item with embedding: Embedding stored correctly")
    
    def test_item_serialization(self):
        """Test item to_dict and from_dict methods"""
        original_item = Item(
            item_id="test_006",
            category="bottom",
            color="black",
            style="jeans",
            formality="casual",
            season="all",
            source="personal",
            metadata={"brand": "Levi's", "size": "32"}
        )
        
        # Test to_dict
        item_dict = original_item.to_dict()
        expected_keys = {'item_id', 'category', 'color', 'style', 'formality', 'season', 'source', 'metadata'}
        self.assertEqual(set(item_dict.keys()), expected_keys)
        self.assertNotIn('embedding', item_dict)  # Embedding should be excluded
        
        # Test from_dict
        recreated_item = Item.from_dict(item_dict, self.sample_embedding)
        self.assertEqual(recreated_item.item_id, original_item.item_id)
        self.assertEqual(recreated_item.category, original_item.category)
        self.assertEqual(recreated_item.metadata, original_item.metadata)
        np.testing.assert_array_equal(recreated_item.embedding, self.sample_embedding)
        
        print("✅ Item serialization: to_dict and from_dict working correctly")
    
    def test_item_from_db_record(self):
        """Test creating item from database record"""
        db_record = {
            'item_id': 'db_001',
            'category': 'shoes',
            'color': 'brown',
            'style': 'loafers',
            'formality': 'smart-casual',
            'season': 'all',
            'source': 'polyvore',
            'embedding': self.sample_embedding,
            'metadata': {'price': 150}
        }
        
        item = Item.from_db_record(db_record)
        self.assertEqual(item.item_id, 'db_001')
        self.assertEqual(item.category, 'shoes')
        self.assertEqual(item.color, 'brown')
        self.assertEqual(item.metadata['price'], 150)
        np.testing.assert_array_equal(item.embedding, self.sample_embedding)
        
        print("✅ Item from DB record: Database record converted correctly")
    
    def test_item_string_representation(self):
        """Test item string representation"""
        item = Item("test_007", "top", "red", "dress", "formal")
        str_repr = str(item)
        
        self.assertIn("test_007", str_repr)
        self.assertIn("red", str_repr)
        self.assertIn("dress", str_repr)
        self.assertIn("(top)", str_repr)
        
        print("✅ Item string representation: String format correct")


class TestRecommendationContext(unittest.TestCase):
    """Test cases for RecommendationContext class"""
    
    def test_context_creation(self):
        """Test basic context creation"""
        context = RecommendationContext(
            weather="cold",
            occasion="work",
            season="cold"
        )
        
        self.assertEqual(context.weather, "cold")
        self.assertEqual(context.occasion, "work")
        self.assertEqual(context.season, "cold")
        self.assertEqual(context.user_preferences, {})
        
        print("✅ Context creation: Basic context created successfully")
    
    def test_context_validation(self):
        """Test context field validation"""
        # Test invalid values are normalized to None
        context = RecommendationContext(
            weather="invalid_weather",
            occasion="invalid_occasion",
            season="invalid_season"
        )
        
        self.assertIsNone(context.weather)
        self.assertIsNone(context.occasion)
        self.assertIsNone(context.season)
        
        print("✅ Context validation: Invalid values normalized to None")
    
    def test_context_item_matching(self):
        """Test context matching with items"""
        # Create context
        work_context = RecommendationContext(
            occasion="work",
            season="cold"
        )
        
        # Create test items
        formal_shirt = Item("shirt_001", "top", formality="formal", season="cold")
        casual_shirt = Item("shirt_002", "top", formality="casual", season="warm")
        smart_casual_shirt = Item("shirt_003", "top", formality="smart-casual", season="all")
        
        # Test matching
        self.assertTrue(work_context.matches_item(formal_shirt))
        self.assertFalse(work_context.matches_item(casual_shirt))  # Wrong formality and season
        self.assertTrue(work_context.matches_item(smart_casual_shirt))  # Smart-casual works for work
        
        print("✅ Context item matching: Item-context matching working correctly")
    
    def test_context_serialization(self):
        """Test context serialization"""
        context = RecommendationContext(
            weather="mild",
            occasion="date",
            user_preferences={"favorite_colors": ["blue", "black"]}
        )
        
        context_dict = context.to_dict()
        self.assertEqual(context_dict['weather'], "mild")
        self.assertEqual(context_dict['occasion'], "date")
        self.assertEqual(context_dict['user_preferences']['favorite_colors'], ["blue", "black"])
        
        print("✅ Context serialization: to_dict working correctly")


class TestRecommendation(unittest.TestCase):
    """Test cases for Recommendation class"""
    
    def setUp(self):
        """Set up test data"""
        self.test_item = Item("rec_001", "top", "blue", "shirt")
    
    def test_recommendation_creation(self):
        """Test basic recommendation creation"""
        rec = Recommendation(
            item=self.test_item,
            score=0.85,
            reason="Great color match",
            recommendation_type="complementary"
        )
        
        self.assertEqual(rec.item, self.test_item)
        self.assertEqual(rec.score, 0.85)
        self.assertEqual(rec.reason, "Great color match")
        self.assertEqual(rec.recommendation_type, "complementary")
        self.assertEqual(rec.confidence, 1.0)  # Default value
        self.assertEqual(rec.context_relevance, 1.0)  # Default value
        
        print("✅ Recommendation creation: Basic recommendation created successfully")
    
    def test_recommendation_validation(self):
        """Test recommendation score validation"""
        # Test score clamping
        rec = Recommendation(self.test_item, score=1.5, reason="test", recommendation_type="similar")
        self.assertEqual(rec.score, 1.0)  # Should be clamped to 1.0
        
        rec = Recommendation(self.test_item, score=-0.5, reason="test", recommendation_type="similar")
        self.assertEqual(rec.score, 0.0)  # Should be clamped to 0.0
        
        # Test invalid recommendation type
        rec = Recommendation(self.test_item, score=0.8, reason="test", recommendation_type="invalid_type")
        self.assertEqual(rec.recommendation_type, "similar")  # Should default to "similar"
        
        print("✅ Recommendation validation: Score clamping and type validation working")
    
    def test_recommendation_overall_score(self):
        """Test overall score calculation"""
        rec = Recommendation(
            item=self.test_item,
            score=0.8,
            reason="test",
            recommendation_type="similar",
            confidence=0.9,
            context_relevance=0.7
        )
        
        expected_overall = 0.8 * 0.9 * 0.7  # score * confidence * context_relevance
        self.assertAlmostEqual(rec.overall_score(), expected_overall, places=3)
        
        print("✅ Recommendation overall score: Calculation working correctly")
    
    def test_recommendation_serialization(self):
        """Test recommendation serialization"""
        rec = Recommendation(
            item=self.test_item,
            score=0.75,
            reason="Style similarity",
            recommendation_type="similar",
            confidence=0.8
        )
        
        rec_dict = rec.to_dict()
        self.assertIn('item', rec_dict)
        self.assertIn('score', rec_dict)
        self.assertIn('overall_score', rec_dict)
        self.assertEqual(rec_dict['recommendation_type'], "similar")
        
        print("✅ Recommendation serialization: to_dict working correctly")


class TestOutfit(unittest.TestCase):
    """Test cases for Outfit class"""
    
    def setUp(self):
        """Set up test data"""
        self.shirt = Item("shirt_001", "top", "white", "shirt", "smart-casual")
        self.pants = Item("pants_001", "bottom", "navy", "chinos", "smart-casual")
        self.shoes = Item("shoes_001", "shoes", "brown", "loafers", "smart-casual")
        self.jacket = Item("jacket_001", "outerwear", "gray", "blazer", "formal")
    
    def test_outfit_creation(self):
        """Test basic outfit creation"""
        outfit = Outfit(
            outfit_id="outfit_001",
            items=[self.shirt, self.pants, self.shoes]
        )
        
        self.assertEqual(outfit.outfit_id, "outfit_001")
        self.assertEqual(len(outfit.items), 3)
        self.assertEqual(outfit.source, "generated")  # Default value
        self.assertEqual(outfit.wear_count, 0)  # Default value
        
        print("✅ Outfit creation: Basic outfit created successfully")
    
    def test_outfit_validation(self):
        """Test outfit validation"""
        # Test empty outfit (should raise error)
        with self.assertRaises(ValueError):
            Outfit("empty_outfit", [])
        
        # Test duplicate item removal
        outfit = Outfit("dup_outfit", [self.shirt, self.shirt, self.pants])
        self.assertEqual(len(outfit.items), 2)  # Duplicate should be removed
        
        print("✅ Outfit validation: Empty outfit rejection and duplicate removal working")
    
    def test_outfit_properties(self):
        """Test outfit computed properties"""
        outfit = Outfit("prop_outfit", [self.shirt, self.pants, self.shoes])
        
        # Test categories
        self.assertEqual(set(outfit.categories), {"top", "bottom", "shoes"})
        self.assertEqual(outfit.category_set, {"top", "bottom", "shoes"})
        
        # Test completeness
        self.assertTrue(outfit.is_complete)  # Has top + bottom
        
        # Test dominant formality
        self.assertEqual(outfit.dominant_formality, "smart-casual")
        
        # Test color palette
        expected_colors = ["white", "navy", "brown"]
        self.assertEqual(outfit.color_palette, expected_colors)
        
        print("✅ Outfit properties: Computed properties working correctly")
    
    def test_outfit_item_management(self):
        """Test adding, removing, and replacing items"""
        outfit = Outfit("mgmt_outfit", [self.shirt, self.pants])
        
        # Test adding item
        self.assertTrue(outfit.add_item(self.shoes))
        self.assertEqual(len(outfit.items), 3)
        
        # Test adding duplicate (should fail)
        self.assertFalse(outfit.add_item(self.shoes))
        self.assertEqual(len(outfit.items), 3)
        
        # Test removing item
        self.assertTrue(outfit.remove_item("shoes_001"))
        self.assertEqual(len(outfit.items), 2)
        
        # Test removing non-existent item
        self.assertFalse(outfit.remove_item("non_existent"))
        
        # Test getting item by category
        top_item = outfit.get_item_by_category("top")
        self.assertEqual(top_item.item_id, "shirt_001")
        
        # Test replacing item
        self.assertTrue(outfit.replace_item("shirt_001", self.jacket))
        replaced_item = outfit.get_item_by_category("outerwear")
        self.assertEqual(replaced_item.item_id, "jacket_001")
        
        print("✅ Outfit item management: Add/remove/replace operations working")
    
    def test_outfit_wear_tracking(self):
        """Test wear count and date tracking"""
        outfit = Outfit("wear_outfit", [self.shirt, self.pants])
        
        # Initial state
        self.assertEqual(outfit.wear_count, 0)
        self.assertIsNone(outfit.last_worn)
        
        # Record wear
        wear_date = datetime.now()
        outfit.record_wear(wear_date)
        
        self.assertEqual(outfit.wear_count, 1)
        self.assertEqual(outfit.last_worn, wear_date)
        
        # Record another wear (should increment)
        outfit.record_wear()
        self.assertEqual(outfit.wear_count, 2)
        
        print("✅ Outfit wear tracking: Wear count and date tracking working")
    
    def test_outfit_serialization(self):
        """Test outfit serialization"""
        context = RecommendationContext(weather="mild", occasion="work")
        outfit = Outfit(
            outfit_id="serial_outfit",
            items=[self.shirt, self.pants],
            compatibility_score=0.85,
            context=context,
            tags=["work", "smart-casual"],
            user_rating=4.5
        )
        
        # Test to_dict
        outfit_dict = outfit.to_dict()
        self.assertEqual(outfit_dict['outfit_id'], "serial_outfit")
        self.assertEqual(len(outfit_dict['items']), 2)
        self.assertEqual(outfit_dict['compatibility_score'], 0.85)
        self.assertIn('context', outfit_dict)
        self.assertEqual(outfit_dict['tags'], ["work", "smart-casual"])
        self.assertEqual(outfit_dict['user_rating'], 4.5)
        
        # Test from_dict
        recreated_outfit = Outfit.from_dict(outfit_dict)
        self.assertEqual(recreated_outfit.outfit_id, "serial_outfit")
        self.assertEqual(len(recreated_outfit.items), 2)
        self.assertEqual(recreated_outfit.compatibility_score, 0.85)
        self.assertEqual(recreated_outfit.context.weather, "mild")
        
        print("✅ Outfit serialization: to_dict and from_dict working correctly")
    
    def test_outfit_string_representation(self):
        """Test outfit string representation"""
        outfit = Outfit("str_outfit", [self.shirt, self.pants], compatibility_score=0.9)
        str_repr = str(outfit)
        
        self.assertIn("str_outfit", str_repr)
        self.assertIn("top(white)", str_repr)
        self.assertIn("bottom(navy)", str_repr)
        self.assertIn("0.90", str_repr)
        
        print("✅ Outfit string representation: String format correct")


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test data"""
        self.db_record = {
            'item_id': 'util_001',
            'category': 'top',
            'color': 'green',
            'style': 'sweater',
            'formality': 'casual',
            'season': 'cold',
            'source': 'personal',
            'embedding': np.random.rand(512),
            'metadata': {'material': 'wool'}
        }
        
        self.items = [
            Item("item_001", "top", "blue", formality="casual", season="all"),
            Item("item_002", "bottom", "black", formality="formal", season="cold"),
            Item("item_003", "shoes", "brown", formality="smart-casual", season="all")
        ]
    
    def test_create_item_from_db_record(self):
        """Test utility function for creating item from DB record"""
        item = create_item_from_db_record(self.db_record)
        
        self.assertEqual(item.item_id, 'util_001')
        self.assertEqual(item.category, 'top')
        self.assertEqual(item.color, 'green')
        self.assertEqual(item.metadata['material'], 'wool')
        
        print("✅ Create item from DB record: Utility function working correctly")
    
    def test_items_to_outfit(self):
        """Test utility function for converting items to outfit"""
        outfit = items_to_outfit(self.items[:2])
        
        self.assertEqual(len(outfit.items), 2)
        self.assertIn("outfit_", outfit.outfit_id)
        self.assertEqual(outfit.source, "generated")
        
        # Test with custom outfit_id
        custom_outfit = items_to_outfit(self.items, outfit_id="custom_001")
        self.assertEqual(custom_outfit.outfit_id, "custom_001")
        
        print("✅ Items to outfit: Utility function working correctly")
    
    def test_filter_items_by_context(self):
        """Test utility function for filtering items by context"""
        context = RecommendationContext(occasion="work", season="cold")
        
        filtered_items = filter_items_by_context(self.items, context)
        
        # Should only include items that match work occasion (formal/smart-casual) and cold season
        self.assertEqual(len(filtered_items), 1)  # Only the formal black bottom
        self.assertEqual(filtered_items[0].item_id, "item_002")
        
        print("✅ Filter items by context: Utility function working correctly")


if __name__ == '__main__':
    unittest.main()