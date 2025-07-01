"""
End-to-end integration tests for all recommendation engines
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import time
import pickle
import sqlite3
from collections import Counter, defaultdict


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "models"))
sys.path.append(str(Path(__file__).parent.parent / "src" / "data"))

from data_structures import Item, RecommendationContext, Recommendation, Outfit
from similarity_engine import SimilarityEngine
from compatibility_engine import CompatibilityEngine
from recommendation_engine import RecommendationEngine
from data_manager import DataManager


class TestEnginesIntegration(unittest.TestCase):
    """End-to-end integration tests for all engines working together"""
    
    def setUp(self):
        """Set up integration test environment"""
        # Create temporary database
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "integration_test.db"
        self.manager = DataManager(str(self.db_path))
        
        # Initialize all engines
        self.similarity_engine = SimilarityEngine()
        self.compatibility_engine = CompatibilityEngine()
        self.recommendation_engine = RecommendationEngine(data_manager=self.manager)
        
        # Create comprehensive test wardrobe
        self.test_wardrobe = self._create_comprehensive_wardrobe()
        self._populate_database()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_comprehensive_wardrobe(self):
        """Create a comprehensive test wardrobe with diverse items"""
        wardrobe = [
            # Casual tops
            Item("casual_white_tee", "top", "white", "tshirt", "casual", season="all", embedding=np.random.rand(512)),
            Item("casual_blue_shirt", "top", "blue", "button_down", "casual", season="all", embedding=np.random.rand(512)),
            Item("casual_gray_hoodie", "top", "gray", "hoodie", "casual", season="cold", embedding=np.random.rand(512)),
            
            # Smart-casual tops
            Item("sc_navy_polo", "top", "navy", "polo", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("sc_white_blouse", "top", "white", "blouse", "smart-casual", season="all", embedding=np.random.rand(512)),
            
            # Formal tops
            Item("formal_white_shirt", "top", "white", "dress_shirt", "formal", season="all", embedding=np.random.rand(512)),
            Item("formal_black_blazer", "outerwear", "black", "blazer", "formal", season="all", embedding=np.random.rand(512)),
            
            # Bottoms
            Item("casual_blue_jeans", "bottom", "blue", "jeans", "casual", season="all", embedding=np.random.rand(512)),
            Item("casual_black_jeans", "bottom", "black", "jeans", "casual", season="all", embedding=np.random.rand(512)),
            Item("sc_khaki_chinos", "bottom", "khaki", "chinos", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("formal_black_pants", "bottom", "black", "dress_pants", "formal", season="all", embedding=np.random.rand(512)),
            Item("summer_shorts", "bottom", "navy", "shorts", "casual", season="warm", embedding=np.random.rand(512)),
            
            # Shoes
            Item("casual_white_sneakers", "shoes", "white", "sneakers", "casual", season="all", embedding=np.random.rand(512)),
            Item("casual_black_sneakers", "shoes", "black", "sneakers", "casual", season="all", embedding=np.random.rand(512)),
            Item("sc_brown_loafers", "shoes", "brown", "loafers", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("formal_black_oxfords", "shoes", "black", "oxfords", "formal", season="all", embedding=np.random.rand(512)),
            
            # Outerwear
            Item("casual_denim_jacket", "outerwear", "blue", "denim_jacket", "casual", season="cold", embedding=np.random.rand(512)),
            Item("winter_puffer_coat", "outerwear", "black", "puffer_coat", "casual", season="cold", embedding=np.random.rand(512)),
            
            # Accessories
            Item("brown_leather_belt", "accessory", "brown", "belt", "smart-casual", season="all", embedding=np.random.rand(512)),
            Item("black_leather_belt", "accessory", "black", "belt", "formal", season="all", embedding=np.random.rand(512)),
            Item("casual_baseball_cap", "accessory", "navy", "hat", "casual", season="all", embedding=np.random.rand(512)),
        ]
        
        # Normalize all embeddings
        for item in wardrobe:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        return wardrobe
    
    def _populate_database(self):
        """Add all test items to database"""
        for item in self.test_wardrobe:
            # Add item
            success = self.manager.add_item(
                item.item_id, item.source, f"{item.item_id}.jpg", item.category,
                item.color, item.style, item.formality, item.season
            )
            self.assertTrue(success)
            
            # Add embedding
            success = self.manager.add_embedding(item.item_id, item.embedding)
            self.assertTrue(success)
    
    def test_complete_recommendation_workflow(self):
        """Test complete workflow from item selection to outfit recommendation"""
        print("\n🔄 Testing complete recommendation workflow...")
        
        # Step 1: User selects a base item
        base_item = next(item for item in self.test_wardrobe if item.item_id == "casual_blue_shirt")
        print(f"1. Selected base item: {base_item}")
        
        # Step 2: Find similar items (for style variations)
        similar_items = self.recommendation_engine.find_similar_items(base_item, top_k=3)
        print(f"2. Found {len(similar_items)} similar items for style variations")
        
        # Step 3: Find complementary items (for outfit building)
        complementary_items = self.recommendation_engine.find_complementary_items(base_item, top_k=5)
        print(f"3. Found {len(complementary_items)} complementary items")
        
        # Step 4: Generate complete outfit
        complete_outfit, outfit_score = self.recommendation_engine.generate_complete_outfit(base_item)
        print(f"4. Generated complete outfit with {len(complete_outfit)} items (score: {outfit_score:.3f})")
        
        # Step 5: Score the outfit using compatibility engine
        manual_score = self.compatibility_engine.score_outfit(complete_outfit)
        print(f"5. Manual compatibility score: {manual_score:.3f}")
        
        # Verify workflow results
        self.assertGreater(len(similar_items), 0)
        self.assertGreater(len(complementary_items), 0)
        self.assertGreaterEqual(len(complete_outfit), 2)  # At least 2 items
        self.assertAlmostEqual(outfit_score, manual_score, places=2)  # Scores should match
        
        print("✅ Complete workflow: All steps successful")
    
    def test_context_aware_recommendations(self):
        """Test context-aware recommendations across all engines"""
        print("\n🌡️ Testing context-aware recommendations...")
        
        base_item = next(item for item in self.test_wardrobe if item.item_id == "formal_white_shirt")
        
        # Test different contexts
        contexts = [
            ("work", RecommendationContext(occasion="work", weather="mild")),
            ("formal_event", RecommendationContext(occasion="formal", weather="cold")),
            ("casual_summer", RecommendationContext(occasion="casual", weather="hot", season="warm"))
        ]
        
        results = {}
        for context_name, context in contexts:
            # Get recommendations with context
            comp_recs = self.recommendation_engine.find_complementary_items(
                base_item, top_k=5, context=context
            )
            
            outfit, score = self.recommendation_engine.generate_complete_outfit(
                base_item, context=context
            )
            
            results[context_name] = {
                'complementary_count': len(comp_recs),
                'outfit_size': len(outfit),
                'outfit_score': score,
                'formality_levels': [item.formality for item in outfit],
                'seasons': [item.season for item in outfit if item.season != 'all']
            }
            
            print(f"Context '{context_name}': {len(comp_recs)} complements, {len(outfit)}-item outfit (score: {score:.3f})")
        
        # Verify context affects recommendations
        # Formal context should prefer formal items
        formal_formalities = results['formal_event']['formality_levels']
        self.assertIn('formal', formal_formalities)
        
        # Summer context should avoid cold-weather items
        summer_seasons = results['casual_summer']['seasons']
        self.assertNotIn('cold', summer_seasons)
        
        print("✅ Context-aware recommendations: Context properly influences recommendations")
    
    def test_similarity_vs_compatibility_distinction(self):
        """Test that similarity and compatibility engines produce different results"""
        print("\n🔍 Testing similarity vs compatibility distinction...")
        
        target_item = next(item for item in self.test_wardrobe if item.item_id == "casual_white_tee")
        candidates = [item for item in self.test_wardrobe if item.item_id != target_item.item_id]
        
        # Get similar items (should prefer same category)
        similar_items = self.similarity_engine.find_similar_items(
            target_item, candidates, top_k=5, same_category_only=True
        )
        
        # Get complementary items (should prefer different categories)
        complementary_items = self.compatibility_engine.find_complementary_items(
            target_item, candidates, top_k=5
        )
        
        # Analyze results
        similar_categories = [item.category for item, _ in similar_items]
        complementary_categories = [item.category for item, _ in complementary_items]
        
        # Similar items should mostly be same category
        same_category_similar = sum(1 for cat in similar_categories if cat == target_item.category)
        similarity_same_category_ratio = same_category_similar / len(similar_items) if similar_items else 0
        
        # Complementary items should mostly be different categories
        different_category_comp = sum(1 for cat in complementary_categories if cat != target_item.category)
        compatibility_diff_category_ratio = different_category_comp / len(complementary_items) if complementary_items else 0
        
        print(f"Similar items: {similarity_same_category_ratio:.1%} same category")
        print(f"Complementary items: {compatibility_diff_category_ratio:.1%} different category")
        
        # Verify distinction
        self.assertGreater(similarity_same_category_ratio, 0.5)  # Most similar items same category
        self.assertGreater(compatibility_diff_category_ratio, 0.5)  # Most complementary items different category
        
        print("✅ Similarity vs Compatibility: Engines produce appropriately different results")
    
    def test_outfit_building_progression(self):
        """Test progressive outfit building"""
        print("\n👔 Testing progressive outfit building...")
        
        # Start with base item
        base_item = next(item for item in self.test_wardrobe if item.item_id == "sc_navy_polo")
        current_outfit = [base_item]
        
        progression_scores = []
        
        # Progressively add items
        for step in range(3):  # Add up to 3 more items
            # Get completion suggestions
            suggestions = self.compatibility_engine.suggest_outfit_completion(
                current_outfit,
                [item for item in self.test_wardrobe if item not in current_outfit]
            )
            
            if suggestions:
                # Add best suggestion
                best_item, best_score = suggestions[0]
                current_outfit.append(best_item)
                
                # Score current outfit
                outfit_score = self.compatibility_engine.score_outfit(current_outfit)
                progression_scores.append(outfit_score)
                
                print(f"Step {step + 1}: Added {best_item.item_id} ({best_item.category}), outfit score: {outfit_score:.3f}")
            else:
                break
        
        # Verify progression
        self.assertGreaterEqual(len(current_outfit), 2)  # Should have added at least 1 item
        self.assertGreater(len(progression_scores), 0)  # Should have some scores
        
        # Final outfit should be reasonably good
        final_score = progression_scores[-1] if progression_scores else 0
        self.assertGreater(final_score, 0.4)  # Should be decent compatibility
        
        print(f"Final outfit: {len(current_outfit)} items with score {final_score:.3f}")
        print("✅ Progressive outfit building: Successfully built outfit step by step")
    
    def test_performance_with_full_wardrobe(self):
        """Test performance with full wardrobe"""
        print("\n⚡ Testing performance with full wardrobe...")
        
        target_item = self.test_wardrobe[0]
        candidates = self.test_wardrobe[1:]
        
        # Test similarity engine performance
        start_time = time.time()
        similar_items = self.similarity_engine.find_similar_items(target_item, candidates, top_k=5)
        similarity_time = time.time() - start_time
        
        # Test compatibility engine performance
        start_time = time.time()
        complementary_items = self.compatibility_engine.find_complementary_items(target_item, candidates, top_k=5)
        compatibility_time = time.time() - start_time
        
        # Test unified engine performance
        start_time = time.time()
        unified_similar = self.recommendation_engine.find_similar_items(target_item, top_k=5)
        unified_complementary = self.recommendation_engine.find_complementary_items(target_item, top_k=5)
        outfit, score = self.recommendation_engine.generate_complete_outfit(target_item)
        unified_time = time.time() - start_time
        
        print(f"Similarity engine: {similarity_time:.3f}s")
        print(f"Compatibility engine: {compatibility_time:.3f}s")
        print(f"Unified engine (all operations): {unified_time:.3f}s")
        
        # Performance should be reasonable
        self.assertLess(similarity_time, 1.0)
        self.assertLess(compatibility_time, 1.0)
        self.assertLess(unified_time, 2.0)
        
        print("✅ Performance: All engines perform within acceptable time limits")
    
    def test_recommendation_consistency(self):
        """Test consistency of recommendations across multiple calls"""
        print("\n🔄 Testing recommendation consistency...")
        
        target_item = next(item for item in self.test_wardrobe if item.item_id == "casual_blue_jeans")
        
        # Run recommendations multiple times
        runs = 3
        similar_results = []
        complementary_results = []
        outfit_results = []
        
        for run in range(runs):
            similar_recs = self.recommendation_engine.find_similar_items(target_item, top_k=3)
            comp_recs = self.recommendation_engine.find_complementary_items(target_item, top_k=3)
            outfit, score = self.recommendation_engine.generate_complete_outfit(target_item)
            
            similar_results.append([rec.item.item_id for rec in similar_recs])
            complementary_results.append([rec.item.item_id for rec in comp_recs])
            outfit_results.append(([item.item_id for item in outfit], score))
        
        # Check consistency (results should be identical since we're using deterministic algorithms)
        # Similar items should be consistent
        if similar_results[0]:  # If we have results
            for i in range(1, runs):
                self.assertEqual(similar_results[0], similar_results[i], "Similar item recommendations should be consistent")
        
        # Complementary items should be consistent
        if complementary_results[0]:  # If we have results
            for i in range(1, runs):
                self.assertEqual(complementary_results[0], complementary_results[i], "Complementary item recommendations should be consistent")
        
        print("✅ Consistency: Recommendations are consistent across multiple calls")
    
    def test_edge_cases_integration(self):
        """Test edge cases in integrated system"""
        print("\n⚠️ Testing edge cases integration...")
        
        # Test with item that has no good matches
        isolated_item = Item("isolated_formal_vest", "outerwear", "burgundy", "vest", "formal", embedding=np.random.rand(512))
        isolated_item.embedding = isolated_item.embedding / np.linalg.norm(isolated_item.embedding)
        
        # Add to database
        self.manager.add_item(isolated_item.item_id, isolated_item.source, f"{isolated_item.item_id}.jpg", 
                             isolated_item.category, isolated_item.color, isolated_item.style, isolated_item.formality)
        self.manager.add_embedding(isolated_item.item_id, isolated_item.embedding)
        
        # Test recommendations for isolated item
        similar_recs = self.recommendation_engine.find_similar_items(isolated_item, top_k=5)
        comp_recs = self.recommendation_engine.find_complementary_items(isolated_item, top_k=5)
        outfit, score = self.recommendation_engine.generate_complete_outfit(isolated_item)
        
        # Should handle gracefully
        self.assertIsInstance(similar_recs, list)
        self.assertIsInstance(comp_recs, list)
        self.assertGreaterEqual(len(outfit), 1)  # At least the base item
        self.assertGreaterEqual(score, 0.0)
        
        # Test with conflicting context
        conflicting_context = RecommendationContext(
            weather="hot",  # Hot weather
            season="cold",  # But cold season
            occasion="formal"  # And formal occasion
        )
        
        base_item = self.test_wardrobe[0]
        context_recs = self.recommendation_engine.find_complementary_items(
            base_item, top_k=5, context=conflicting_context
        )
        
        # Should still return some results
        self.assertIsInstance(context_recs, list)
        
        print("✅ Edge cases: System handles edge cases gracefully")


class TestEnginesWithPolyvoreIntegration(unittest.TestCase):
    """Integration tests with real Polyvore data"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class with production database"""
        cls.production_db_path = Path("data/closetgpt.db")
        if not cls.production_db_path.exists():
            raise unittest.SkipTest("Production database not found - skipping Polyvore integration tests")
        
        cls.manager = DataManager(str(cls.production_db_path))
        cls.recommendation_engine = RecommendationEngine(data_manager=cls.manager)
        
        print("🗃️ Testing engines integration with Polyvore database")
    
    def test_polyvore_end_to_end_workflow(self):
        """Test end-to-end workflow with Polyvore data"""
        # Load a sample of Polyvore items
        sample_items = self.recommendation_engine.load_items_from_database(limit=50)
        
        if len(sample_items) < 10:
            self.skipTest("Need at least 10 Polyvore items")
        
        base_item = sample_items[0]
        
        # Run complete workflow
        similar_recs = self.recommendation_engine.find_similar_items(base_item, top_k=3)
        comp_recs = self.recommendation_engine.find_complementary_items(base_item, top_k=5)
        outfit, score = self.recommendation_engine.generate_complete_outfit(base_item)
        
        # Verify results
        self.assertIsInstance(similar_recs, list)
        self.assertIsInstance(comp_recs, list)
        self.assertGreaterEqual(len(outfit), 1)
        self.assertGreaterEqual(score, 0.0)
        
        print(f"✅ Polyvore end-to-end: Similar: {len(similar_recs)}, Complementary: {len(comp_recs)}, Outfit: {len(outfit)} items")
    
    def test_polyvore_large_scale_performance(self):
        """Test performance with larger Polyvore dataset"""
        import time
        
        # Load larger sample
        sample_items = self.recommendation_engine.load_items_from_database(limit=200)
        
        if len(sample_items) < 100:
            self.skipTest("Need at least 100 Polyvore items for large-scale testing")
        
        base_item = sample_items[0]
        
        # Test performance with large candidate pool
        start_time = time.time()
        
        similar_recs = self.recommendation_engine.find_similar_items(
            base_item, candidate_items=sample_items[1:100], top_k=10
        )
        comp_recs = self.recommendation_engine.find_complementary_items(
            base_item, candidate_items=sample_items[1:100], top_k=10
        )
        outfit, score = self.recommendation_engine.generate_complete_outfit(
            base_item, candidate_items=sample_items[1:100]
        )
        
        total_time = time.time() - start_time
        
        # Should complete in reasonable time
        print(f"✅ Polyvore large scale performance test time: {total_time:.4f}s")
        #
        # TODO: edit the threshold based on actual performance
        #
        self.assertLess(total_time, 5.0)

    def test_polyvore_embedding_quality_validation(self):
        """Test quality and consistency of CLIP embeddings"""
        print("\n🧠 Testing Polyvore embedding quality...")
        
        conn = sqlite3.connect(str(self.production_db_path))
        cursor = conn.cursor()
        
        # Sample embeddings for analysis
        cursor.execute("SELECT item_id, embedding_vector FROM embeddings LIMIT 100")
        sample_embeddings = cursor.fetchall()
        conn.close()
        
        if len(sample_embeddings) < 10:
            self.skipTest("Need at least 10 embeddings")
        
        embeddings_data = []
        embedding_dims = []
        
        for item_id, blob_data in sample_embeddings:
            try:
                # Unpickle the numpy array
                embedding = pickle.loads(blob_data)
                self.assertIsInstance(embedding, np.ndarray)
                
                embeddings_data.append(embedding)
                embedding_dims.append(len(embedding))
                
                # Check embedding is normalized (L2 norm should be ~1.0)
                norm = np.linalg.norm(embedding)
                self.assertAlmostEqual(norm, 1.0, places=1, 
                                    msg=f"Embedding {item_id} not normalized: {norm}")
                
            except Exception as e:
                self.fail(f"Failed to load embedding for {item_id}: {e}")
        
        # Check embedding dimensions consistency
        unique_dims = set(embedding_dims)
        self.assertEqual(len(unique_dims), 1, 
                        f"Inconsistent embedding dimensions: {unique_dims}")
        
        expected_dim = 512  # CLIP ViT-B/32
        actual_dim = embedding_dims[0]
        self.assertEqual(actual_dim, expected_dim,
                        f"Expected {expected_dim}D embeddings, got {actual_dim}D")
        
        # Check embedding distribution (should not be all zeros or identical)
        if len(embeddings_data) >= 2:
            embedding_matrix = np.stack(embeddings_data)
            
            # Check variance (embeddings should not be identical)
            variances = np.var(embedding_matrix, axis=0)
            min_variance = np.min(variances)
            self.assertGreater(min_variance, 1e-6, "Embeddings appear to be identical")
            
            # Check mean is reasonable (should be roughly centered)
            mean_embedding = np.mean(embedding_matrix, axis=0)
            mean_magnitude = np.linalg.norm(mean_embedding)
            self.assertLess(mean_magnitude, 0.5, "Embeddings not well distributed")
        
        print(f"✅ Embedding quality: {len(embeddings_data)} embeddings, {actual_dim}D, properly normalized")

    def test_polyvore_missing_metadata_handling(self):
        """Test handling of incomplete Polyvore metadata"""
        print("\n⚠️ Testing missing metadata handling...")
        
        # Load items with missing/unknown data
        sample_items = self.recommendation_engine.load_items_from_database(limit=100)
        
        # Find items with missing metadata
        unknown_category_items = [item for item in sample_items if item.category == "unknown"]
        missing_color_items = [item for item in sample_items if not item.color]
        missing_style_items = [item for item in sample_items if not item.style]
        
        print(f"Items with unknown category: {len(unknown_category_items)}")
        print(f"Items with missing color: {len(missing_color_items)}")
        print(f"Items with missing style: {len(missing_style_items)}")
        
        if unknown_category_items:
            # Test recommendations with unknown category items
            unknown_item = unknown_category_items[0]
            
            # Should not crash with unknown category
            similar_recs = self.recommendation_engine.find_similar_items(unknown_item, top_k=5)
            comp_recs = self.recommendation_engine.find_complementary_items(unknown_item, top_k=5)
            outfit, score = self.recommendation_engine.generate_complete_outfit(unknown_item)
            
            # Should handle gracefully
            self.assertIsInstance(similar_recs, list)
            self.assertIsInstance(comp_recs, list)
            self.assertGreaterEqual(len(outfit), 1)
            self.assertGreaterEqual(score, 0.0)
            
            print(f"✅ Unknown category handling: Generated {len(similar_recs)} similar, {len(comp_recs)} complementary")
        
        # Test with completely minimal metadata item
        if sample_items:
            minimal_item = sample_items[0]
            # Temporarily clear metadata
            original_color = minimal_item.color
            original_style = minimal_item.style
            
            minimal_item.color = None
            minimal_item.style = None
            
            try:
                outfit, score = self.recommendation_engine.generate_complete_outfit(minimal_item)
                self.assertGreaterEqual(len(outfit), 1)
                print("✅ Minimal metadata: System handles items with missing fields")
            finally:
                # Restore original data
                minimal_item.color = original_color
                minimal_item.style = original_style

    def test_polyvore_recommendation_diversity(self):
        """Test diversity of recommendations to avoid echo chambers"""
        print("\n🌈 Testing recommendation diversity...")
        
        # Load diverse sample
        sample_items = self.recommendation_engine.load_items_from_database(limit=200)
        if len(sample_items) < 50:
            self.skipTest("Need at least 50 items for diversity testing")
        
        base_item = sample_items[0]
        
        # Get larger recommendation sets
        similar_recs = self.recommendation_engine.find_similar_items(base_item, top_k=20)
        comp_recs = self.recommendation_engine.find_complementary_items(base_item, top_k=20)
        
        # Test diversity metrics
        def calculate_diversity_score(recommendations):
            if len(recommendations) < 2:
                return 1.0
            
            embeddings = [rec.item.embedding for rec in recommendations]
            similarities = []
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Diversity = 1 - average similarity
            return 1.0 - np.mean(similarities)
        
        similar_diversity = calculate_diversity_score(similar_recs)
        comp_diversity = calculate_diversity_score(comp_recs)
        
        print(f"Similar items diversity: {similar_diversity:.3f}")
        print(f"Complementary items diversity: {comp_diversity:.3f}")
        
        # Similar items can be less diverse, but complementary should be diverse
        self.assertGreater(comp_diversity, 0.1, "Complementary recommendations too similar")
        
        # Test source diversity (should not all come from same small subset)
        if len(similar_recs) >= 10:
            item_ids = [rec.item.item_id for rec in similar_recs[:10]]
            unique_prefixes = set(item_id.split('_')[1][:3] for item_id in item_ids)  # First 3 chars of ID
            diversity_ratio = len(unique_prefixes) / len(item_ids)
            self.assertGreater(diversity_ratio, 0.3, "Recommendations too concentrated")
        
        print("✅ Diversity: Recommendations show appropriate diversity")

    def test_polyvore_category_distribution(self):
        """Validate category distribution in recommendations"""
        print("\n📊 Testing category distribution...")
        
        # Test with items from different categories
        sample_items = self.recommendation_engine.load_items_from_database(limit=100)
        
        # Group by category
        category_items = defaultdict(list)
        for item in sample_items:
            category_items[item.category].append(item)
        
        print(f"Categories found: {list(category_items.keys())}")
        
        # Test recommendations for each category
        for category, items in category_items.items():
            if len(items) < 5:  # Skip categories with too few items
                continue
                
            base_item = items[0]
            comp_recs = self.recommendation_engine.find_complementary_items(base_item, top_k=10)
            
            if comp_recs:
                rec_categories = [rec.item.category for rec in comp_recs]
                rec_category_counts = Counter(rec_categories)
                
                print(f"Category '{category}' -> recommendations: {dict(rec_category_counts)}")
                
                # For known categories, should recommend different categories
                if category != "unknown":
                    different_category_count = sum(1 for cat in rec_categories if cat != category)
                    total_recs = len(rec_categories)
                    different_ratio = different_category_count / total_recs if total_recs > 0 else 0
                    
                    # At least 30% should be different categories for outfit completion
                    self.assertGreater(different_ratio, 0.3, 
                                    f"Category '{category}' recommendations too homogeneous")
        
        print("✅ Category distribution: Appropriate cross-category recommendations")

    def test_polyvore_scaling_performance(self):
        """Test performance scaling with increasing dataset sizes"""
        print("\n⚡ Testing scaling performance...")
        
        # Test with different dataset sizes
        test_sizes = [50, 100, 200, 500]
        performance_results = {}
        
        for size in test_sizes:
            sample_items = self.recommendation_engine.load_items_from_database(limit=size)
            if len(sample_items) < size:
                continue
                
            base_item = sample_items[0]
            candidates = sample_items[1:]
            
            # Time the operations
            start_time = time.time()
            
            similar_recs = self.recommendation_engine.find_similar_items(
                base_item, candidate_items=candidates, top_k=10
            )
            comp_recs = self.recommendation_engine.find_complementary_items(
                base_item, candidate_items=candidates, top_k=10
            )
            outfit, score = self.recommendation_engine.generate_complete_outfit(
                base_item, candidate_items=candidates
            )
            
            total_time = time.time() - start_time
            performance_results[size] = total_time
            
            print(f"Size {size:3d}: {total_time:.3f}s ({len(similar_recs)} sim, {len(comp_recs)} comp)")
            
            # Performance should be reasonable
            if size <= 200:
                self.assertLess(total_time, 2.0, f"Performance too slow for {size} items")
            elif size <= 500:
                self.assertLess(total_time, 5.0, f"Performance too slow for {size} items")
        
        # Check scaling is reasonable (not exponential)
        if len(performance_results) >= 2:
            sizes = sorted(performance_results.keys())
            times = [performance_results[size] for size in sizes]
            
            # Time should not increase faster than O(n^1.5)
            if len(sizes) >= 3:
                ratio_1 = times[1] / times[0] if times[0] > 0 else 1
                ratio_2 = times[2] / times[1] if times[1] > 0 else 1
                
                # Second ratio should not be much larger than first
                scaling_factor = ratio_2 / ratio_1 if ratio_1 > 0 else 1
                self.assertLess(scaling_factor, 3.0, "Performance scaling too poor")
        
        print("✅ Scaling performance: Performance scales reasonably with dataset size")

    def test_polyvore_memory_pressure(self):
        """Test system behavior under memory constraints"""
        print("\n🧠 Testing memory pressure handling...")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load progressively larger datasets
        max_items = min(1000, 261196)  # Don't exceed available items
        batch_size = 200
        
        memory_usage = []
        
        for size in range(batch_size, max_items, batch_size):
            # Load items
            sample_items = self.recommendation_engine.load_items_from_database(limit=size)
            
            # Perform operations
            if sample_items:
                base_item = sample_items[0]
                similar_recs = self.recommendation_engine.find_similar_items(base_item, top_k=5)
                comp_recs = self.recommendation_engine.find_complementary_items(base_item, top_k=5)
            
            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            memory_usage.append((size, memory_increase))
            
            print(f"Size {size:4d}: Memory +{memory_increase:.1f}MB")
            
            # Memory should not grow excessively
            if memory_increase > 1000:  # More than 1GB increase
                print(f"⚠️ High memory usage detected: {memory_increase:.1f}MB")
                break
            
            # Force garbage collection
            gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase: {total_increase:.1f}MB")
        
        # Should not use excessive memory
        self.assertLess(total_increase, 2000, "Excessive memory usage detected")
        
        print("✅ Memory pressure: System handles large datasets without excessive memory usage")

    def test_polyvore_cache_efficiency(self):
        """Test embedding cache efficiency"""
        print("\n💾 Testing cache efficiency...")
        
        # Load sample items
        sample_items = self.recommendation_engine.load_items_from_database(limit=100)
        if len(sample_items) < 20:
            self.skipTest("Need at least 20 items for cache testing")
        
        base_item = sample_items[0]
        
        # First run - cache miss
        start_time = time.time()
        similar_recs_1 = self.recommendation_engine.find_similar_items(base_item, top_k=10)
        first_run_time = time.time() - start_time
        
        # Second run - should use cache
        start_time = time.time()
        similar_recs_2 = self.recommendation_engine.find_similar_items(base_item, top_k=10)
        second_run_time = time.time() - start_time
        
        # Results should be identical
        if similar_recs_1 and similar_recs_2:
            first_ids = [rec.item.item_id for rec in similar_recs_1]
            second_ids = [rec.item.item_id for rec in similar_recs_2]
            self.assertEqual(first_ids, second_ids, "Cache should return identical results")
        
        # Second run should be faster (if caching is implemented)
        print(f"First run: {first_run_time:.3f}s, Second run: {second_run_time:.3f}s")
        
        # This test will help identify if caching is working
        if second_run_time < first_run_time * 0.8:
            print("✅ Cache efficiency: Caching appears to be working")
        else:
            print("ℹ️ Cache efficiency: No significant speedup detected (caching may not be implemented)")

    def test_polyvore_concurrent_recommendations(self):
        """Test handling of concurrent recommendation requests"""
        print("\n🔄 Testing concurrent recommendations...")
        
        import threading
        import queue
        
        sample_items = self.recommendation_engine.load_items_from_database(limit=50)
        if len(sample_items) < 10:
            self.skipTest("Need at least 10 items for concurrency testing")
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker(item, worker_id):
            try:
                # Perform recommendations
                similar_recs = self.recommendation_engine.find_similar_items(item, top_k=5)
                comp_recs = self.recommendation_engine.find_complementary_items(item, top_k=5)
                outfit, score = self.recommendation_engine.generate_complete_outfit(item)
                
                results_queue.put({
                    'worker_id': worker_id,
                    'similar_count': len(similar_recs),
                    'comp_count': len(comp_recs),
                    'outfit_size': len(outfit),
                    'score': score
                })
            except Exception as e:
                errors_queue.put(f"Worker {worker_id}: {e}")
        
        # Launch concurrent workers
        threads = []
        num_workers = min(5, len(sample_items))
        
        start_time = time.time()
        
        for i in range(num_workers):
            thread = threading.Thread(target=worker, args=(sample_items[i], i))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        print(f"Concurrent execution: {len(results)} successful, {len(errors)} errors in {total_time:.3f}s")
        
        # Should have no errors
        for error in errors:
            print(f"Error: {error}")
        self.assertEqual(len(errors), 0, "Concurrent access should not cause errors")
        
        # Should get results from all workers
        self.assertEqual(len(results), num_workers, "All workers should complete successfully")
        
        print("✅ Concurrent recommendations: System handles concurrent requests correctly")

    def test_polyvore_embedding_distribution(self):
        """Validate embedding space coverage and clustering"""
        print("\n📈 Testing embedding distribution...")
        
        # Load embeddings directly from database
        conn = sqlite3.connect(str(self.production_db_path))
        cursor = conn.cursor()
        
        # Sample more embeddings for distribution analysis
        cursor.execute("SELECT item_id, embedding_vector FROM embeddings LIMIT 500")
        embedding_data = cursor.fetchall()
        conn.close()
        
        if len(embedding_data) < 100:
            self.skipTest("Need at least 100 embeddings for distribution testing")
        
        embeddings = []
        for item_id, blob_data in embedding_data:
            try:
                embedding = pickle.loads(blob_data)
                embeddings.append(embedding)
            except:
                continue
        
        if len(embeddings) < 50:
            self.skipTest("Could not load enough valid embeddings")
        
        embeddings_matrix = np.stack(embeddings)
        
        # Test 1: Distribution statistics
        mean_embedding = np.mean(embeddings_matrix, axis=0)
        std_embedding = np.std(embeddings_matrix, axis=0)
        
        # Mean should be roughly centered (not all positive or negative)
        mean_magnitude = np.linalg.norm(mean_embedding)
        self.assertLess(mean_magnitude, 0.3, "Embeddings not well centered")
        
        # Standard deviation should be reasonable (not too small or large)
        mean_std = np.mean(std_embedding)
        self.assertGreater(mean_std, 0.05, "Embeddings lack diversity")
        self.assertLess(mean_std, 1.0, "Embeddings too scattered")
        
        # Test 2: Clustering analysis
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Try different numbers of clusters
        cluster_scores = []
        for n_clusters in [5, 10, 20]:
            if len(embeddings) > n_clusters * 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_matrix)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(embeddings_matrix, cluster_labels)
                cluster_scores.append((n_clusters, silhouette_avg))
        
        print(f"Embedding distribution stats:")
        print(f"  Mean magnitude: {mean_magnitude:.3f}")
        print(f"  Average std: {mean_std:.3f}")
        if cluster_scores:
            print(f"  Clustering scores: {cluster_scores}")
            
            # Should have reasonable clustering structure
            best_score = max(score for _, score in cluster_scores)
            self.assertGreater(best_score, 0.1, "Embeddings show poor clustering structure")
        
        print("✅ Embedding distribution: Embeddings show good distribution and clustering properties")

    def test_polyvore_metadata_consistency(self):
        """Check metadata consistency across the dataset"""
        print("\n🔍 Testing metadata consistency...")
        
        # Load larger sample for consistency checking
        sample_items = self.recommendation_engine.load_items_from_database(limit=1000)
        
        # Analyze metadata patterns
        categories = Counter(item.category for item in sample_items)
        formalities = Counter(item.formality for item in sample_items)
        seasons = Counter(item.season for item in sample_items)
        sources = Counter(item.source for item in sample_items)
        
        print(f"Categories: {dict(categories.most_common())}")
        print(f"Formalities: {dict(formalities.most_common())}")
        print(f"Seasons: {dict(seasons.most_common())}")
        print(f"Sources: {dict(sources.most_common())}")
        
        # Check for data quality issues
        total_items = len(sample_items)
        
        # Too many unknown categories indicates data quality issue
        unknown_ratio = categories.get('unknown', 0) / total_items
        if unknown_ratio > 0.5:
            print(f"⚠️ High unknown category ratio: {unknown_ratio:.1%}")
        
        # Check for missing critical fields
        missing_color = sum(1 for item in sample_items if not item.color)
        missing_style = sum(1 for item in sample_items if not item.style)
        
        missing_color_ratio = missing_color / total_items
        missing_style_ratio = missing_style / total_items
        
        print(f"Missing color: {missing_color_ratio:.1%}")
        print(f"Missing style: {missing_style_ratio:.1%}")
        
        # Verify basic consistency
        self.assertGreater(len(categories), 0, "Should have at least some categories")
        self.assertGreater(len(formalities), 0, "Should have formality levels")
        self.assertIn('polyvore', sources, "Should have polyvore source items")
        
        # Data quality warnings (not failures)
        if unknown_ratio > 0.8:
            print("⚠️ Warning: Very high ratio of unknown categories")
        if missing_color_ratio > 0.9:
            print("⚠️ Warning: Most items missing color information")
        
        print("✅ Metadata consistency: Basic consistency checks passed")

if __name__ == '__main__':
    unittest.main()