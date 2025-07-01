"""
Unified Recommendation Engine for ClosetGPT v1
Combines similarity and compatibility engines for comprehensive recommendations
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import logging
import sys

from data_structures import Item, RecommendationContext, Recommendation, create_item_from_db_record
from similarity_engine import SimilarityEngine

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from similarity_engine import SimilarityEngine, Item, create_item_from_db_record
from compatibility_engine import CompatibilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Unified engine combining similarity and compatibility for comprehensive recommendations"""
    
    def __init__(self, data_manager=None):
        """
        Initialize recommendation engine
        
        Args:
            data_manager: DataManager instance for database access
        """
        self.similarity_engine = SimilarityEngine()
        self.compatibility_engine = CompatibilityEngine()
        self.data_manager = data_manager
        self.items_cache = {}
        logger.info("RecommendationEngine initialized")
    
    def load_items_from_database(self, 
                                source: Optional[str] = None,
                                category: Optional[str] = None,
                                limit: Optional[int] = None) -> List[Item]:
        """
        Load items from database and convert to Item objects
        
        Args:
            source: Filter by source (e.g., 'polyvore', 'personal')
            category: Filter by category
            limit: Maximum number of items to load
            
        Returns:
            List of Item objects with embeddings
        """
        if not self.data_manager:
            logger.error("No data_manager provided to load items from database")
            return []
        
        # Get items from database
        db_items = self.data_manager.get_items(source=source, category=category)
        
        if limit:
            db_items = db_items[:limit]
        
        items = []
        item_ids = [item['item_id'] for item in db_items]
        
        # Get embeddings in batch
        embeddings = self.data_manager.get_embeddings(item_ids)
        
        # Convert to Item objects
        for db_item in db_items:
            item = create_item_from_db_record(db_item)
            item.embedding = embeddings.get(db_item['item_id'])
            
            if item.embedding is not None:
                items.append(item)
            else:
                logger.warning(f"No embedding found for item {db_item['item_id']}")
        
        logger.info(f"Loaded {len(items)} items from database")
        return items
    
    def find_similar_items(self,
                          target_item: Item,
                          candidate_items: Optional[List[Item]] = None,
                          top_k: int = 10,
                          context: Optional[RecommendationContext] = None) -> List[Recommendation]:
        """
        Find items similar to the target item (for style variations)
        
        Args:
            target_item: Item to find similarities for
            candidate_items: Pool of candidates (loads from DB if None)
            top_k: Number of recommendations
            context: Recommendation context for filtering
            
        Returns:
            List of similarity-based recommendations
        """
        if candidate_items is None:
            candidate_items = self.load_items_from_database(category=target_item.category)
        
        # Apply context filtering
        if context:
            candidate_items = self._apply_context_filter(candidate_items, context)
        
        # Get similar items
        similar_items = self.similarity_engine.find_similar_items(
            target_item, candidate_items, top_k=top_k, same_category_only=True
        )
        
        # Convert to recommendations
        recommendations = []
        for item, score in similar_items:
            reason = f"Similar {item.category} - {score:.0%} style match"
            recommendations.append(Recommendation(item, score, reason, "similar"))
        
        return recommendations
    
    def find_complementary_items(self,
                                target_item: Item,
                                candidate_items: Optional[List[Item]] = None,
                                top_k: int = 10,
                                context: Optional[RecommendationContext] = None) -> List[Recommendation]:
        """
        Find items that complement the target item (for outfit building)
        
        Args:
            target_item: Item to find complements for
            candidate_items: Pool of candidates (loads from DB if None)
            top_k: Number of recommendations
            context: Recommendation context for filtering
            
        Returns:
            List of compatibility-based recommendations
        """
        if candidate_items is None:
            candidate_items = self.load_items_from_database()
        
        # Apply context filtering
        if context:
            candidate_items = self._apply_context_filter(candidate_items, context)
        
        # Get complementary items
        complementary_items = self.compatibility_engine.find_complementary_items(
            target_item, candidate_items, top_k=top_k
        )
        
        # Convert to recommendations
        recommendations = []
        for item, score in complementary_items:
            reason = f"Complements {target_item.category} - {score:.0%} compatibility"
            recommendations.append(Recommendation(item, score, reason, "complementary"))
        
        return recommendations
    
    def complete_outfit(self,
                       partial_outfit: List[Item],
                       candidate_items: Optional[List[Item]] = None,
                       target_categories: Optional[List[str]] = None,
                       context: Optional[RecommendationContext] = None) -> List[Recommendation]:
        """
        Suggest items to complete a partial outfit
        
        Args:
            partial_outfit: Items already selected
            candidate_items: Pool of candidates (loads from DB if None)
            target_categories: Specific categories to suggest
            context: Recommendation context
            
        Returns:
            List of outfit completion recommendations
        """
        if candidate_items is None:
            candidate_items = self.load_items_from_database()
        
        # Apply context filtering
        if context:
            candidate_items = self._apply_context_filter(candidate_items, context)
        
        # Get outfit completion suggestions
        suggestions = self.compatibility_engine.suggest_outfit_completion(
            partial_outfit, candidate_items, target_categories
        )
        
        # Convert to recommendations
        recommendations = []
        for item, score in suggestions:
            reason = f"Completes outfit - {score:.0%} compatibility"
            recommendations.append(Recommendation(item, score, reason, "outfit_completion"))
        
        return recommendations
    
    def score_outfit(self, items: List[Item]) -> float:
        """Score an outfit's overall compatibility"""
        return self.compatibility_engine.score_outfit(items)
    
    def generate_complete_outfit(self,
                                base_item: Item,
                                candidate_items: Optional[List[Item]] = None,
                                context: Optional[RecommendationContext] = None) -> Tuple[List[Item], float]:
        """
        Generate a complete outfit starting from a base item
        
        Args:
            base_item: Starting item for the outfit
            candidate_items: Pool of candidates
            context: Recommendation context
            
        Returns:
            Tuple of (outfit_items, overall_score)
        """
        if candidate_items is None:
            candidate_items = self.load_items_from_database()
        
        # Apply context filtering
        if context:
            candidate_items = self._apply_context_filter(candidate_items, context)
        
        outfit = [base_item]
        used_categories = {base_item.category}
        
        # Define outfit completion priority
        if base_item.category == 'top':
            target_sequence = ['bottom', 'shoes', 'outerwear']
        elif base_item.category == 'bottom':
            target_sequence = ['top', 'shoes', 'outerwear']
        elif base_item.category == 'shoes':
            target_sequence = ['top', 'bottom', 'outerwear']
        else:
            target_sequence = ['top', 'bottom', 'shoes']
        
        # Add items in priority order
        for category in target_sequence:
            if category in used_categories:
                continue
            
            # Find best item for this category
            category_candidates = [item for item in candidate_items 
                                 if item.category == category and item.item_id != base_item.item_id]
            
            if not category_candidates:
                continue
            
            # Get compatibility scores with current outfit
            best_item = None
            best_score = 0.0
            
            for candidate in category_candidates:
                # Calculate average compatibility with all current outfit items
                compatibilities = []
                for outfit_item in outfit:
                    comp = self.compatibility_engine.calculate_compatibility(candidate, outfit_item)
                    compatibilities.append(comp)
                
                avg_compatibility = np.mean(compatibilities) if compatibilities else 0.0
                
                if avg_compatibility > best_score:
                    best_score = avg_compatibility
                    best_item = candidate
            
            # Add item if it meets minimum threshold
            if best_item and best_score > 0.4:
                outfit.append(best_item)
                used_categories.add(category)
        
        # Score the complete outfit
        overall_score = self.score_outfit(outfit)
        
        logger.info(f"Generated outfit with {len(outfit)} items, score: {overall_score:.3f}")
        return outfit, overall_score
    
    def _apply_context_filter(self, items: List[Item], context: RecommendationContext) -> List[Item]:
        """Apply context-based filtering to items"""
        filtered_items = items.copy()
        
        # Season filtering
        if context.season and context.season != 'all':
            filtered_items = [item for item in filtered_items 
                            if item.season == 'all' or item.season == context.season]
        
        # Weather-based filtering (simple rules)
        if context.weather:
            if context.weather == 'hot':
                # Filter out heavy items
                filtered_items = [item for item in filtered_items 
                                if not any(heavy in (item.style or '').lower() 
                                         for heavy in ['coat', 'jacket', 'sweater', 'boot'])]
            elif context.weather == 'cold':
                # Prefer warm items for outerwear
                pass  # Could add logic to boost warm items
        
        # Occasion filtering
        if context.occasion:
            occasion_formality_map = {
                'casual': ['casual'],
                'work': ['smart-casual', 'formal'],
                'formal': ['formal'],
                'date': ['smart-casual', 'formal']
            }
            
            allowed_formalities = occasion_formality_map.get(context.occasion, ['casual', 'smart-casual', 'formal'])
            filtered_items = [item for item in filtered_items 
                            if item.formality in allowed_formalities]
        
        logger.info(f"Context filtering: {len(items)} → {len(filtered_items)} items")
        return filtered_items
    
    def get_recommendations_for_user(self,
                                   user_id: str,
                                   recommendation_type: str = "daily",
                                   context: Optional[RecommendationContext] = None,
                                   limit: int = 10) -> Dict[str, List[Recommendation]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User identifier
            recommendation_type: Type of recommendations ("daily", "similar", "outfits")
            context: Recommendation context
            limit: Number of recommendations per category
            
        Returns:
            Dictionary of recommendation categories
        """
        recommendations = {
            'similar_items': [],
            'complementary_items': [],
            'complete_outfits': []
        }
        
        # For demo, use a sample of items from user's wardrobe or popular items
        if recommendation_type == "daily":
            # Load some items to base recommendations on
            sample_items = self.load_items_from_database(limit=5)
            
            if sample_items:
                # Generate recommendations based on first item
                base_item = sample_items[0]
                
                recommendations['similar_items'] = self.find_similar_items(
                    base_item, context=context, top_k=limit//2
                )
                
                recommendations['complementary_items'] = self.find_complementary_items(
                    base_item, context=context, top_k=limit//2
                )
                
                # Generate a complete outfit
                outfit, score = self.generate_complete_outfit(base_item, context=context)
                if len(outfit) > 1:
                    outfit_rec = Recommendation(
                        item=outfit[0],  # Base item
                        score=score,
                        reason=f"Complete outfit with {len(outfit)} items",
                        recommendation_type="complete_outfit"
                    )
                    recommendations['complete_outfits'] = [outfit_rec]
        
        return recommendations


# Test and example usage
def test_recommendation_engine():
    """Test the unified recommendation engine"""
    # Create sample items
    white_shirt = Item("shirt_001", "top", "white", "shirt", "smart-casual", embedding=np.random.rand(512))
    blue_jeans = Item("jeans_001", "bottom", "blue", "jeans", "casual", embedding=np.random.rand(512))
    brown_shoes = Item("shoes_001", "shoes", "brown", "loafers", "smart-casual", embedding=np.random.rand(512))
    navy_shirt = Item("shirt_002", "top", "navy", "shirt", "smart-casual", embedding=np.random.rand(512))
    black_pants = Item("pants_001", "bottom", "black", "dress_pants", "formal", embedding=np.random.rand(512))
    
    # Normalize embeddings
    for item in [white_shirt, blue_jeans, brown_shoes, navy_shirt, black_pants]:
        item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    candidates = [blue_jeans, brown_shoes, navy_shirt, black_pants]
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Test similar items
    print("=== SIMILAR ITEMS ===")
    similar_recs = engine.find_similar_items(white_shirt, candidates, top_k=3)
    for rec in similar_recs:
        print(f"{rec.item.item_id}: {rec.score:.3f} - {rec.reason}")
    
    # Test complementary items
    print("\n=== COMPLEMENTARY ITEMS ===")
    comp_recs = engine.find_complementary_items(white_shirt, candidates, top_k=3)
    for rec in comp_recs:
        print(f"{rec.item.item_id}: {rec.score:.3f} - {rec.reason}")
    
    # Test outfit generation
    print("\n=== COMPLETE OUTFIT ===")
    outfit, score = engine.generate_complete_outfit(white_shirt, candidates)
    print(f"Generated outfit (score: {score:.3f}):")
    for item in outfit:
        print(f"  - {item.item_id} ({item.category}): {item.color} {item.style}")
    
    # Test with context
    print("\n=== WITH CONTEXT ===")
    context = RecommendationContext(weather="cold", occasion="work")
    context_recs = engine.find_complementary_items(white_shirt, candidates, top_k=3, context=context)
    for rec in context_recs:
        print(f"{rec.item.item_id}: {rec.score:.3f} - {rec.reason}")


if __name__ == "__main__":
    test_recommendation_engine()