"""
Compatibility Engine for ClosetGPT v1
Determines how well items work together in outfits
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import logging
from data_structures import Item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompatibilityEngine:
    """Engine for determining outfit compatibility and finding complementary items"""
    
    def __init__(self):
        """Initialize compatibility engine"""
        self.color_rules = self._load_color_compatibility_rules()
        self.style_rules = self._load_style_compatibility_rules()
        self.category_rules = self._load_category_compatibility_rules()
        logger.info("CompatibilityEngine initialized")
    
    def _load_color_compatibility_rules(self) -> Dict[str, Dict[str, float]]:
        """Load color compatibility rules"""
        # Based on fashion color theory
        return {
            'white': {'black': 0.95, 'blue': 0.9, 'red': 0.85, 'green': 0.8, 'brown': 0.85, 'gray': 0.9},
            'black': {'white': 0.95, 'gray': 0.9, 'red': 0.8, 'blue': 0.85, 'green': 0.75},
            'blue': {'white': 0.9, 'gray': 0.85, 'khaki': 0.9, 'brown': 0.8, 'black': 0.85},
            'navy': {'white': 0.95, 'khaki': 0.9, 'gray': 0.85, 'brown': 0.8, 'red': 0.75},
            'gray': {'white': 0.9, 'black': 0.9, 'blue': 0.85, 'red': 0.8, 'yellow': 0.7},
            'red': {'white': 0.85, 'black': 0.8, 'gray': 0.8, 'blue': 0.7, 'khaki': 0.75},
            'green': {'white': 0.8, 'khaki': 0.85, 'brown': 0.8, 'black': 0.75},
            'brown': {'white': 0.85, 'khaki': 0.9, 'blue': 0.8, 'green': 0.8, 'cream': 0.85},
            'khaki': {'white': 0.9, 'blue': 0.9, 'brown': 0.9, 'navy': 0.9, 'green': 0.85},
            'beige': {'white': 0.85, 'brown': 0.9, 'blue': 0.8, 'black': 0.75},
            'cream': {'brown': 0.85, 'blue': 0.8, 'black': 0.8, 'green': 0.75}
        }
    
    def _load_style_compatibility_rules(self) -> Dict[str, Dict[str, float]]:
        """Load style compatibility rules"""
        return {
            'casual': {'casual': 0.9, 'smart-casual': 0.7, 'formal': 0.3},
            'smart-casual': {'casual': 0.7, 'smart-casual': 0.9, 'formal': 0.8},
            'formal': {'casual': 0.3, 'smart-casual': 0.8, 'formal': 0.9}
        }
    
    def _load_category_compatibility_rules(self) -> Dict[str, Set[str]]:
        """Load category compatibility rules - which categories work together"""
        return {
            'top': {'bottom', 'shoes', 'outerwear', 'accessory'},
            'bottom': {'top', 'shoes', 'outerwear', 'accessory'},
            'shoes': {'top', 'bottom', 'outerwear'},
            'outerwear': {'top', 'bottom', 'shoes', 'accessory'},
            'accessory': {'top', 'bottom', 'outerwear', 'shoes'}
        }
    
    def calculate_compatibility(self, item1: Item, item2: Item) -> float:
        """
        Calculate compatibility between two items (how well they work together)
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            Compatibility score (0.0 to 1.0, higher = more compatible)
        """
        # Same item = no compatibility (can't wear same item twice)
        if item1.item_id == item2.item_id:
            return 0.0
        
        # Same category penalty (usually don't want two tops together)
        if item1.category == item2.category:
            return self._calculate_same_category_compatibility(item1, item2)
        
        # Different category compatibility
        return self._calculate_cross_category_compatibility(item1, item2)
    
    def _calculate_same_category_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate compatibility between items in same category"""
        # Generally low compatibility, but some exceptions
        base_score = 0.1
        
        # Exception: layering pieces
        layering_styles = {'cardigan', 'blazer', 'jacket', 'vest', 'sweater'}
        if (item1.category == 'outerwear' or 
            (item1.category == 'top' and any(style in (item1.style or '').lower() 
                                           for style in layering_styles))):
            base_score = 0.6
        
        # Exception: accessories can be combined
        if item1.category == 'accessory':
            base_score = 0.7
        
        return base_score
    
    def _calculate_cross_category_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate compatibility between items in different categories"""
        # Check if categories are compatible
        if item2.category not in self.category_rules.get(item1.category, set()):
            return 0.1
        
        # Base CLIP similarity (but weighted down since we want complementary, not similar)
        clip_score = 0.5
        if item1.embedding is not None and item2.embedding is not None:
            clip_similarity = np.dot(item1.embedding, item2.embedding)
            # Convert similarity to compatibility (inverse relationship for different categories)
            clip_score = 0.3 + (0.4 * clip_similarity)  # Scale to 0.3-0.7 range
        
        # Color compatibility
        color_score = self._calculate_color_compatibility(item1, item2)
        
        # Style/formality compatibility
        style_score = self._calculate_style_compatibility(item1, item2)
        
        # Season compatibility
        season_score = self._calculate_season_compatibility(item1, item2)
        
        # Pattern/texture compatibility (simple rules)
        pattern_score = self._calculate_pattern_compatibility(item1, item2)
        
        # Weighted combination
        compatibility = (
            clip_score * 0.2 +
            color_score * 0.35 +
            style_score * 0.25 +
            season_score * 0.1 +
            pattern_score * 0.1
        )
        
        return max(0.0, min(1.0, compatibility))
    
    def _calculate_color_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate color compatibility between items"""
        if not item1.color or not item2.color:
            return 0.6  # Neutral if color info missing
        
        color1 = item1.color.lower()
        color2 = item2.color.lower()
        
        # Check direct compatibility rules
        if color1 in self.color_rules:
            return self.color_rules[color1].get(color2, 0.4)
        
        # Fallback: neutral colors work with everything
        neutral_colors = {'white', 'black', 'gray', 'grey', 'beige', 'cream'}
        if color1 in neutral_colors or color2 in neutral_colors:
            return 0.8
        
        # Same color family but different shades
        if self._same_color_family(color1, color2):
            return 0.6
        
        return 0.4
    
    def _calculate_style_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate style/formality compatibility"""
        formality1 = item1.formality or 'casual'
        formality2 = item2.formality or 'casual'
        
        return self.style_rules.get(formality1, {}).get(formality2, 0.5)
    
    def _calculate_season_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate season compatibility"""
        season1 = item1.season or 'all'
        season2 = item2.season or 'all'
        
        # 'all' season items work with everything
        if season1 == 'all' or season2 == 'all':
            return 1.0
        
        # Same season items work well together
        if season1 == season2:
            return 1.0
        
        # Different seasons don't work well
        return 0.3
    
    def _calculate_pattern_compatibility(self, item1: Item, item2: Item) -> float:
        """Calculate pattern compatibility (simplified)"""
        # This would analyze patterns from metadata or style
        # For now, simple rules based on style names
        
        patterns = {'striped', 'plaid', 'floral', 'polka', 'geometric'}
        
        style1 = (item1.style or '').lower()
        style2 = (item2.style or '').lower()
        
        has_pattern1 = any(pattern in style1 for pattern in patterns)
        has_pattern2 = any(pattern in style2 for pattern in patterns)
        
        # If both have patterns, lower compatibility
        if has_pattern1 and has_pattern2:
            return 0.4
        
        # One patterned, one solid = good
        if has_pattern1 or has_pattern2:
            return 0.8
        
        # Both solid = neutral
        return 0.7
    
    def _same_color_family(self, color1: str, color2: str) -> bool:
        """Check if two colors are in the same family"""
        color_families = [
            {'blue', 'navy', 'royal', 'teal'},
            {'red', 'burgundy', 'maroon', 'pink'},
            {'green', 'olive', 'forest', 'mint'},
            {'brown', 'tan', 'khaki', 'beige'},
            {'gray', 'grey', 'silver', 'charcoal'}
        ]
        
        for family in color_families:
            if color1 in family and color2 in family:
                return True
        return False
    
    def find_complementary_items(self,
                                target_item: Item,
                                candidate_items: List[Item],
                                top_k: int = 10,
                                min_compatibility: float = 0.4,
                                exclude_categories: Optional[Set[str]] = None) -> List[Tuple[Item, float]]:
        """
        Find items that complement the target item
        
        Args:
            target_item: Item to find complements for
            candidate_items: Pool of candidate items
            top_k: Maximum number of items to return
            min_compatibility: Minimum compatibility threshold
            exclude_categories: Categories to exclude from results
            
        Returns:
            List of (item, compatibility_score) tuples, sorted by compatibility
        """
        if exclude_categories is None:
            exclude_categories = {target_item.category}  # Exclude same category by default
        
        complements = []
        
        for candidate in candidate_items:
            # Skip self
            if candidate.item_id == target_item.item_id:
                continue
            
            # Skip excluded categories
            if candidate.category in exclude_categories:
                continue
            
            compatibility = self.calculate_compatibility(target_item, candidate)
            
            if compatibility >= min_compatibility:
                complements.append((candidate, compatibility))
        
        # Sort by compatibility (descending) and return top_k
        complements.sort(key=lambda x: x[1], reverse=True)
        result = complements[:top_k]
        
        logger.info(f"Found {len(result)} complementary items for {target_item.item_id}")
        return result
    
    def score_outfit(self, items: List[Item]) -> float:
        """
        Score the overall compatibility of an outfit
        
        Args:
            items: List of items in the outfit
            
        Returns:
            Overall outfit compatibility score (0.0 to 1.0)
        """
        if len(items) < 2:
            return 0.0
        
        # Calculate pairwise compatibilities
        pairwise_scores = []
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score = self.calculate_compatibility(items[i], items[j])
                pairwise_scores.append(score)
        
        if not pairwise_scores:
            return 0.0
        
        # Use average of pairwise scores
        outfit_score = np.mean(pairwise_scores)
        
        # Apply outfit completeness bonus
        categories = {item.category for item in items}
        
        # Bonus for having core categories (top + bottom)
        if 'top' in categories and 'bottom' in categories:
            outfit_score *= 1.1
        
        # Bonus for having shoes
        if 'shoes' in categories:
            outfit_score *= 1.05
        
        # Small penalty for missing basic categories
        if len(categories) < 2:
            outfit_score *= 0.9
        
        return max(0.0, min(1.0, outfit_score))
    
    def suggest_outfit_completion(self,
                                 partial_outfit: List[Item],
                                 candidate_items: List[Item],
                                 target_categories: Optional[List[str]] = None) -> List[Tuple[Item, float]]:
        """
        Suggest items to complete a partial outfit
        
        Args:
            partial_outfit: Items already in the outfit
            candidate_items: Pool of items to choose from
            target_categories: Specific categories to suggest (e.g., ['shoes', 'outerwear'])
            
        Returns:
            List of suggested items with compatibility scores
        """
        if not partial_outfit:
            return []
        
        # Determine missing categories
        existing_categories = {item.category for item in partial_outfit}
        
        if target_categories is None:
            # Suggest common missing categories
            all_categories = {'top', 'bottom', 'shoes', 'outerwear'}
            target_categories = list(all_categories - existing_categories)
        
        suggestions = []
        
        for candidate in candidate_items:
            if candidate.category not in target_categories:
                continue
            
            # Calculate compatibility with all items in partial outfit
            compatibilities = []
            for outfit_item in partial_outfit:
                comp = self.calculate_compatibility(candidate, outfit_item)
                compatibilities.append(comp)
            
            # Use minimum compatibility (weakest link)
            min_compatibility = min(compatibilities) if compatibilities else 0.0
            
            if min_compatibility > 0.4:  # Threshold for suggestions
                suggestions.append((candidate, min_compatibility))
        
        # Sort by compatibility and return
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated {len(suggestions)} outfit completion suggestions")
        return suggestions


# Example usage and testing
def test_compatibility_engine():
    """Test the compatibility engine with sample data"""
    # Create sample items
    white_shirt = Item("shirt_001", "top", "white", "shirt", "smart-casual", embedding=np.random.rand(512))
    blue_jeans = Item("jeans_001", "bottom", "blue", "jeans", "casual", embedding=np.random.rand(512))
    black_dress = Item("dress_001", "top", "black", "dress", "formal", embedding=np.random.rand(512))
    brown_shoes = Item("shoes_001", "shoes", "brown", "loafers", "smart-casual", embedding=np.random.rand(512))
    red_shirt = Item("shirt_002", "top", "red", "shirt", "casual", embedding=np.random.rand(512))
    
    # Normalize embeddings
    for item in [white_shirt, blue_jeans, black_dress, brown_shoes, red_shirt]:
        item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    engine = CompatibilityEngine()
    
    # Test compatibility calculation
    compatibility = engine.calculate_compatibility(white_shirt, blue_jeans)
    print(f"White shirt + Blue jeans compatibility: {compatibility:.3f}")
    
    compatibility = engine.calculate_compatibility(white_shirt, red_shirt)
    print(f"White shirt + Red shirt compatibility: {compatibility:.3f}")
    
    # Test finding complementary items
    candidates = [blue_jeans, black_dress, brown_shoes, red_shirt]
    complements = engine.find_complementary_items(white_shirt, candidates, top_k=3)
    
    print(f"\nComplementary items for white shirt:")
    for item, comp in complements:
        print(f"  {item.item_id} ({item.color} {item.style}): {comp:.3f}")
    
    # Test outfit scoring
    outfit1 = [white_shirt, blue_jeans, brown_shoes]
    score1 = engine.score_outfit(outfit1)
    print(f"\nOutfit 1 (white shirt + blue jeans + brown shoes): {score1:.3f}")
    
    outfit2 = [white_shirt, red_shirt]  # Two tops
    score2 = engine.score_outfit(outfit2)
    print(f"Outfit 2 (white shirt + red shirt): {score2:.3f}")
    
    # Test outfit completion
    partial_outfit = [white_shirt]
    suggestions = engine.suggest_outfit_completion(partial_outfit, candidates, target_categories=['bottom', 'shoes'])
    
    print(f"\nOutfit completion suggestions for white shirt:")
    for item, comp in suggestions[:3]:
        print(f"  {item.item_id} ({item.category}): {comp:.3f}")


if __name__ == "__main__":
    test_compatibility_engine()