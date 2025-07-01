"""
Similarity Engine for ClosetGPT v1
Finds similar items for style variations and substitutions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from data_structures import Item

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Engine for finding similar items for style variations"""
    
    def __init__(self):
        """Initialize similarity engine"""
        self.items_cache = {}
        logger.info("SimilarityEngine initialized")
    
    def calculate_similarity(self, item1: Item, item2: Item) -> float:
        """
        Calculate pure similarity between two items (how alike they are)
        
        Args:
            item1: First item
            item2: Second item
            
        Returns:
            Similarity score (0.0 to 1.0, higher = more similar)
        """
        if item1.embedding is None or item2.embedding is None:
            logger.warning(f"Missing embeddings for similarity calculation")
            return 0.0
        
        # Base CLIP similarity
        clip_similarity = np.dot(item1.embedding, item2.embedding)
        
        # Category boost - same category items are inherently more similar
        category_boost = 1.0 if item1.category == item2.category else 0.3
        
        # Style similarity
        style_similarity = self._calculate_style_similarity(item1, item2)
        
        # Color similarity  
        color_similarity = self._calculate_color_similarity(item1, item2)
        
        # Formality similarity
        formality_similarity = self._calculate_formality_similarity(item1, item2)
        
        # Weighted combination (CLIP is primary signal for similarity)
        similarity_score = (
            clip_similarity * 0.6 +
            style_similarity * 0.15 +
            color_similarity * 0.15 +
            formality_similarity * 0.1
        ) * category_boost
        
        return max(0.0, min(1.0, similarity_score))
    
    def _calculate_style_similarity(self, item1: Item, item2: Item) -> float:
        """Calculate style similarity"""
        if not item1.style or not item2.style:
            return 0.5  # Neutral if missing style info
        
        # Exact match
        if item1.style.lower() == item2.style.lower():
            return 1.0
        
        # Partial matches for similar styles
        style_groups = {
            'shirts': ['shirt', 'blouse', 'top', 'tee', 'tank'],
            'pants': ['pants', 'jeans', 'trousers', 'chinos'],
            'dresses': ['dress', 'gown', 'frock'],
            'outerwear': ['jacket', 'coat', 'blazer', 'cardigan'],
            'casual_shoes': ['sneakers', 'flats', 'loafers', 'sandals'],
            'formal_shoes': ['heels', 'pumps', 'oxfords', 'boots']
        }
        
        style1 = item1.style.lower()
        style2 = item2.style.lower()
        
        for group in style_groups.values():
            if style1 in group and style2 in group:
                return 0.8
        
        return 0.2
    
    def _calculate_color_similarity(self, item1: Item, item2: Item) -> float:
        """Calculate color similarity"""
        if not item1.color or not item2.color:
            return 0.5
        
        color1 = item1.color.lower()
        color2 = item2.color.lower()
        
        # Exact match
        if color1 == color2:
            return 1.0
        
        # Color family similarities
        color_families = {
            'neutral': ['white', 'black', 'gray', 'grey', 'beige', 'cream', 'tan'],
            'blue': ['blue', 'navy', 'royal', 'teal', 'turquoise'],
            'red': ['red', 'burgundy', 'maroon', 'crimson', 'pink'],
            'green': ['green', 'olive', 'forest', 'mint', 'lime'],
            'brown': ['brown', 'tan', 'khaki', 'camel', 'chocolate'],
            'warm': ['red', 'orange', 'yellow', 'pink', 'coral'],
            'cool': ['blue', 'green', 'purple', 'teal', 'mint']
        }
        
        # Check if colors are in same family
        for family in color_families.values():
            if color1 in family and color2 in family:
                return 0.7
        
        return 0.1
    
    def _calculate_formality_similarity(self, item1: Item, item2: Item) -> float:
        """Calculate formality level similarity"""
        formality_levels = {'casual': 1, 'smart-casual': 2, 'formal': 3}
        
        level1 = formality_levels.get(item1.formality, 1)
        level2 = formality_levels.get(item2.formality, 1)
        
        # Perfect match
        if level1 == level2:
            return 1.0
        
        # Adjacent levels (casual-smart, smart-formal)
        if abs(level1 - level2) == 1:
            return 0.6
        
        # Distant levels (casual-formal)
        return 0.2
    
    def find_similar_items(self, 
                          target_item: Item, 
                          candidate_items: List[Item],
                          top_k: int = 10,
                          min_similarity: float = 0.3,
                          same_category_only: bool = True) -> List[Tuple[Item, float]]:
        """
        Find items similar to the target item
        
        Args:
            target_item: Item to find similarities for
            candidate_items: Pool of items to search in
            top_k: Maximum number of similar items to return
            min_similarity: Minimum similarity threshold
            same_category_only: Whether to only consider same category items
            
        Returns:
            List of (item, similarity_score) tuples, sorted by similarity (descending)
        """
        similarities = []
        
        for candidate in candidate_items:
            # Skip self-comparison
            if candidate.item_id == target_item.item_id:
                continue
            
            # Category filter
            if same_category_only and candidate.category != target_item.category:
                continue
            
            similarity = self.calculate_similarity(target_item, candidate)
            
            if similarity >= min_similarity:
                similarities.append((candidate, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        result = similarities[:top_k]
        
        logger.info(f"Found {len(result)} similar items for {target_item.item_id}")
        return result
    
    def find_style_variations(self,
                             target_item: Item,
                             candidate_items: List[Item],
                             variation_type: str = "color",
                             top_k: int = 5) -> List[Tuple[Item, float]]:
        """
        Find specific style variations of an item
        
        Args:
            target_item: Base item
            candidate_items: Pool of candidates
            variation_type: Type of variation ("color", "style", "formality")
            top_k: Number of variations to return
            
        Returns:
            List of variations sorted by similarity
        """
        variations = []
        
        for candidate in candidate_items:
            if candidate.item_id == target_item.item_id:
                continue
            
            # Must be same category for style variations
            if candidate.category != target_item.category:
                continue
            
            # Check if it's a valid variation
            is_variation = False
            
            if variation_type == "color":
                # Same style/formality, different color
                is_variation = (
                    candidate.color != target_item.color and
                    candidate.style == target_item.style and
                    candidate.formality == target_item.formality
                )
            elif variation_type == "style":
                # Same color/formality, different style
                is_variation = (
                    candidate.style != target_item.style and
                    candidate.color == target_item.color and
                    candidate.formality == target_item.formality
                )
            elif variation_type == "formality":
                # Same color/style, different formality
                is_variation = (
                    candidate.formality != target_item.formality and
                    candidate.color == target_item.color and
                    candidate.style == target_item.style
                )
            
            if is_variation:
                similarity = self.calculate_similarity(target_item, candidate)
                variations.append((candidate, similarity))
        
        # Sort by similarity and return top_k
        variations.sort(key=lambda x: x[1], reverse=True)
        result = variations[:top_k]
        
        logger.info(f"Found {len(result)} {variation_type} variations for {target_item.item_id}")
        return result
    
    def batch_similarity_matrix(self, items: List[Item]) -> np.ndarray:
        """
        Calculate similarity matrix for a batch of items
        
        Args:
            items: List of items to compare
            
        Returns:
            NxN similarity matrix where matrix[i][j] = similarity(items[i], items[j])
        """
        n = len(items)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                elif i < j:  # Calculate only upper triangle
                    sim = self.calculate_similarity(items[i], items[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Matrix is symmetric
        
        logger.info(f"Calculated similarity matrix for {n} items")
        return similarity_matrix


def create_item_from_db_record(record: Dict) -> Item:
    """Convert database record to Item object"""
    return Item(
        item_id=record['item_id'],
        category=record['category'],
        color=record.get('color'),
        style=record.get('style'),
        formality=record.get('formality', 'casual'),
        season=record.get('season', 'all'),
        embedding=record.get('embedding'),
        metadata=record.get('metadata', {})
    )


# Example usage functions
def test_similarity_engine():
    """Test the similarity engine with sample data"""
    # Create sample items
    blue_shirt = Item("shirt_001", "top", "blue", "shirt", "casual", embedding=np.random.rand(512))
    navy_shirt = Item("shirt_002", "top", "navy", "shirt", "casual", embedding=np.random.rand(512))
    red_dress = Item("dress_001", "top", "red", "dress", "formal", embedding=np.random.rand(512))
    blue_jeans = Item("jeans_001", "bottom", "blue", "jeans", "casual", embedding=np.random.rand(512))
    
    # Normalize embeddings
    for item in [blue_shirt, navy_shirt, red_dress, blue_jeans]:
        item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    engine = SimilarityEngine()
    
    # Test similarity calculation
    similarity = engine.calculate_similarity(blue_shirt, navy_shirt)
    print(f"Blue shirt vs Navy shirt similarity: {similarity:.3f}")
    
    similarity = engine.calculate_similarity(blue_shirt, red_dress)
    print(f"Blue shirt vs Red dress similarity: {similarity:.3f}")
    
    # Test finding similar items
    candidates = [navy_shirt, red_dress, blue_jeans]
    similar_items = engine.find_similar_items(blue_shirt, candidates, top_k=3)
    
    print(f"\nSimilar items to blue shirt:")
    for item, sim in similar_items:
        print(f"  {item.item_id} ({item.color} {item.style}): {sim:.3f}")


if __name__ == "__main__":
    test_similarity_engine()