"""
Core Data Structures for ClosetGPT v1
Shared data models used across the recommendation system
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Item:
    """Represents a clothing item"""
    item_id: str
    category: str
    color: Optional[str] = None
    style: Optional[str] = None
    formality: str = "casual"
    season: str = "all"
    source: str = "unknown"
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize fields after initialization"""
        # Normalize category
        valid_categories = {'top', 'bottom', 'shoes', 'outerwear', 'accessory', 'unknown'}
        if self.category not in valid_categories:
            self.category = 'unknown'
        
        # Normalize formality
        valid_formality = {'casual', 'smart-casual', 'formal'}
        if self.formality not in valid_formality:
            self.formality = 'casual'
        
        # Normalize season
        valid_seasons = {'warm', 'cold', 'all'}
        if self.season not in valid_seasons:
            self.season = 'all'
        
        # Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding for serialization)"""
        return {
            'item_id': self.item_id,
            'category': self.category,
            'color': self.color,
            'style': self.style,
            'formality': self.formality,
            'season': self.season,
            'source': self.source,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> 'Item':
        """Create Item from dictionary"""
        return cls(
            item_id=data['item_id'],
            category=data['category'],
            color=data.get('color'),
            style=data.get('style'),
            formality=data.get('formality', 'casual'),
            season=data.get('season', 'all'),
            source=data.get('source', 'unknown'),
            embedding=embedding,
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_db_record(cls, record: Dict[str, Any]) -> 'Item':
        """Create Item from database record"""
        return cls(
            item_id=record['item_id'],
            category=record['category'],
            color=record.get('color'),
            style=record.get('style'),
            formality=record.get('formality', 'casual'),
            season=record.get('season', 'all'),
            source=record.get('source', 'unknown'),
            embedding=record.get('embedding'),
            metadata=record.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation"""
        parts = [self.item_id]
        if self.color:
            parts.append(self.color)
        if self.style:
            parts.append(self.style)
        parts.append(f"({self.category})")
        return " ".join(parts)

@dataclass
class RecommendationContext:
    """Context for outfit recommendations"""
    weather: Optional[str] = None      # "hot", "cold", "mild"
    occasion: Optional[str] = None     # "casual", "work", "formal", "date"
    season: Optional[str] = None       # "warm", "cold", "all"
    user_preferences: Optional[Dict] = field(default_factory=dict)
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    location: Optional[str] = None     # "indoor", "outdoor"
    
    def __post_init__(self):
        """Validate context values"""
        valid_weather = {'hot', 'cold', 'mild', None}
        valid_occasion = {'casual', 'work', 'formal', 'date', 'athletic', 'sleep', None}
        valid_season = {'warm', 'cold', 'all', None}
        valid_time = {'morning', 'afternoon', 'evening', None}
        valid_location = {'indoor', 'outdoor', None}
        
        if self.weather not in valid_weather:
            self.weather = None
        if self.occasion not in valid_occasion:
            self.occasion = None
        if self.season not in valid_season:
            self.season = None
        if self.time_of_day not in valid_time:
            self.time_of_day = None
        if self.location not in valid_location:
            self.location = None
        
        if self.user_preferences is None:
            self.user_preferences = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            'weather': self.weather,
            'occasion': self.occasion,
            'season': self.season,
            'user_preferences': self.user_preferences,
            'time_of_day': self.time_of_day,
            'location': self.location
        }
    
    def matches_item(self, item: Item) -> bool:
        """Check if item matches this context"""
        # Season matching
        if self.season and item.season != 'all' and item.season != self.season:
            return False
        
        # Occasion/formality matching
        if self.occasion:
            occasion_formality_map = {
                'casual': ['casual'],
                'work': ['smart-casual', 'formal'],
                'formal': ['formal'],
                'date': ['smart-casual', 'formal'],
                'athletic': ['casual']
            }
            required_formalities = occasion_formality_map.get(self.occasion, ['casual', 'smart-casual', 'formal'])
            if item.formality not in required_formalities:
                return False
        
        return True

@dataclass
class Recommendation:
    """A single recommendation with metadata"""
    item: Item
    score: float
    reason: str
    recommendation_type: str  # "similar", "complementary", "outfit_completion"
    confidence: float = 1.0
    context_relevance: float = 1.0
    
    def __post_init__(self):
        """Validate recommendation data"""
        # Clamp scores to valid ranges
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.context_relevance = max(0.0, min(1.0, self.context_relevance))
        
        # Validate recommendation type
        valid_types = {'similar', 'complementary', 'outfit_completion', 'trending', 'personalized'}
        if self.recommendation_type not in valid_types:
            self.recommendation_type = 'similar'
    
    def overall_score(self) -> float:
        """Calculate overall recommendation score considering confidence and context"""
        return self.score * self.confidence * self.context_relevance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'item': self.item.to_dict(),
            'score': self.score,
            'reason': self.reason,
            'recommendation_type': self.recommendation_type,
            'confidence': self.confidence,
            'context_relevance': self.context_relevance,
            'overall_score': self.overall_score()
        }

@dataclass  
class Outfit:
    """Represents a complete outfit with metadata"""
    outfit_id: str
    items: List[Item]
    compatibility_score: Optional[float] = None
    context: Optional[RecommendationContext] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    source: str = "generated"  # "manual", "generated", "template"
    user_rating: Optional[float] = None  # 1-5 star rating from user
    wear_count: int = 0
    last_worn: Optional[datetime] = None
    notes: str = ""
    
    def __post_init__(self):
        """Validate outfit data"""
        if not self.items:
            raise ValueError("Outfit must contain at least one item")
        
        # Remove duplicate items
        seen_ids = set()
        unique_items = []
        for item in self.items:
            if item.item_id not in seen_ids:
                unique_items.append(item)
                seen_ids.add(item.item_id)
        self.items = unique_items
        
        # Validate source
        valid_sources = {'manual', 'generated', 'template', 'imported'}
        if self.source not in valid_sources:
            self.source = 'generated'
        
        # Clamp ratings
        if self.user_rating is not None:
            self.user_rating = max(1.0, min(5.0, self.user_rating))
        if self.compatibility_score is not None:
            self.compatibility_score = max(0.0, min(1.0, self.compatibility_score))
    
    @property
    def categories(self) -> List[str]:
        """Get list of categories in the outfit"""
        return [item.category for item in self.items]
    
    @property 
    def category_set(self) -> set:
        """Get unique categories in the outfit"""
        return {item.category for item in self.items}
    
    @property
    def is_complete(self) -> bool:
        """Check if outfit has core categories (top + bottom OR dress)"""
        categories = self.category_set
        return ('top' in categories and 'bottom' in categories) or \
               ('top' in categories and any('dress' in (item.style or '') for item in self.items))
    
    @property
    def dominant_formality(self) -> str:
        """Get the most formal level in the outfit"""
        formality_levels = {'casual': 1, 'smart-casual': 2, 'formal': 3}
        max_level = max(formality_levels.get(item.formality, 1) for item in self.items)
        return {1: 'casual', 2: 'smart-casual', 3: 'formal'}[max_level]
    
    @property
    def color_palette(self) -> List[str]:
        """Get list of colors in the outfit"""
        return [item.color for item in self.items if item.color]
    
    def add_item(self, item: Item) -> bool:
        """Add an item to the outfit if not already present"""
        if item.item_id not in {i.item_id for i in self.items}:
            self.items.append(item)
            return True
        return False
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the outfit"""
        initial_length = len(self.items)
        self.items = [item for item in self.items if item.item_id != item_id]
        return len(self.items) < initial_length
    
    def get_item_by_category(self, category: str) -> Optional[Item]:
        """Get the first item of a specific category"""
        for item in self.items:
            if item.category == category:
                return item
        return None
    
    def replace_item(self, old_item_id: str, new_item: Item) -> bool:
        """Replace an item in the outfit"""
        for i, item in enumerate(self.items):
            if item.item_id == old_item_id:
                self.items[i] = new_item
                return True
        return False
    
    def record_wear(self, date: Optional[datetime] = None):
        """Record that this outfit was worn"""
        self.wear_count += 1
        self.last_worn = date or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            'outfit_id': self.outfit_id,
            'items': [item.to_dict() for item in self.items],
            'compatibility_score': self.compatibility_score,
            'context': self.context.to_dict() if self.context else None,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'source': self.source,
            'user_rating': self.user_rating,
            'wear_count': self.wear_count,
            'last_worn': self.last_worn.isoformat() if self.last_worn else None,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embeddings: Optional[Dict[str, np.ndarray]] = None) -> 'Outfit':
        """Create Outfit from dictionary"""
        # Convert items
        items = []
        for item_data in data['items']:
            embedding = embeddings.get(item_data['item_id']) if embeddings else None
            item = Item.from_dict(item_data, embedding)
            items.append(item)
        
        # Convert context
        context = None
        if data.get('context'):
            context = RecommendationContext(**data['context'])
        
        # Convert dates
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        last_worn = datetime.fromisoformat(data['last_worn']) if data.get('last_worn') else None
        
        return cls(
            outfit_id=data['outfit_id'],
            items=items,
            compatibility_score=data.get('compatibility_score'),
            context=context,
            created_at=created_at,
            tags=data.get('tags', []),
            source=data.get('source', 'generated'),
            user_rating=data.get('user_rating'),
            wear_count=data.get('wear_count', 0),
            last_worn=last_worn,
            notes=data.get('notes', '')
        )
    
    def __str__(self) -> str:
        """String representation"""
        item_summary = ', '.join(f"{item.category}({item.color or 'unknown'})" for item in self.items)
        score_str = f", score: {self.compatibility_score:.2f}" if self.compatibility_score else ""
        return f"Outfit {self.outfit_id}: {item_summary}{score_str}"
    
    def __len__(self) -> int:
        """Number of items in outfit"""
        return len(self.items)


# Utility functions for working with data structures
def create_item_from_db_record(record: Dict[str, Any]) -> Item:
    """Convert database record to Item object (backward compatibility)"""
    return Item.from_db_record(record)

def items_to_outfit(items: List[Item], 
                   outfit_id: Optional[str] = None,
                   context: Optional[RecommendationContext] = None) -> Outfit:
    """Convert a list of items to an Outfit object"""
    if outfit_id is None:
        outfit_id = f"outfit_{hash(tuple(item.item_id for item in items)) % 1000000:06d}"
    
    return Outfit(
        outfit_id=outfit_id,
        items=items,
        context=context,
        source='generated'
    )

def filter_items_by_context(items: List[Item], context: RecommendationContext) -> List[Item]:
    """Filter items that match the given context"""
    return [item for item in items if context.matches_item(item)]