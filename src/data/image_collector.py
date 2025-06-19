"""
Image Collector for ClosetGPT v1
Handles image intake, organization, and metadata extraction
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from PIL import Image, ExifTags
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCollector:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize image collector
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.personal_raw_dir = self.data_dir / "images" / "personal" / "raw"
        self.personal_processed_dir = self.data_dir / "images" / "personal" / "processed"
        self.metadata_file = self.data_dir / "personal_metadata.json"
        
        # Create directories
        self.personal_raw_dir.mkdir(parents=True, exist_ok=True)
        self.personal_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"ImageCollector initialized. Data dir: {self.data_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"items": {}, "last_updated": None}
    
    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to detect duplicates"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_exif_data(self, image_path: Path) -> Dict:
        """Extract EXIF data from image"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()
            
            exif_dict = {}
            for tag_id in exifdata:
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                exif_dict[tag] = data
            
            return exif_dict
        except Exception as e:
            logger.warning(f"Could not extract EXIF from {image_path}: {e}")
            return {}
    
    def add_image(self, 
                  source_path: str, 
                  category: str,
                  color: str,
                  style: Optional[str] = None,
                  formality: str = "casual",
                  season: str = "all",
                  notes: Optional[str] = None) -> str:
        """
        Add a new image to the collection
        
        Args:
            source_path: Path to source image
            category: Clothing category (top, bottom, shoes, outerwear, accessory)
            color: Primary color
            style: Style description (optional)
            formality: Formality level (casual, smart-casual, formal)
            season: Season (warm, cold, all)
            notes: Additional notes
            
        Returns:
            Final filename of processed image
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        # Generate hash to check for duplicates
        file_hash = self._get_file_hash(source_path)
        
        # Check for duplicates
        for item_id, item_data in self.metadata["items"].items():
            if item_data.get("file_hash") == file_hash:
                logger.warning(f"Duplicate image detected: {source_path}")
                return item_data["filename"]
        
        # Generate filename
        file_ext = source_path.suffix.lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Create standardized filename
        base_name = f"{category}_{color}"
        if style:
            base_name += f"_{style}"
        
        # Find next available number
        counter = 1
        while True:
            filename = f"{base_name}_{counter:03d}{file_ext}"
            if not (self.personal_raw_dir / filename).exists():
                break
            counter += 1
        
        # Copy to raw directory
        dest_path = self.personal_raw_dir / filename
        shutil.copy2(source_path, dest_path)
        
        # Extract EXIF data
        exif_data = self._extract_exif_data(dest_path)
        
        # Get image dimensions
        try:
            with Image.open(dest_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None
        
        # Create metadata entry
        item_id = f"personal_{counter:04d}"
        self.metadata["items"][item_id] = {
            "filename": filename,
            "category": category,
            "color": color,
            "style": style,
            "formality": formality,
            "season": season,
            "notes": notes,
            "file_hash": file_hash,
            "added_date": datetime.now().isoformat(),
            "source_path": str(source_path),
            "width": width,
            "height": height,
            "exif": exif_data
        }
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Added image: {filename}")
        return filename
    
    def batch_add_images(self, image_list: List[Dict]) -> List[str]:
        """
        Add multiple images at once
        
        Args:
            image_list: List of dicts with image info
                       Each dict should have: source_path, category, color, etc.
        
        Returns:
            List of processed filenames
        """
        processed_files = []
        
        for img_info in image_list:
            try:
                filename = self.add_image(**img_info)
                processed_files.append(filename)
            except Exception as e:
                logger.error(f"Failed to add {img_info.get('source_path')}: {e}")
                continue
        
        return processed_files
    
    def get_image_list(self, category: Optional[str] = None) -> List[Dict]:
        """
        Get list of all images with metadata
        
        Args:
            category: Filter by category (optional)
            
        Returns:
            List of image metadata dicts
        """
        items = []
        for item_id, item_data in self.metadata["items"].items():
            if category is None or item_data["category"] == category:
                items.append({
                    "item_id": item_id,
                    "path": str(self.personal_raw_dir / item_data["filename"]),
                    **item_data
                })
        
        return items
    
    def process_image_for_model(self, filename: str, target_size: Tuple[int, int] = (224, 224)) -> str:
        """
        Process image for model input (resize, normalize)
        
        Args:
            filename: Filename in raw directory
            target_size: Target dimensions
            
        Returns:
            Path to processed image
        """
        raw_path = self.personal_raw_dir / filename
        processed_path = self.personal_processed_dir / filename
        
        if processed_path.exists():
            return str(processed_path)
        
        try:
            with Image.open(raw_path) as img:
                # Convert to RGB
                img = img.convert("RGB")
                
                # Resize maintaining aspect ratio
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste centered
                new_img = Image.new("RGB", target_size, (255, 255, 255))
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # Save processed image
                new_img.save(processed_path, "JPEG", quality=90)
            
            logger.info(f"Processed image: {filename}")
            return str(processed_path)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return str(raw_path)  # Return original if processing fails
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        categories = {}
        colors = {}
        formality = {}
        
        for item_data in self.metadata["items"].values():
            # Count categories
            cat = item_data["category"]
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count colors
            color = item_data["color"]
            colors[color] = colors.get(color, 0) + 1
            
            # Count formality
            form = item_data["formality"]
            formality[form] = formality.get(form, 0) + 1
        
        return {
            "total_items": len(self.metadata["items"]),
            "categories": categories,
            "colors": colors,
            "formality": formality
        }

# Example usage functions
def quick_add_personal_items():
    """Example function for quickly adding personal items"""
    collector = ImageCollector()
    
    # Example batch add
    sample_items = [
        {
            "source_path": "/path/to/your/photos/shirt1.jpg",
            "category": "top",
            "color": "blue",
            "style": "tshirt",
            "formality": "casual",
            "season": "all"
        },
        # Add more items here
    ]
    
    processed = collector.batch_add_images(sample_items)
    print(f"Added {len(processed)} items")
    print("Collection stats:", collector.get_stats())

if __name__ == "__main__":
    # Test the collector
    collector = ImageCollector()
    print("Current collection stats:", collector.get_stats())