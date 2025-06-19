"""
CLIP Embedding Generator for ClosetGPT v1
Generates and caches CLIP embeddings for fashion images
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import json
from typing import Union, List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: CLIP model to use
            device: Device to run on ("auto", "cpu", "mps", "cuda")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        
        # Set to eval mode for inference
        self.model.eval()
        
        logger.info(f"CLIP model loaded on device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate CLIP embedding for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector (512-dim for base model)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # L2 normalize
                embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def embed_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 32, 
                    skip_existing: bool = True, embeddings_cache: Optional[Dict] = None,
                    save_every: int = 5, save_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple images in batches
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once
            skip_existing: Skip images that already have embeddings
            embeddings_cache: Pre-loaded embeddings to check against
            save_every: Save embeddings every N batches (0 to disable)
            save_path: Path to save incremental progress
            
        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        
        # Start with existing embeddings
        if embeddings_cache:
            embeddings.update(embeddings_cache)
        
        # Filter out existing embeddings if requested
        if skip_existing and embeddings_cache:
            paths_to_process = []
            skipped_count = 0
            for path in image_paths:
                path_str = str(path)
                if path_str in embeddings_cache:
                    skipped_count += 1
                else:
                    paths_to_process.append(path)
            
            if skipped_count > 0:
                logger.info(f"Skipped {skipped_count} existing embeddings")
            image_paths = paths_to_process
        
        if not image_paths:
            logger.info("All embeddings already exist")
            return embeddings
        
        total_batches = (len(image_paths) - 1) // batch_size + 1
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load batch images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(image)
                    valid_paths.append(str(path))
                except Exception as e:
                    logger.warning(f"Skipping {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # Process batch
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # L2 normalize
                    normalized_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Store embeddings
                for j, path in enumerate(valid_paths):
                    embeddings[path] = normalized_features[j].cpu().numpy()
                
                # Same-line progress update
                batch_num = i // batch_size + 1
                progress_msg = f"Processed batch {batch_num}/{total_batches} ({len(embeddings)} embeddings)"
                
                # Incremental save every N batches
                if save_every > 0 and save_path and batch_num % save_every == 0:
                    try:
                        self.save_embeddings(embeddings, save_path, silent=True)
                        progress_msg += f" [Saved batch #{batch_num}]"
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                
                print(f"\r{progress_msg}", end="", flush=True)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        # New line after progress updates
        if image_paths:
            print()  # New line after progress bar
        
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], save_path: Union[str, Path], silent: bool = False):
        """Save embeddings to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for efficiency
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

        if not silent:
            logger.info(f"Saved {len(embeddings)} embeddings to {save_path}")

    def load_embeddings(self, load_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load embeddings from file"""
        with open(load_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded {len(embeddings)} embeddings from {load_path}")
        return embeddings

def process_polyvore_images(data_dir: str = "data", batch_size: int = 32, save_every: int = 10):
    """
    Process all Polyvore images and generate embeddings
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for processing
        save_every: Save progress every N batches (prevents data loss on interruption)
    """
    data_path = Path(data_dir)
    image_dir = data_path / "images" / "polyvore"
    embeddings_dir = data_path / "embeddings" / "polyvore"
    embeddings_file = embeddings_dir / "clip_embeddings.pkl"
    
    # Load existing embeddings if they exist
    existing_embeddings = {}
    if embeddings_file.exists():
        logger.info("Loading existing embeddings...")
        embedder_temp = CLIPEmbedder()
        existing_embeddings = embedder_temp.load_embeddings(embeddings_file)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(image_dir.rglob(f"*{ext}"))
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return existing_embeddings
    
    logger.info(f"Found {len(image_paths)} total images")
    
    # Initialize embedder
    embedder = CLIPEmbedder()
    
    # Generate embeddings with incremental saving
    try:
        all_embeddings = embedder.embed_batch(
            image_paths, 
            batch_size=batch_size,
            skip_existing=True,
            embeddings_cache=existing_embeddings,
            save_every=save_every,
            save_path=embeddings_file
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been saved.")
        # Load what was saved
        if embeddings_file.exists():
            all_embeddings = embedder.load_embeddings(embeddings_file)
        else:
            all_embeddings = existing_embeddings
        return all_embeddings
    
    # Final save
    embedder.save_embeddings(all_embeddings, embeddings_file)
    
    new_count = len(all_embeddings) - len(existing_embeddings)
    logger.info(f"Generated {new_count} new embeddings. Total: {len(all_embeddings)}")
    
    return all_embeddings

if __name__ == "__main__":
    # Process Polyvore dataset
    embeddings = process_polyvore_images()
    print(f"Generated embeddings for {len(embeddings)} images")