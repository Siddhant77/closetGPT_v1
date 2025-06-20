"""
Test cases for CLIPEmbedder
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.clip_embedder import CLIPEmbedder


class TestCLIPEmbedder(unittest.TestCase):
    """Test cases for CLIPEmbedder"""
    
    # @classmethod
    # def setUpClass(cls):
    #     """Set up test fixtures"""
    #     cls.temp_dir = Path(tempfile.mkdtemp())
    #     cls.embedder = CLIPEmbedder()
        
    #     # Create test images
    #     cls.test_images = []
    #     for i in range(3):
    #         img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
    #         img_path = cls.temp_dir / f"test_image_{i}.jpg"
    #         img.save(img_path)
    #         cls.test_images.append(img_path)

    @classmethod
    def setUpClass(cls):
        print(f"🔧 Setting up {cls.__name__}")
        try:
            cls.temp_dir = Path(tempfile.mkdtemp())
            print(f"📁 Temp dir: {cls.temp_dir}")
            
            cls.embedder = CLIPEmbedder()
            print(f"🤖 CLIP embedder initialized: {cls.embedder}")
            print(f"🔌 Device: {cls.embedder.device}")
            
            # Create test images
            cls.test_images = []
            for i in range(3):
                img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
                img_path = cls.temp_dir / f"test_image_{i}.jpg"
                img.save(img_path)
                cls.test_images.append(img_path)
            
            print(f"🖼️ Created {len(cls.test_images)} test images")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)
    
    def test_device_detection(self):
        """Test device detection works correctly"""
        device = self.embedder._get_device("auto")
        self.assertIn(device, ["cpu", "mps", "cuda"])
        print(f"✅ Device detection: {device}")
    
    def test_single_image_embedding(self):
        """Test embedding generation for single image"""
        embedding = self.embedder.embed_image(self.test_images[0])
        
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (512,))  # CLIP base model
        
        # Check normalization (L2 norm should be ~1)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
        print(f"✅ Single image embedding: shape {embedding.shape}, norm {norm:.6f}")
    
    def test_batch_embedding(self):
        """Test batch embedding generation"""
        embeddings = self.embedder.embed_batch(self.test_images, batch_size=2)
        
        self.assertEqual(len(embeddings), len(self.test_images))
        
        for img_path in self.test_images:
            self.assertIn(str(img_path), embeddings)
            embedding = embeddings[str(img_path)]
            self.assertEqual(embedding.shape, (512,))
            
            # Check normalization
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)
        
        print(f"✅ Batch embedding: {len(embeddings)} embeddings generated")
    
    def test_embedding_consistency(self):
        """Test that same image produces same embedding"""
        embedding1 = self.embedder.embed_image(self.test_images[0])
        embedding2 = self.embedder.embed_image(self.test_images[0])
        
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
        print("✅ Embedding consistency: Same image produces same embedding")
    
    def test_embedding_save_load(self):
        """Test saving and loading embeddings"""
        embeddings = self.embedder.embed_batch(self.test_images)
        save_path = self.temp_dir / "test_embeddings.pkl"
        
        # Save embeddings
        self.embedder.save_embeddings(embeddings, save_path)
        self.assertTrue(save_path.exists())
        
        # Load embeddings
        loaded_embeddings = self.embedder.load_embeddings(save_path)
        self.assertEqual(len(loaded_embeddings), len(embeddings))
        
        # Check consistency
        for path in embeddings:
            np.testing.assert_array_equal(embeddings[path], loaded_embeddings[path])
        
        print(f"✅ Save/Load: {len(loaded_embeddings)} embeddings preserved")
    
    def test_skip_existing_embeddings(self):
        """Test skipping existing embeddings functionality"""
        # Generate initial embeddings
        initial_embeddings = self.embedder.embed_batch([self.test_images[0]])
        
        # Try to generate again with skip_existing=True
        all_embeddings = self.embedder.embed_batch(
            self.test_images, 
            skip_existing=True, 
            embeddings_cache=initial_embeddings
        )
        
        # Should have all images but only computed new ones
        self.assertEqual(len(all_embeddings), len(self.test_images))
        
        # First image should be from cache (same reference)
        first_key = str(self.test_images[0])
        np.testing.assert_array_equal(
            all_embeddings[first_key], 
            initial_embeddings[first_key]
        )
        
        print("✅ Skip existing: Cached embeddings reused correctly")


if __name__ == '__main__':
    unittest.main()