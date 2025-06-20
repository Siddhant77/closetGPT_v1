"""
Integration tests for CLIPEmbedder and DataManager
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
from src.data.data_manager import DataManager


class TestIntegration(unittest.TestCase):
    """Integration tests for CLIPEmbedder and DataManager"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "integration_test.db"
        self.manager = DataManager(str(self.db_path))
        self.embedder = CLIPEmbedder()
        
        # Create test image
        self.test_image = self.temp_dir / "integration_test.jpg"
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        img.save(self.test_image)
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test complete pipeline: image -> embedding -> database"""
        # Generate embedding
        embedding = self.embedder.embed_image(self.test_image)
        self.assertIsNotNone(embedding)
        
        # Add item to database
        success = self.manager.add_item(
            item_id="integration_001",
            source="personal",
            filename="integration_test.jpg",
            category="top",
            color="blue"
        )
        self.assertTrue(success)
        
        # Add embedding to database
        success = self.manager.add_embedding("integration_001", embedding)
        self.assertTrue(success)
        
        # Retrieve item with embedding
        item = self.manager.get_item_with_embedding("integration_001")
        self.assertIsNotNone(item)
        self.assertIsNotNone(item['embedding'])
        
        # Verify embedding integrity
        np.testing.assert_array_almost_equal(item['embedding'], embedding)
        
        print("✅ Full Pipeline: Image -> Embedding -> Database integration working")
    
    def test_similarity_calculation(self):
        """Test similarity calculations between embeddings"""
        # Create two similar test images
        img1 = Image.new('RGB', (224, 224), color=(100, 100, 100))
        img2 = Image.new('RGB', (224, 224), color=(110, 110, 110))  # Slightly different
        img3 = Image.new('RGB', (224, 224), color=(255, 0, 0))     # Very different
        
        img1_path = self.temp_dir / "similar1.jpg"
        img2_path = self.temp_dir / "similar2.jpg" 
        img3_path = self.temp_dir / "different.jpg"
        
        img1.save(img1_path)
        img2.save(img2_path)
        img3.save(img3_path)
        
        # Generate embeddings
        emb1 = self.embedder.embed_image(img1_path)
        emb2 = self.embedder.embed_image(img2_path)
        emb3 = self.embedder.embed_image(img3_path)
        
        # Calculate similarities
        sim_1_2 = np.dot(emb1, emb2)  # Cosine similarity (since vectors are normalized)
        sim_1_3 = np.dot(emb1, emb3)
        
        # Similar images should have higher similarity than different ones
        self.assertGreater(sim_1_2, sim_1_3)
        print(f"✅ Similarity: Similar images sim={sim_1_2:.3f}, Different images sim={sim_1_3:.3f}")
    
    def test_batch_processing_pipeline(self):
        """Test batch processing from images to database"""
        # Create multiple test images
        test_images = []
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
            img_path = self.temp_dir / f"batch_test_{i}.jpg"
            img.save(img_path)
            test_images.append(img_path)
        
        # Generate batch embeddings
        embeddings = self.embedder.embed_batch(test_images, batch_size=2)
        self.assertEqual(len(embeddings), len(test_images))
        
        # Add items and embeddings to database
        for i, img_path in enumerate(test_images):
            # Add item
            success = self.manager.add_item(
                item_id=f"batch_{i:03d}",
                source="test_batch",
                filename=img_path.name,
                category="top",
                color=f"color_{i}"
            )
            self.assertTrue(success)
            
            # Add embedding
            embedding = embeddings[str(img_path)]
            success = self.manager.add_embedding(f"batch_{i:03d}", embedding)
            self.assertTrue(success)
        
        # Verify all items are in database
        items = self.manager.get_items(source="test_batch")
        self.assertEqual(len(items), 5)
        
        # Verify embeddings coverage
        all_embeddings = self.manager.get_embeddings()
        batch_embeddings = {k: v for k, v in all_embeddings.items() if k.startswith("batch_")}
        self.assertEqual(len(batch_embeddings), 5)
        
        print("✅ Batch Processing Pipeline: Multiple images processed and stored correctly")
    
    def test_end_to_end_similarity_search(self):
        """Test end-to-end similarity search workflow"""
        # Create reference image and similar/different images
        ref_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        similar_img = Image.new('RGB', (224, 224), color=(120, 120, 120))
        different_img = Image.new('RGB', (224, 224), color=(255, 0, 0))
        
        ref_path = self.temp_dir / "reference.jpg"
        similar_path = self.temp_dir / "similar.jpg"
        different_path = self.temp_dir / "different.jpg"
        
        ref_img.save(ref_path)
        similar_img.save(similar_path)
        different_img.save(different_path)
        
        # Process all images through pipeline
        images = [ref_path, similar_path, different_path]
        embeddings = self.embedder.embed_batch(images)
        
        # Add to database
        for i, img_path in enumerate(images):
            item_id = f"similarity_{i}"
            self.manager.add_item(item_id, "similarity_test", img_path.name, "top")
            self.manager.add_embedding(item_id, embeddings[str(img_path)])
        
        # Perform similarity search
        ref_embedding = embeddings[str(ref_path)]
        all_db_embeddings = self.manager.get_embeddings()
        
        similarities = {}
        for item_id, embedding in all_db_embeddings.items():
            if item_id.startswith("similarity_"):
                similarity = np.dot(ref_embedding, embedding)
                similarities[item_id] = similarity
        
        # Sort by similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Reference should be most similar to itself
        self.assertEqual(sorted_similarities[0][0], "similarity_0")  # Reference image
        
        # Similar image should be more similar than different image
        similar_item_sim = similarities["similarity_1"]
        different_item_sim = similarities["similarity_2"]
        self.assertGreater(similar_item_sim, different_item_sim)
        
        print(f"✅ Similarity Search: Reference (1.000), Similar ({similar_item_sim:.3f}), Different ({different_item_sim:.3f})")


if __name__ == '__main__':
    unittest.main()