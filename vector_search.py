"""
AI-Powered Video Commerce Recommender - Vector Search Engine
============================================================

This module implements FAISS-based vector similarity search for product recommendations.
It handles product embeddings, content-product similarity matching, and efficient
nearest neighbor search for real-time recommendations.
"""

import faiss
import numpy as np
import asyncio
import logging
import pickle
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path
import threading

# Local imports
from models import ProductData, CandidateProduct
from config import VectorConfig

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """
    FAISS-based vector search engine for product similarity matching.
    
    Features:
    - HNSW index for fast approximate search
    - Product embedding management
    - Content-to-product similarity search
    - Real-time index updates
    - GPU acceleration support
    """
    
    def __init__(self, config: VectorConfig):
        """Initialize the vector search engine."""
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # FAISS components
        self.index: Optional[faiss.Index] = None
        self.product_index_map: Dict[int, str] = {}  # FAISS index -> product_id
        self.product_embeddings: Dict[str, np.ndarray] = {}  # product_id -> embedding
        self.product_metadata: Dict[str, Dict[str, Any]] = {}  # product_id -> metadata
        
        # Index management
        self.index_lock = threading.RLock()
        self.is_loaded = False
        self.last_updated = 0
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info(f"VectorSearchEngine initialized with embedding_dim={self.embedding_dim}")
    
    async def load_index(self, force_rebuild: bool = False):
        """Load or build the FAISS index."""
        try:
            index_path = Path(self.config.index_path)
            metadata_path = index_path.with_suffix('.metadata.json')
            
            # Try to load existing index
            if not force_rebuild and index_path.exists() and metadata_path.exists():
                await self._load_existing_index(index_path, metadata_path)
            else:
                # Build new index with sample data
                await self._build_new_index()
            
            self.is_loaded = True
            logger.info(f"Vector search engine ready with {len(self.product_embeddings)} products")
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            # Create empty index as fallback
            await self._create_empty_index()
            raise
    
    async def _load_existing_index(self, index_path: Path, metadata_path: Path):
        """Load existing FAISS index from disk."""
        try:
            logger.info(f"Loading existing index from {index_path}")
            
            # Load FAISS index
            with self.index_lock:
                self.index = faiss.read_index(str(index_path))
                
                # Configure search parameters
                if hasattr(self.index, 'hnsw'):
                    self.index.hnsw.efSearch = self.config.hnsw_ef_search
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.product_index_map = {int(k): v for k, v in metadata['index_map'].items()}
                self.product_metadata = metadata['product_metadata']
                self.last_updated = metadata.get('last_updated', 0)
            
            # Reconstruct embeddings dictionary
            for faiss_idx, product_id in self.product_index_map.items():
                if faiss_idx < self.index.ntotal:
                    # Get embedding from index (approximation)
                    embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                    self.product_embeddings[product_id] = embedding
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            raise
    
    async def _build_new_index(self):
        """Build a new FAISS index with sample data."""
        try:
            logger.info("Building new FAISS index")
            
            # Create HNSW index
            with self.index_lock:
                if self.config.index_type.upper() == "HNSW":
                    self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.config.hnsw_m)
                    self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
                    self.index.hnsw.efSearch = self.config.hnsw_ef_search
                else:
                    # Fallback to flat index
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Generate sample product embeddings
            await self._create_sample_products()
            
            # Save index
            await self.save_index()
            
            logger.info("New index built successfully")
            
        except Exception as e:
            logger.error(f"Error building new index: {e}")
            raise
    
    async def _create_empty_index(self):
        """Create empty index as fallback."""
        with self.index_lock:
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.product_index_map = {}
            self.product_embeddings = {}
            self.product_metadata = {}
        
        self.is_loaded = True
        logger.warning("Created empty index as fallback")
    
    async def _create_sample_products(self, num_products: int = 1000):
        """Create sample products with embeddings for demo purposes."""
        try:
            logger.info(f"Creating {num_products} sample products")
            
            # Product categories and their typical embeddings
            categories = {
                'electronics': {'phone', 'laptop', 'headphones', 'camera', 'tablet'},
                'fashion': {'shirt', 'dress', 'shoes', 'jacket', 'bag'},
                'home': {'furniture', 'decoration', 'kitchen', 'bedding', 'lighting'},
                'beauty': {'skincare', 'makeup', 'perfume', 'haircare', 'wellness'},
                'sports': {'fitness', 'outdoor', 'sportswear', 'equipment', 'accessories'}
            }
            
            embeddings_to_add = []
            
            for i in range(num_products):
                product_id = f"prod_{i:06d}"
                
                # Select random category
                category = np.random.choice(list(categories.keys()))
                subcategory = np.random.choice(list(categories[category]))
                
                # Generate embedding (simulate CLIP-like embeddings)
                base_embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
                
                # Add category-specific patterns
                category_offset = hash(category) % self.embedding_dim
                base_embedding[category_offset:category_offset+10] += np.random.normal(0.5, 0.1, 10)
                
                # Normalize embedding
                norm = np.linalg.norm(base_embedding)
                if norm > 0:
                    base_embedding = base_embedding / norm
                
                # Store data
                self.product_embeddings[product_id] = base_embedding
                embeddings_to_add.append(base_embedding)
                
                # Store metadata
                self.product_metadata[product_id] = {
                    'title': f"{subcategory.title()} Product {i}",
                    'category': category,
                    'subcategory': subcategory,
                    'price': round(np.random.uniform(10, 500), 2),
                    'rating': round(np.random.uniform(3.0, 5.0), 1),
                    'brand': f"Brand {i % 50}",
                    'created_at': time.time()
                }
                
                # Update index mapping
                self.product_index_map[i] = product_id
            
            # Add embeddings to FAISS index
            if embeddings_to_add:
                embeddings_matrix = np.vstack(embeddings_to_add)
                with self.index_lock:
                    self.index.add(embeddings_matrix)
            
            self.last_updated = time.time()
            logger.info(f"Added {len(embeddings_to_add)} products to index")
            
        except Exception as e:
            logger.error(f"Error creating sample products: {e}")
            raise
    
    async def search_similar_products(
        self, 
        query_embedding: np.ndarray, 
        k: int = None,
        filter_categories: List[str] = None
    ) -> List[CandidateProduct]:
        """
        Search for similar products using content embedding.
        
        Args:
            query_embedding: Content embedding vector
            k: Number of results to return
            filter_categories: Optional category filter
            
        Returns:
            List of candidate products with similarity scores
        """
        if not self.is_loaded or self.index is None:
            logger.warning("Index not loaded, returning empty results")
            return []
        
        start_time = time.time()
        k = k or self.config.search_k
        
        try:
            # Ensure query embedding is the right shape and type
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            if query_embedding.shape[-1] != self.embedding_dim:
                logger.error(f"Query embedding dimension {query_embedding.shape} != {self.embedding_dim}")
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Perform search
            with self.index_lock:
                search_k = min(k * 2, self.index.ntotal) if self.index.ntotal > 0 else k
                scores, indices = self.index.search(query_embedding, search_k)
            
            # Convert results to candidate products
            candidates = []
            seen_products = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                product_id = self.product_index_map.get(idx)
                if not product_id or product_id in seen_products:
                    continue
                
                seen_products.add(product_id)
                
                # Apply similarity threshold
                if score < self.config.similarity_threshold:
                    continue
                
                # Apply category filter if specified
                metadata = self.product_metadata.get(product_id, {})
                if filter_categories and metadata.get('category') not in filter_categories:
                    continue
                
                candidate = CandidateProduct(
                    product_id=product_id,
                    content_similarity_score=float(score),
                    combined_score=float(score),
                    source="content_similarity"
                )
                candidates.append(candidate)
                
                # Stop when we have enough results
                if len(candidates) >= k:
                    break
            
            # Update search statistics
            search_time = time.time() - start_time
            self.search_stats['total_searches'] += 1
            self.search_stats['avg_search_time'] = (
                (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
                self.search_stats['total_searches']
            )
            
            logger.debug(f"Found {len(candidates)} similar products in {search_time:.3f}s")
            return candidates
            
        except Exception as e:
            logger.error(f"Error searching similar products: {e}")
            return []
    
    async def add_product_embedding(self, product_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a new product embedding to the index."""
        try:
            if not self.is_loaded:
                logger.warning("Index not loaded, cannot add product")
                return
            
            # Prepare embedding
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            embedding = embedding.reshape(-1).astype(np.float32)
            if embedding.shape[0] != self.embedding_dim:
                logger.error(f"Embedding dimension {embedding.shape[0]} != {self.embedding_dim}")
                return
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Add to index
            with self.index_lock:
                current_idx = self.index.ntotal
                self.index.add(embedding.reshape(1, -1))
                
                # Update mappings
                self.product_index_map[current_idx] = product_id
                self.product_embeddings[product_id] = embedding
                
                if metadata:
                    self.product_metadata[product_id] = metadata
            
            self.last_updated = time.time()
            logger.debug(f"Added product {product_id} to index at position {current_idx}")
            
        except Exception as e:
            logger.error(f"Error adding product embedding for {product_id}: {e}")
    
    async def add_content_embedding(self, content_id: str, embedding: np.ndarray):
        """Add content embedding (for future content-to-content similarity)."""
        try:
            # For now, we can store content embeddings separately
            # This could be extended to support content-to-content recommendations
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Store in a separate structure (could be extended to another FAISS index)
            logger.debug(f"Content embedding received for {content_id}")
            
        except Exception as e:
            logger.error(f"Error adding content embedding for {content_id}: {e}")
    
    async def get_product_metadata(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product metadata by ID."""
        return self.product_metadata.get(product_id)
    
    async def update_product_metadata(self, product_id: str, metadata: Dict[str, Any]):
        """Update product metadata."""
        if product_id in self.product_metadata:
            self.product_metadata[product_id].update(metadata)
            self.last_updated = time.time()
    
    async def remove_product(self, product_id: str):
        """Remove product from search (mark as inactive rather than rebuild index)."""
        try:
            # Find the product in our mappings
            faiss_idx = None
            for idx, pid in self.product_index_map.items():
                if pid == product_id:
                    faiss_idx = idx
                    break
            
            if faiss_idx is not None:
                # Mark as inactive rather than rebuilding entire index
                self.product_metadata[product_id] = self.product_metadata.get(product_id, {})
                self.product_metadata[product_id]['active'] = False
                
                logger.debug(f"Marked product {product_id} as inactive")
            
        except Exception as e:
            logger.error(f"Error removing product {product_id}: {e}")
    
    async def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            if not self.index:
                return
            
            index_path = Path(self.config.index_path)
            metadata_path = index_path.with_suffix('.metadata.json')
            
            # Create directory if it doesn't exist
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            with self.index_lock:
                faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'index_map': {str(k): v for k, v in self.product_index_map.items()},
                'product_metadata': self.product_metadata,
                'last_updated': self.last_updated,
                'embedding_dim': self.embedding_dim,
                'index_type': self.config.index_type,
                'total_products': len(self.product_embeddings)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved index with {len(self.product_embeddings)} products to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    async def rebuild_index(self, products: List[Dict[str, Any]] = None):
        """Rebuild the entire index (expensive operation)."""
        try:
            logger.info("Rebuilding vector search index")
            
            # Clear existing data
            with self.index_lock:
                self.index = None
                self.product_index_map = {}
                self.product_embeddings = {}
                self.product_metadata = {}
            
            # Rebuild index
            await self._build_new_index()
            
            # Add provided products if any
            if products:
                for product_data in products:
                    if 'embedding' in product_data and 'product_id' in product_data:
                        await self.add_product_embedding(
                            product_data['product_id'],
                            product_data['embedding'],
                            product_data.get('metadata', {})
                        )
            
            # Save rebuilt index
            await self.save_index()
            
            logger.info("Index rebuild completed")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
    
    async def get_random_products(self, k: int = 10) -> List[CandidateProduct]:
        """Get random products for diversity or cold start scenarios."""
        try:
            if not self.product_index_map:
                return []
            
            # Select random product IDs
            product_ids = list(self.product_index_map.values())
            selected_ids = np.random.choice(
                product_ids, 
                size=min(k, len(product_ids)), 
                replace=False
            )
            
            candidates = []
            for product_id in selected_ids:
                metadata = self.product_metadata.get(product_id, {})
                if metadata.get('active', True):  # Skip inactive products
                    candidate = CandidateProduct(
                        product_id=product_id,
                        popularity_score=np.random.uniform(0.1, 0.3),
                        combined_score=np.random.uniform(0.1, 0.3),
                        source="random"
                    )
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting random products: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'total_products': len(self.product_embeddings),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'last_updated': self.last_updated,
            'is_loaded': self.is_loaded,
            'search_stats': self.search_stats.copy(),
            'index_type': self.config.index_type
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector search engine."""
        try:
            status = {
                'status': 'healthy' if self.is_loaded else 'unhealthy',
                'index_loaded': self.index is not None,
                'total_products': len(self.product_embeddings),
                'index_size': self.index.ntotal if self.index else 0,
                'embedding_dim': self.embedding_dim,
                'last_updated': self.last_updated
            }
            
            # Test search functionality if loaded
            if self.is_loaded and self.index and len(self.product_embeddings) > 0:
                try:
                    # Create dummy query
                    dummy_query = np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
                    dummy_query = dummy_query / np.linalg.norm(dummy_query)
                    
                    start_time = time.time()
                    with self.index_lock:
                        scores, indices = self.index.search(dummy_query.reshape(1, -1), min(5, self.index.ntotal))
                    search_time = time.time() - start_time
                    
                    status['search_test_passed'] = True
                    status['search_test_time_ms'] = round(search_time * 1000, 2)
                    
                except Exception as test_error:
                    status['search_test_passed'] = False
                    status['search_test_error'] = str(test_error)
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'index_loaded': False
            }