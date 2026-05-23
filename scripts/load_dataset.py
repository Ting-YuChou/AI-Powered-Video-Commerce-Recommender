#!/usr/bin/env python3
"""
Dataset Loading Script
======================

This script loads CSV dataset files from the Dataset directory into the system.
It can be run independently to load data before starting the application.

Usage:
    python scripts/load_dataset.py [--dataset-dir Dataset] [--limit-users N] [--limit-products N] [--limit-interactions N] [--limit-content N]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_commerce.common.config import Config
from video_commerce.data_plane.feature_store import FeatureStore
from video_commerce.ml.vector_search import VectorSearchEngine
from video_commerce.ml import data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to load dataset."""
    parser = argparse.ArgumentParser(description="Load CSV dataset into video commerce system")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Dataset",
        help="Directory containing CSV files (default: Dataset)"
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Limit number of users to load (default: load all)"
    )
    parser.add_argument(
        "--limit-products",
        type=int,
        default=None,
        help="Limit number of products to load (default: load all)"
    )
    parser.add_argument(
        "--limit-interactions",
        type=int,
        default=None,
        help="Limit number of interactions to load (default: load all)"
    )
    parser.add_argument(
        "--limit-content",
        type=int,
        default=None,
        help="Limit number of content items to load (default: load all)"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=None,
        help="Redis host override (default: use config/env)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=None,
        help="Redis port override (default: use config/env)"
    )
    parser.add_argument(
        "--vector-index-path",
        type=str,
        default="/tmp/vector_index.faiss",
        help="Path to FAISS vector index (default: /tmp/vector_index.faiss)"
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 60)
        logger.info("Dataset Loading Script")
        logger.info("=" * 60)

        # Load configuration
        config = Config()

        # Override config with command line arguments
        if args.redis_host is not None:
            config.redis_config.host = args.redis_host
        if args.redis_port is not None:
            config.redis_config.port = args.redis_port
        config.vector_config.index_path = args.vector_index_path

        # Initialize feature store
        logger.info("Initializing feature store...")
        feature_store = FeatureStore(config.redis_config)
        await feature_store.initialize()
        logger.info("Feature store initialized")

        # Initialize vector search
        logger.info("Initializing vector search...")
        vector_search = VectorSearchEngine(config.vector_config)
        await vector_search.load_index()
        logger.info("Vector search initialized")

        # Load dataset
        logger.info(f"Loading dataset from {args.dataset_dir}...")
        summary = await data.load_dataset_from_csv(
            dataset_dir=args.dataset_dir,
            feature_store=feature_store,
            vector_search=vector_search,
            limit_users=args.limit_users,
            limit_products=args.limit_products,
            limit_interactions=args.limit_interactions,
            limit_content=args.limit_content
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Dataset Loading Summary")
        logger.info("=" * 60)
        logger.info(f"Users loaded: {summary['users_loaded']}")
        logger.info(f"Products loaded: {summary['products_loaded']}")
        logger.info(f"Interactions loaded: {summary['interactions_loaded']}")
        logger.info(f"Content items loaded: {summary['content_items_loaded']}")
        logger.info(f"Loaded into feature store: {summary['loaded_into_feature_store']}")
        logger.info(f"Loaded into vector search: {summary['loaded_into_vector_search']}")
        logger.info("=" * 60)
        logger.info("Dataset loading completed successfully!")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
