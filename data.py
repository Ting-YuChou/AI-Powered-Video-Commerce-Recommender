"""
AI-Powered Video Commerce Recommender - Sample Data Generation
===============================================================

This module generates realistic sample data for development, testing, and demo purposes.
It creates users, products, interactions, and content features that simulate a real
video commerce environment.
"""

import asyncio
import json
import random
import time
import numpy as np
import logging
import pandas as pd
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Local imports
from models import UserFeatures, ContentFeatures, ProductData, InteractionType
from feature_store import FeatureStore
from vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """
    Generates realistic sample data for the video commerce system.
    Creates users, products, content, and interactions that simulate
    real-world usage patterns.
    """
    
    def __init__(self):
        # Product categories and their characteristics
        self.product_categories = {
            'electronics': {
                'subcategories': ['smartphones', 'laptops', 'headphones', 'cameras', 'tablets', 'smartwatches'],
                'brands': ['Apple', 'Samsung', 'Sony', 'Canon', 'Dell', 'HP', 'Bose', 'JBL'],
                'price_range': (50, 2000),
                'popularity_weight': 1.2
            },
            'fashion': {
                'subcategories': ['dresses', 'shoes', 'bags', 'jewelry', 'shirts', 'jackets'],
                'brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Gucci', 'Prada', 'Levi\'s', 'Calvin Klein'],
                'price_range': (20, 800),
                'popularity_weight': 1.5
            },
            'home': {
                'subcategories': ['furniture', 'decoration', 'kitchen', 'bedding', 'lighting', 'storage'],
                'brands': ['IKEA', 'West Elm', 'Pottery Barn', 'CB2', 'Williams Sonoma', 'Crate & Barrel'],
                'price_range': (30, 1200),
                'popularity_weight': 0.8
            },
            'beauty': {
                'subcategories': ['skincare', 'makeup', 'perfume', 'haircare', 'wellness', 'tools'],
                'brands': ['L\'Oreal', 'Maybelline', 'Chanel', 'MAC', 'Clinique', 'Estee Lauder'],
                'price_range': (15, 300),
                'popularity_weight': 1.1
            },
            'sports': {
                'subcategories': ['fitness', 'outdoor', 'sportswear', 'equipment', 'accessories', 'supplements'],
                'brands': ['Nike', 'Adidas', 'Under Armour', 'Patagonia', 'North Face', 'Lululemon'],
                'price_range': (25, 600),
                'popularity_weight': 0.9
            },
            'books': {
                'subcategories': ['fiction', 'non-fiction', 'textbooks', 'children', 'comics', 'audiobooks'],
                'brands': ['Penguin', 'HarperCollins', 'Random House', 'Simon & Schuster', 'Macmillan'],
                'price_range': (10, 80),
                'popularity_weight': 0.6
            },
            'toys': {
                'subcategories': ['action figures', 'dolls', 'puzzles', 'games', 'educational', 'outdoor'],
                'brands': ['LEGO', 'Mattel', 'Hasbro', 'Fisher-Price', 'Playmobil', 'VTech'],
                'price_range': (15, 200),
                'popularity_weight': 0.7
            }
        }
        
        # User persona templates
        self.user_personas = {
            'tech_enthusiast': {
                'preferred_categories': ['electronics', 'books'],
                'price_sensitivity': 0.3,
                'engagement_level': 'high',
                'session_length_range': (300, 1200)  # 5-20 minutes
            },
            'fashion_lover': {
                'preferred_categories': ['fashion', 'beauty'],
                'price_sensitivity': 0.6,
                'engagement_level': 'high',
                'session_length_range': (180, 800)
            },
            'home_decorator': {
                'preferred_categories': ['home', 'beauty'],
                'price_sensitivity': 0.4,
                'engagement_level': 'medium',
                'session_length_range': (240, 600)
            },
            'fitness_fanatic': {
                'preferred_categories': ['sports', 'beauty'],
                'price_sensitivity': 0.5,
                'engagement_level': 'medium',
                'session_length_range': (200, 500)
            },
            'bargain_hunter': {
                'preferred_categories': ['fashion', 'home', 'books'],
                'price_sensitivity': 0.8,
                'engagement_level': 'medium',
                'session_length_range': (300, 900)
            },
            'casual_shopper': {
                'preferred_categories': ['electronics', 'toys', 'books'],
                'price_sensitivity': 0.7,
                'engagement_level': 'low',
                'session_length_range': (120, 400)
            }
        }
        
        # Content types and their characteristics
        self.content_types = {
            'product_review': {
                'interaction_boost': 1.3,
                'conversion_boost': 1.5,
                'typical_length': (180, 600)  # 3-10 minutes
            },
            'unboxing': {
                'interaction_boost': 1.2,
                'conversion_boost': 1.4,
                'typical_length': (300, 900)
            },
            'tutorial': {
                'interaction_boost': 1.1,
                'conversion_boost': 1.2,
                'typical_length': (600, 1800)
            },
            'haul': {
                'interaction_boost': 1.4,
                'conversion_boost': 1.3,
                'typical_length': (480, 1200)
            },
            'comparison': {
                'interaction_boost': 1.2,
                'conversion_boost': 1.6,
                'typical_length': (300, 900)
            }
        }
    
    def generate_sample_users(self, num_users: int = 1000) -> List[UserFeatures]:
        """Generate realistic user profiles with diverse characteristics."""
        try:
            logger.info(f"Generating {num_users} sample users")
            
            users = []
            persona_names = list(self.user_personas.keys())
            
            for i in range(num_users):
                user_id = f"user_{i:06d}"
                
                # Assign persona (with some randomness)
                persona_name = random.choice(persona_names)
                persona = self.user_personas[persona_name]
                
                # Generate user characteristics based on persona
                total_interactions = max(1, int(np.random.exponential(50) * 
                                              (2 if persona['engagement_level'] == 'high' else
                                               1.5 if persona['engagement_level'] == 'medium' else 1)))
                
                avg_session_length = np.random.uniform(*persona['session_length_range'])
                
                # Price sensitivity with some noise
                price_sensitivity = max(0.1, min(0.9, 
                    persona['price_sensitivity'] + np.random.normal(0, 0.1)))
                
                # Calculate realistic CTR and conversion rates
                base_ctr = 0.08 if persona['engagement_level'] == 'high' else 0.05 if persona['engagement_level'] == 'medium' else 0.03
                ctr = max(0.01, min(0.25, base_ctr + np.random.normal(0, 0.02)))
                
                base_cvr = 0.06 if price_sensitivity < 0.5 else 0.03
                cvr = max(0.005, min(0.15, base_cvr + np.random.normal(0, 0.01)))
                
                # Last active time (most users active recently)
                days_ago = np.random.exponential(2)  # Average 2 days ago
                last_active = time.time() - (days_ago * 24 * 3600)
                
                user = UserFeatures(
                    user_id=user_id,
                    total_interactions=total_interactions,
                    avg_session_length=avg_session_length,
                    preferred_categories=persona['preferred_categories'].copy(),
                    price_sensitivity=price_sensitivity,
                    click_through_rate=ctr,
                    conversion_rate=cvr,
                    last_active=last_active,
                    demographics={
                        'persona': persona_name,
                        'engagement_level': persona['engagement_level'],
                        'age_group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                        'location': random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR']),
                        'device_preference': random.choice(['mobile', 'desktop', 'tablet'])
                    }
                )
                
                users.append(user)
            
            logger.info(f"Generated {len(users)} user profiles")
            return users
            
        except Exception as e:
            logger.error(f"Error generating sample users: {e}")
            return []
    
    def generate_sample_products(self, num_products: int = 2000) -> List[ProductData]:
        """Generate realistic product catalog with diverse items."""
        try:
            logger.info(f"Generating {num_products} sample products")
            
            products = []
            
            for i in range(num_products):
                product_id = f"prod_{i:06d}"
                
                # Choose random category
                category = random.choice(list(self.product_categories.keys()))
                category_info = self.product_categories[category]
                
                # Choose subcategory and brand
                subcategory = random.choice(category_info['subcategories'])
                brand = random.choice(category_info['brands'])
                
                # Generate price within category range
                min_price, max_price = category_info['price_range']
                price = round(np.random.uniform(min_price, max_price), 2)
                
                # Generate realistic rating (biased toward higher ratings)
                rating = max(1.0, min(5.0, np.random.beta(7, 2) * 5))
                rating = round(rating, 1)
                
                # Number of reviews (correlated with rating and age)
                base_reviews = max(1, int(np.random.exponential(100)))
                review_multiplier = 1.2 if rating > 4.0 else 0.8 if rating < 3.0 else 1.0
                num_reviews = max(1, int(base_reviews * review_multiplier))
                
                # Generate product title
                adjectives = ['Premium', 'Professional', 'Classic', 'Modern', 'Luxury', 'Essential', 'Smart', 'Wireless']
                adjective = random.choice(adjectives)
                title = f"{brand} {adjective} {subcategory.title()}"
                
                # Generate description
                features = ['high-quality', 'durable', 'stylish', 'innovative', 'user-friendly', 'reliable']
                selected_features = random.sample(features, random.randint(2, 4))
                description = f"This {adjective.lower()} {subcategory} offers {', '.join(selected_features)} design perfect for modern lifestyle."
                
                # Stock status (mostly in stock)
                in_stock = random.random() > 0.05  # 95% in stock
                
                # Generate tags
                base_tags = [category, subcategory, brand.lower()]
                additional_tags = ['bestseller', 'new-arrival', 'trending', 'eco-friendly', 'limited-edition']
                tags = base_tags + random.sample(additional_tags, random.randint(0, 2))
                
                # Create timestamps
                created_days_ago = np.random.exponential(365)  # Average 1 year ago
                created_at = time.time() - (created_days_ago * 24 * 3600)
                updated_at = created_at + np.random.uniform(0, time.time() - created_at)
                
                product = ProductData(
                    product_id=product_id,
                    title=title,
                    description=description,
                    price=price,
                    currency="USD",
                    category=category,
                    brand=brand,
                    image_url=f"https://example.com/images/{product_id}.jpg",
                    rating=rating,
                    num_reviews=num_reviews,
                    in_stock=in_stock,
                    tags=tags,
                    created_at=created_at,
                    updated_at=updated_at
                )
                
                products.append(product)
            
            logger.info(f"Generated {len(products)} products across {len(self.product_categories)} categories")
            return products
            
        except Exception as e:
            logger.error(f"Error generating sample products: {e}")
            return []
    
    def generate_sample_interactions(
        self, 
        users: List[UserFeatures], 
        products: List[ProductData], 
        num_interactions: int = 10000
    ) -> List[Dict[str, Any]]:
        """Generate realistic user-product interactions."""
        try:
            logger.info(f"Generating {num_interactions} sample interactions")
            
            interactions = []
            
            # Create product lookup for easy access
            product_dict = {p.product_id: p for p in products}
            
            for _ in range(num_interactions):
                # Choose random user
                user = random.choice(users)
                
                # Choose product (biased toward user's preferred categories)
                if random.random() < 0.7 and user.preferred_categories:
                    # Choose from preferred categories
                    preferred_category = random.choice(user.preferred_categories)
                    category_products = [p for p in products if p.category == preferred_category]
                    if category_products:
                        product = random.choice(category_products)
                    else:
                        product = random.choice(products)
                else:
                    # Choose random product
                    product = random.choice(products)
                
                # Generate interaction type based on user characteristics
                interaction_type = self._choose_interaction_type(user, product)
                
                # Generate timestamp (biased toward recent interactions)
                days_ago = np.random.exponential(7)  # Average 1 week ago
                timestamp = time.time() - (days_ago * 24 * 3600)
                
                # Generate context
                context = self._generate_interaction_context(user, product, interaction_type)
                
                interaction = {
                    'user_id': user.user_id,
                    'product_id': product.product_id,
                    'action': interaction_type,
                    'timestamp': timestamp,
                    'context': context
                }
                
                # Add purchase value for purchase interactions
                if interaction_type == InteractionType.PURCHASE.value:
                    # Quantity (mostly 1, sometimes more)
                    quantity = 1 if random.random() < 0.8 else random.randint(2, 5)
                    interaction['value'] = product.price * quantity
                    interaction['quantity'] = quantity
                
                interactions.append(interaction)
            
            # Sort by timestamp
            interactions.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Generated {len(interactions)} interactions")
            return interactions
            
        except Exception as e:
            logger.error(f"Error generating sample interactions: {e}")
            return []
    
    def _choose_interaction_type(self, user: UserFeatures, product: ProductData) -> str:
        """Choose interaction type based on user and product characteristics."""
        # Base probabilities
        probs = {
            InteractionType.VIEW.value: 0.60,
            InteractionType.CLICK.value: 0.25,
            InteractionType.ADD_TO_CART.value: 0.08,
            InteractionType.PURCHASE.value: 0.04,
            InteractionType.FAVORITE.value: 0.02,
            InteractionType.SHARE.value: 0.01
        }
        
        # Adjust based on user characteristics
        engagement_multiplier = {
            'high': 1.3,
            'medium': 1.0,
            'low': 0.7
        }.get(user.demographics.get('engagement_level', 'medium'), 1.0)
        
        # Adjust based on price sensitivity and product price
        if product.price > 200 and user.price_sensitivity > 0.7:
            # Price-sensitive user with expensive product
            probs[InteractionType.PURCHASE.value] *= 0.5
            probs[InteractionType.ADD_TO_CART.value] *= 0.7
        
        # Adjust based on product rating
        if product.rating and product.rating > 4.5:
            probs[InteractionType.PURCHASE.value] *= 1.3
            probs[InteractionType.FAVORITE.value] *= 1.5
        
        # Apply engagement multiplier
        for action in [InteractionType.CLICK.value, InteractionType.PURCHASE.value]:
            probs[action] *= engagement_multiplier
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        normalized_probs = {k: v/total_prob for k, v in probs.items()}
        
        # Choose action
        rand = random.random()
        cumulative = 0
        
        for action, prob in normalized_probs.items():
            cumulative += prob
            if rand <= cumulative:
                return action
        
        return InteractionType.VIEW.value  # Fallback
    
    def _generate_interaction_context(
        self, 
        user: UserFeatures, 
        product: ProductData, 
        interaction_type: str
    ) -> Dict[str, Any]:
        """Generate realistic context for an interaction."""
        context = {
            'device': user.demographics.get('device_preference', 'mobile'),
            'location': user.demographics.get('location', 'US'),
            'session_position': random.randint(1, 15),
            'time_on_page': random.randint(10, 300),
            'referrer': random.choice(['search', 'social', 'direct', 'recommendation']),
            'product_category': product.category,
            'product_price_range': 'budget' if product.price < 50 else 'mid' if product.price < 200 else 'premium'
        }
        
        # Add interaction-specific context
        if interaction_type in [InteractionType.PURCHASE.value, InteractionType.ADD_TO_CART.value]:
            context['checkout_step'] = interaction_type
            context['payment_method'] = random.choice(['card', 'paypal', 'apple_pay'])
        
        return context
    
    def generate_sample_content(self, num_content: int = 500) -> List[ContentFeatures]:
        """Generate sample video content features."""
        try:
            logger.info(f"Generating {num_content} sample content items")
            
            content_list = []
            
            for i in range(num_content):
                content_id = f"content_{i:06d}"
                
                # Choose content type and category
                content_type = random.choice(list(self.content_types.keys()))
                category = random.choice(list(self.product_categories.keys()))
                
                # Generate video duration
                min_duration, max_duration = self.content_types[content_type]['typical_length']
                duration = random.randint(min_duration, max_duration)
                
                # Generate visual embedding (512-dimensional)
                # Simulate CLIP-like embeddings with category-specific patterns
                base_embedding = np.random.normal(0, 0.1, 512)
                
                # Add category-specific patterns
                category_hash = hash(category) % 512
                base_embedding[category_hash:category_hash+20] += np.random.normal(0.3, 0.1, 20)
                
                # Add content type patterns
                type_hash = hash(content_type) % 512
                base_embedding[type_hash:type_hash+15] += np.random.normal(0.2, 0.1, 15)
                
                # Normalize embedding
                base_embedding = base_embedding / np.linalg.norm(base_embedding)
                
                # Generate detected objects based on category
                objects = self._generate_detected_objects(category)
                
                # Generate extracted text
                extracted_text = self._generate_extracted_text(content_type, category)
                
                # Generate product mentions
                product_mentions = [f"{category} product", f"new {category}"]
                
                # Generate category scores
                category_scores = {cat: 0.1 for cat in self.product_categories.keys()}
                category_scores[category] = random.uniform(0.7, 0.95)
                
                # Add some noise to other categories
                for cat in category_scores:
                    if cat != category:
                        category_scores[cat] = random.uniform(0.05, 0.3)
                
                content = ContentFeatures(
                    content_id=content_id,
                    visual_embedding=base_embedding.tolist(),
                    duration_seconds=duration,
                    detected_objects=objects,
                    extracted_text=extracted_text,
                    product_mentions=product_mentions,
                    category_scores=category_scores,
                    processing_time=random.uniform(2.0, 8.0),
                    audio_features={
                        'has_audio': random.random() > 0.1,  # 90% have audio
                        'audio_length': duration,
                        'speech_detected': random.random() > 0.3,
                        'music_detected': random.random() > 0.4
                    },
                    text_features={
                        'total_text_regions': len(extracted_text),
                        'commerce_score': random.uniform(0.3, 0.9),
                        'price_mentions': [f"${random.randint(10, 500)}" for _ in range(random.randint(0, 3))]
                    }
                )
                
                content_list.append(content)
            
            logger.info(f"Generated {len(content_list)} content items")
            return content_list
            
        except Exception as e:
            logger.error(f"Error generating sample content: {e}")
            return []
    
    def _generate_detected_objects(self, category: str) -> List[str]:
        """Generate realistic detected objects for a content category."""
        category_objects = {
            'electronics': ['phone', 'laptop', 'headphones', 'camera', 'screen', 'cable'],
            'fashion': ['clothing', 'shoes', 'bag', 'jewelry', 'watch', 'sunglasses'],
            'home': ['furniture', 'pillow', 'lamp', 'plant', 'book', 'decoration'],
            'beauty': ['cosmetics', 'mirror', 'brush', 'bottle', 'perfume', 'skincare'],
            'sports': ['equipment', 'shoes', 'clothing', 'water bottle', 'fitness gear'],
            'books': ['book', 'bookshelf', 'reading glasses', 'notebook', 'pen'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'blocks', 'ball']
        }
        
        base_objects = ['person', 'hand', 'table', 'background']
        category_specific = category_objects.get(category, [])
        
        # Select 3-8 objects
        num_objects = random.randint(3, 8)
        selected_objects = random.sample(
            base_objects + category_specific, 
            min(num_objects, len(base_objects + category_specific))
        )
        
        return selected_objects
    
    def _generate_extracted_text(self, content_type: str, category: str) -> List[str]:
        """Generate realistic extracted text for content."""
        text_templates = {
            'product_review': [
                f"This {category} product is amazing",
                f"Highly recommend this {category}",
                f"Perfect for everyday use",
                f"Great value for money",
                f"5 stars!"
            ],
            'unboxing': [
                f"Unboxing new {category}",
                f"First impressions",
                f"What's in the box",
                f"Let's take a look",
                f"Subscribe for more"
            ],
            'tutorial': [
                f"How to use {category}",
                f"Step by step guide",
                f"Pro tips",
                f"Easy tutorial",
                f"Learn with me"
            ],
            'haul': [
                f"Shopping haul",
                f"Recent purchases",
                f"Got this {category}",
                f"Mini haul",
                f"New arrivals"
            ],
            'comparison': [
                f"{category.title()} comparison",
                f"Which one to buy?",
                f"Pros and cons",
                f"Honest review",
                f"Final verdict"
            ]
        }
        
        templates = text_templates.get(content_type, [f"About {category}", "Check this out"])
        return random.sample(templates, random.randint(2, len(templates)))

async def initialize_sample_data(
    feature_store: FeatureStore, 
    vector_search: VectorSearchEngine,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Initialize the system with comprehensive sample data."""
    try:
        logger.info("Initializing sample data for video commerce system")
        
        # Default configuration
        default_config = {
            'num_users': 1000,
            'num_products': 2000,
            'num_interactions': 10000,
            'num_content': 500
        }
        
        if config:
            default_config.update(config)
        
        generator = SampleDataGenerator()
        
        # Generate data
        logger.info("Generating users...")
        users = generator.generate_sample_users(default_config['num_users'])
        
        logger.info("Generating products...")
        products = generator.generate_sample_products(default_config['num_products'])
        
        logger.info("Generating interactions...")
        interactions = generator.generate_sample_interactions(
            users, products, default_config['num_interactions']
        )
        
        logger.info("Generating content...")
        content_items = generator.generate_sample_content(default_config['num_content'])
        
        # Store data in feature store
        logger.info("Storing sample data in feature store...")
        
        # Store users
        for user in users:
            await feature_store._set_user_features(user.user_id, user)
        
        # Store interactions
        for interaction in interactions:
            await feature_store.log_user_interaction(
                interaction['user_id'],
                interaction['product_id'], 
                interaction['action'],
                interaction.get('context', {})
            )
        
        # Store content features
        for content in content_items:
            await feature_store.store_content_features(content.content_id, content)
        
        # Initialize vector search with product embeddings
        logger.info("Initializing vector search with product embeddings...")
        
        # Generate embeddings for products and add to vector search
        for product in products[:1000]:  # Limit to first 1000 for performance
            # Generate embedding for product (simulate content-product alignment)
            category_embedding = np.random.normal(0, 0.1, 512)
            
            # Add category-specific patterns
            category_hash = hash(product.category) % 512
            category_embedding[category_hash:category_hash+20] += 0.3
            
            # Normalize
            category_embedding = category_embedding / np.linalg.norm(category_embedding)
            
            # Add to vector search
            metadata = {
                'title': product.title,
                'category': product.category,
                'price': product.price,
                'rating': product.rating,
                'brand': product.brand
            }
            
            await vector_search.add_product_embedding(
                product.product_id,
                category_embedding,
                metadata
            )
        
        # Save vector index
        await vector_search.save_index()
        
        # Generate summary statistics
        summary = {
            'users_created': len(users),
            'products_created': len(products),
            'interactions_created': len(interactions),
            'content_items_created': len(content_items),
            'categories': list(generator.product_categories.keys()),
            'user_personas': list(generator.user_personas.keys()),
            'content_types': list(generator.content_types.keys()),
            'initialization_time': time.time()
        }
        
        logger.info("Sample data initialization completed successfully")
        logger.info(f"Summary: {summary}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        raise

def save_sample_data_to_files(
    users: List[UserFeatures],
    products: List[ProductData], 
    interactions: List[Dict[str, Any]],
    output_dir: str = "sample_data"
) -> None:
    """Save generated sample data to JSON files for later use."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save users
        with open(output_path / "users.json", "w") as f:
            json.dump([user.dict() for user in users], f, indent=2, default=str)
        
        # Save products  
        with open(output_path / "products.json", "w") as f:
            json.dump([product.dict() for product in products], f, indent=2, default=str)
        
        # Save interactions
        with open(output_path / "interactions.json", "w") as f:
            json.dump(interactions, f, indent=2, default=str)
        
        logger.info(f"Sample data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving sample data: {e}")

def load_sample_data_from_files(data_dir: str = "sample_data") -> Dict[str, Any]:
    """Load sample data from JSON files."""
    try:
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Sample data directory {data_path} does not exist")
            return {}
        
        data = {}
        
        # Load users
        users_file = data_path / "users.json"
        if users_file.exists():
            with open(users_file, "r") as f:
                users_data = json.load(f)
                data['users'] = [UserFeatures(**user) for user in users_data]
        
        # Load products
        products_file = data_path / "products.json"
        if products_file.exists():
            with open(products_file, "r") as f:
                products_data = json.load(f)
                data['products'] = [ProductData(**product) for product in products_data]
        
        # Load interactions
        interactions_file = data_path / "interactions.json"
        if interactions_file.exists():
            with open(interactions_file, "r") as f:
                data['interactions'] = json.load(f)
        
        logger.info(f"Sample data loaded from {data_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return {}

class DataValidator:
    """Validates generated sample data for consistency and realism."""
    
    @staticmethod
    def validate_users(users: List[UserFeatures]) -> Dict[str, Any]:
        """Validate user data for consistency."""
        validation_results = {
            'total_users': len(users),
            'valid_users': 0,
            'issues': []
        }
        
        for user in users:
            issues = []
            
            # Check price sensitivity range
            if not (0 <= user.price_sensitivity <= 1):
                issues.append(f"Invalid price sensitivity: {user.price_sensitivity}")
            
            # Check CTR and CVR ranges
            if not (0 <= user.click_through_rate <= 1):
                issues.append(f"Invalid CTR: {user.click_through_rate}")
            
            if not (0 <= user.conversion_rate <= 1):
                issues.append(f"Invalid CVR: {user.conversion_rate}")
            
            # Check if CVR is reasonable compared to CTR
            if user.conversion_rate > user.click_through_rate:
                issues.append("CVR higher than CTR (impossible)")
            
            # Check session length
            if user.avg_session_length < 0:
                issues.append(f"Negative session length: {user.avg_session_length}")
            
            # Check last active time
            if user.last_active > time.time():
                issues.append("Last active time in the future")
            
            if not issues:
                validation_results['valid_users'] += 1
            else:
                validation_results['issues'].extend([f"{user.user_id}: {issue}" for issue in issues])
        
        validation_results['valid_percentage'] = (validation_results['valid_users'] / len(users)) * 100
        return validation_results
    
    @staticmethod
    def validate_products(products: List[ProductData]) -> Dict[str, Any]:
        """Validate product data for consistency."""
        validation_results = {
            'total_products': len(products),
            'valid_products': 0,
            'issues': [],
            'price_distribution': {},
            'category_distribution': {}
        }
        
        prices = []
        categories = {}
        
        for product in products:
            issues = []
            
            # Check price
            if product.price <= 0:
                issues.append(f"Invalid price: {product.price}")
            else:
                prices.append(product.price)
            
            # Check rating
            if product.rating and not (1.0 <= product.rating <= 5.0):
                issues.append(f"Invalid rating: {product.rating}")
            
            # Check review count
            if product.num_reviews and product.num_reviews < 0:
                issues.append(f"Negative review count: {product.num_reviews}")
            
            # Check timestamps
            if product.created_at > time.time():
                issues.append("Created time in the future")
            
            if product.updated_at < product.created_at:
                issues.append("Updated time before created time")
            
            # Count categories
            if product.category not in categories:
                categories[product.category] = 0
            categories[product.category] += 1
            
            if not issues:
                validation_results['valid_products'] += 1
            else:
                validation_results['issues'].extend([f"{product.product_id}: {issue}" for issue in issues])
        
        # Calculate distributions
        if prices:
            validation_results['price_distribution'] = {
                'min': min(prices),
                'max': max(prices),
                'mean': sum(prices) / len(prices),
                'median': sorted(prices)[len(prices) // 2]
            }
        
        validation_results['category_distribution'] = categories
        validation_results['valid_percentage'] = (validation_results['valid_products'] / len(products)) * 100
        
        return validation_results
    
    @staticmethod
    def validate_interactions(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate interaction data for consistency."""
        validation_results = {
            'total_interactions': len(interactions),
            'valid_interactions': 0,
            'issues': [],
            'action_distribution': {},
            'temporal_distribution': {}
        }
        
        actions = {}
        timestamps = []
        
        for i, interaction in enumerate(interactions):
            issues = []
            
            # Check required fields
            required_fields = ['user_id', 'product_id', 'action', 'timestamp']
            for field in required_fields:
                if field not in interaction:
                    issues.append(f"Missing field: {field}")
            
            # Check action type
            action = interaction.get('action')
            if action:
                if action not in [e.value for e in InteractionType]:
                    issues.append(f"Invalid action type: {action}")
                else:
                    if action not in actions:
                        actions[action] = 0
                    actions[action] += 1
            
            # Check timestamp
            timestamp = interaction.get('timestamp')
            if timestamp:
                if timestamp > time.time():
                    issues.append("Timestamp in the future")
                else:
                    timestamps.append(timestamp)
            
            # Check purchase value
            if action == InteractionType.PURCHASE.value:
                if 'value' not in interaction or interaction['value'] <= 0:
                    issues.append("Purchase interaction missing or invalid value")
            
            if not issues:
                validation_results['valid_interactions'] += 1
            else:
                validation_results['issues'].extend([f"Interaction {i}: {issue}" for issue in issues])
        
        # Calculate distributions
        validation_results['action_distribution'] = actions
        
        if timestamps:
            timestamps.sort()
            validation_results['temporal_distribution'] = {
                'earliest': timestamps[0],
                'latest': timestamps[-1],
                'span_days': (timestamps[-1] - timestamps[0]) / (24 * 3600)
            }
        
        validation_results['valid_percentage'] = (validation_results['valid_interactions'] / len(interactions)) * 100
        
        return validation_results

def generate_and_validate_sample_data(
    num_users: int = 1000,
    num_products: int = 2000, 
    num_interactions: int = 10000,
    save_to_files: bool = True,
    output_dir: str = "sample_data"
) -> Dict[str, Any]:
    """Generate sample data and validate it for quality."""
    try:
        logger.info("Generating and validating sample data")
        
        generator = SampleDataGenerator()
        validator = DataValidator()
        
        # Generate data
        users = generator.generate_sample_users(num_users)
        products = generator.generate_sample_products(num_products)
        interactions = generator.generate_sample_interactions(users, products, num_interactions)
        
        # Validate data
        user_validation = validator.validate_users(users)
        product_validation = validator.validate_products(products)
        interaction_validation = validator.validate_interactions(interactions)
        
        # Save to files if requested
        if save_to_files:
            save_sample_data_to_files(users, products, interactions, output_dir)
        
        # Compile results
        results = {
            'generation_summary': {
                'users_generated': len(users),
                'products_generated': len(products),
                'interactions_generated': len(interactions),
                'generation_timestamp': time.time()
            },
            'validation_results': {
                'users': user_validation,
                'products': product_validation,
                'interactions': interaction_validation
            },
            'data_saved': save_to_files,
            'output_directory': output_dir if save_to_files else None
        }
        
        # Log summary
        logger.info(f"Data generation complete:")
        logger.info(f"  Users: {len(users)} ({user_validation['valid_percentage']:.1f}% valid)")
        logger.info(f"  Products: {len(products)} ({product_validation['valid_percentage']:.1f}% valid)")
        logger.info(f"  Interactions: {len(interactions)} ({interaction_validation['valid_percentage']:.1f}% valid)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in data generation and validation: {e}")
        return {'error': str(e)}

# Convenience functions for testing
def quick_sample_data(scale: str = "small") -> Dict[str, Any]:
    """Generate quick sample data for testing purposes."""
    scale_configs = {
        'small': {'num_users': 100, 'num_products': 200, 'num_interactions': 1000},
        'medium': {'num_users': 500, 'num_products': 1000, 'num_interactions': 5000},
        'large': {'num_users': 1000, 'num_products': 2000, 'num_interactions': 10000},
        'xlarge': {'num_users': 2000, 'num_products': 5000, 'num_interactions': 25000}
    }
    
    config = scale_configs.get(scale, scale_configs['small'])
    return generate_and_validate_sample_data(**config, save_to_files=False)

# ============================================================================
# CSV Dataset Loading Functions
# ============================================================================

def decode_base64_embedding(encoded_str: str, dim: int = 128) -> np.ndarray:
    """Decode base64 encoded embedding string to numpy array."""
    try:
        decoded_bytes = base64.b64decode(encoded_str)
        embedding = np.frombuffer(decoded_bytes, dtype=np.float32)
        if len(embedding) != dim:
            logger.warning(f"Embedding dimension mismatch: expected {dim}, got {len(embedding)}")
            # Pad or truncate if needed
            if len(embedding) < dim:
                embedding = np.pad(embedding, (0, dim - len(embedding)), 'constant')
            else:
                embedding = embedding[:dim]
        return embedding
    except Exception as e:
        logger.error(f"Error decoding embedding: {e}")
        # Return zero vector as fallback
        return np.zeros(dim, dtype=np.float32)

def load_users_from_csv(csv_path: str, limit: Optional[int] = None) -> List[UserFeatures]:
    """Load users from CSV file and convert to UserFeatures."""
    try:
        logger.info(f"Loading users from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
        
        users = []
        for _, row in df.iterrows():
            # Calculate user features from interactions (will be updated later)
            user = UserFeatures(
                user_id=row['user_id'],
                total_interactions=0,  # Will be updated from interactions CSV
                avg_session_length=0.0,
                preferred_categories=[],  # Will be inferred from interactions
                price_sensitivity=0.5,  # Default value
                click_through_rate=0.0,
                conversion_rate=0.0,
                last_active=time.time(),
                demographics={
                    'country': row.get('country', 'US'),
                    'platform': row.get('platform', 'unknown'),
                    'preferred_language': row.get('preferred_language', 'en-US'),
                    'marketing_source': row.get('marketing_source', 'unknown'),
                    'membership_level': row.get('membership_level', 'Silver'),
                    'signup_date': row.get('signup_date', '')
                }
            )
            users.append(user)
        
        logger.info(f"Loaded {len(users)} users from CSV")
        return users
        
    except Exception as e:
        logger.error(f"Error loading users from CSV: {e}")
        return []

def load_products_from_csv(csv_path: str, limit: Optional[int] = None) -> List[ProductData]:
    """Load products from CSV file and convert to ProductData."""
    try:
        logger.info(f"Loading products from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
        
        products = []
        for _, row in df.iterrows():
            # Parse availability
            availability = str(row.get('availability', 'in_stock')).lower()
            in_stock = availability in ['in_stock', 'in stock', 'available']
            
            # Create description from title and category if description is missing
            description = f"{row.get('title', '')} - {row.get('product_category', '')}"
            
            product = ProductData(
                product_id=row['product_id'],
                title=row.get('title', 'Unknown Product'),
                description=description,
                price=float(row.get('price_value', 0.0)),
                currency=row.get('currency', 'USD'),
                category=row.get('product_category', 'Unknown'),
                brand=row.get('brand', 'Unknown'),
                image_url=None,  # CSV doesn't have image URLs
                rating=float(row.get('product_rating', 0.0)) if pd.notna(row.get('product_rating')) else None,
                num_reviews=0,  # CSV doesn't have review count
                in_stock=in_stock,
                tags=[row.get('product_category', ''), row.get('brand', '')],
                created_at=time.time(),
                updated_at=time.time()
            )
            products.append(product)
        
        logger.info(f"Loaded {len(products)} products from CSV")
        return products
        
    except Exception as e:
        logger.error(f"Error loading products from CSV: {e}")
        return []

def load_content_features_from_csv(
    csv_path: str, 
    embeddings_npy_path: Optional[str] = None,
    limit: Optional[int] = None
) -> List[ContentFeatures]:
    """
    Load content features from CSV file and convert to ContentFeatures.
    
    Args:
        csv_path: Path to multimodal_features.csv
        embeddings_npy_path: Optional path to video_embeddings_128d.npy file
                           If provided, embeddings will be loaded from .npy file instead of CSV
        limit: Optional limit on number of records to load
    """
    try:
        logger.info(f"Loading content features from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
        
        # Load embeddings from .npy file if provided
        embeddings_array = None
        if embeddings_npy_path and Path(embeddings_npy_path).exists():
            logger.info(f"Loading visual embeddings from {embeddings_npy_path}")
            embeddings_array = np.load(embeddings_npy_path)
            logger.info(f"Loaded embeddings array with shape: {embeddings_array.shape}")
            # Limit embeddings to match CSV limit if specified
            if limit and embeddings_array.shape[0] > limit:
                embeddings_array = embeddings_array[:limit]
        
        content_items = []
        for idx, row in df.iterrows():
            video_id = row['video_id']
            
            # Load visual embedding from .npy file or CSV
            if embeddings_array is not None:
                # Use embedding from .npy file (index should match CSV row index)
                if idx < len(embeddings_array):
                    visual_embedding_128 = embeddings_array[idx].astype(np.float32)
                else:
                    logger.warning(f"Index {idx} out of range for embeddings array, using zero vector")
                    visual_embedding_128 = np.zeros(128, dtype=np.float32)
            else:
                # Fallback: try to decode from CSV base64 field
                visual_embedding_encoded = row.get('visual_embedding_128d', '')
                if pd.notna(visual_embedding_encoded) and visual_embedding_encoded:
                    visual_embedding_128 = decode_base64_embedding(visual_embedding_encoded, dim=128)
                else:
                    visual_embedding_128 = np.zeros(128, dtype=np.float32)
            
            # Convert to 512-dim if needed (pad with zeros)
            if len(visual_embedding_128) == 128:
                # Pad to 512 dimensions for compatibility
                visual_embedding_512 = np.pad(visual_embedding_128, (0, 512 - 128), 'constant')
            else:
                visual_embedding_512 = visual_embedding_128[:512] if len(visual_embedding_128) >= 512 else np.pad(visual_embedding_128, (0, 512 - len(visual_embedding_128)), 'constant')
            
            # Parse OCR text
            ocr_text = str(row.get('ocr_text', '')) if pd.notna(row.get('ocr_text')) else ''
            extracted_text = [t.strip() for t in ocr_text.split(',') if t.strip()] if ocr_text else []
            
            # Parse scene labels
            scene_labels = str(row.get('scene_labels', '')) if pd.notna(row.get('scene_labels')) else ''
            detected_objects = [s.strip() for s in scene_labels.split(',') if s.strip()] if scene_labels else []
            
            # Audio features
            audio_transcript = str(row.get('audio_transcript', '')) if pd.notna(row.get('audio_transcript')) else ''
            audio_sentiment = float(row.get('audio_sentiment', 0.0)) if pd.notna(row.get('audio_sentiment')) else 0.0
            
            # Extract product mentions from transcript and OCR
            product_mentions = []
            if audio_transcript:
                product_mentions.append(audio_transcript[:100])  # First 100 chars
            if extracted_text:
                product_mentions.extend(extracted_text[:3])  # First 3 OCR texts
            
            content = ContentFeatures(
                content_id=video_id,
                visual_embedding=visual_embedding_512.tolist(),
                duration_seconds=None,  # Not in CSV
                detected_objects=detected_objects,
                extracted_text=extracted_text,
                product_mentions=product_mentions,
                category_scores={},  # Will be inferred from content
                processing_time=None,
                audio_features={
                    'has_audio': bool(audio_transcript),
                    'audio_transcript': audio_transcript,
                    'audio_sentiment': audio_sentiment
                },
                text_features={
                    'total_text_regions': len(extracted_text),
                    'ocr_text': ocr_text,
                    'quality_score': float(row.get('quality_score', 0.5)) if pd.notna(row.get('quality_score')) else 0.5
                }
            )
            content_items.append(content)
        
        logger.info(f"Loaded {len(content_items)} content items from CSV")
        return content_items
        
    except Exception as e:
        logger.error(f"Error loading content features from CSV: {e}")
        return []

def load_interactions_from_csv(
    csv_path: str, 
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load interactions from CSV file."""
    try:
        logger.info(f"Loading interactions from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
        
        interactions = []
        for _, row in df.iterrows():
            # Determine interaction type based on boolean columns
            action = InteractionType.VIEW.value  # Default
            
            if row.get('purchased', False):
                action = InteractionType.PURCHASE.value
            elif row.get('added_to_cart', False):
                action = InteractionType.ADD_TO_CART.value
            elif row.get('liked', False):
                action = InteractionType.FAVORITE.value
            elif row.get('shared', False):
                action = InteractionType.SHARE.value
            elif row.get('commented', False) or row.get('watch_percentage', 0) > 0.5:
                action = InteractionType.CLICK.value
            
            # Parse timestamp
            timestamp_str = row.get('timestamp', '')
            try:
                if pd.notna(timestamp_str):
                    # Try parsing ISO format
                    dt = pd.to_datetime(timestamp_str)
                    timestamp = dt.timestamp()
                else:
                    timestamp = time.time()
            except:
                timestamp = time.time()
            
            interaction = {
                'user_id': row['user_id'],
                'product_id': row['product_id'],
                'action': action,
                'timestamp': timestamp,
                'context': {
                    'video_id': row.get('video_id', ''),
                    'session_id': row.get('session_id', ''),
                    'device_type': row.get('device_type', 'unknown'),
                    'platform': row.get('platform', 'unknown'),
                    'watch_time_seconds': float(row.get('watch_time_seconds', 0)) if pd.notna(row.get('watch_time_seconds')) else 0,
                    'watch_percentage': float(row.get('watch_percentage', 0)) if pd.notna(row.get('watch_percentage')) else 0,
                    'video_category': row.get('video_category', ''),
                    'product_category': row.get('product_category', ''),
                    'product_price': float(row.get('product_price', 0)) if pd.notna(row.get('product_price')) else 0,
                    'purchase_amount': float(row.get('purchase_amount', 0)) if pd.notna(row.get('purchase_amount')) else 0
                }
            }
            
            # Add purchase value if it's a purchase
            if action == InteractionType.PURCHASE.value:
                interaction['value'] = float(row.get('purchase_amount', 0)) if pd.notna(row.get('purchase_amount')) else 0
            
            interactions.append(interaction)
        
        logger.info(f"Loaded {len(interactions)} interactions from CSV")
        return interactions
        
    except Exception as e:
        logger.error(f"Error loading interactions from CSV: {e}")
        return []

async def load_dataset_from_csv(
    dataset_dir: str = "Dataset",
    feature_store: Optional[FeatureStore] = None,
    vector_search: Optional[VectorSearchEngine] = None,
    limit_users: Optional[int] = None,
    limit_products: Optional[int] = None,
    limit_interactions: Optional[int] = None,
    limit_content: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load dataset from CSV files in the Dataset directory.
    
    Args:
        dataset_dir: Directory containing CSV files
        feature_store: FeatureStore instance to load data into
        vector_search: VectorSearchEngine instance to load embeddings into
        limit_*: Optional limits on number of records to load
    
    Returns:
        Dictionary with loading statistics
    """
    try:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        logger.info(f"Loading dataset from {dataset_dir}")
        
        # Load data from CSV files
        users_csv = dataset_path / "users.csv"
        products_csv = dataset_path / "products.csv"
        interactions_csv = dataset_path / "video_commerce_interactions_500k.csv"
        features_csv = dataset_path / "multimodal_features.csv"
        
        # Load users
        users = []
        if users_csv.exists():
            users = load_users_from_csv(str(users_csv), limit=limit_users)
        else:
            logger.warning(f"Users CSV not found: {users_csv}")
        
        # Load products
        products = []
        if products_csv.exists():
            products = load_products_from_csv(str(products_csv), limit=limit_products)
        else:
            logger.warning(f"Products CSV not found: {products_csv}")
        
        # Load interactions
        interactions = []
        if interactions_csv.exists():
            interactions = load_interactions_from_csv(str(interactions_csv), limit=limit_interactions)
        else:
            logger.warning(f"Interactions CSV not found: {interactions_csv}")
        
        # Load content features
        content_items = []
        if features_csv.exists():
            # Check for embeddings .npy file
            embeddings_npy = dataset_path / "video_embeddings_128d.npy"
            # Also check in parent directory (project root)
            if not embeddings_npy.exists():
                embeddings_npy = Path("video_embeddings_128d.npy")
            
            embeddings_path = str(embeddings_npy) if embeddings_npy.exists() else None
            if embeddings_path:
                logger.info(f"Found embeddings file: {embeddings_path}")
            else:
                logger.warning(f"Embeddings .npy file not found, will try to use CSV base64 field")
            
            content_items = load_content_features_from_csv(
                str(features_csv), 
                embeddings_npy_path=embeddings_path,
                limit=limit_content
            )
        else:
            logger.warning(f"Features CSV not found: {features_csv}")
        
        # Update user features based on interactions
        logger.info("Updating user features from interactions...")
        user_interaction_counts = {}
        user_categories = {}
        user_last_active = {}
        
        for interaction in interactions:
            user_id = interaction['user_id']
            if user_id not in user_interaction_counts:
                user_interaction_counts[user_id] = 0
                user_categories[user_id] = set()
                user_last_active[user_id] = interaction['timestamp']
            
            user_interaction_counts[user_id] += 1
            user_last_active[user_id] = max(user_last_active[user_id], interaction['timestamp'])
            
            # Extract category from context
            category = interaction.get('context', {}).get('product_category', '')
            if category:
                user_categories[user_id].add(category)
        
        # Update user objects
        user_dict = {u.user_id: u for u in users}
        for user_id, count in user_interaction_counts.items():
            if user_id in user_dict:
                user = user_dict[user_id]
                user.total_interactions = count
                user.preferred_categories = list(user_categories.get(user_id, set()))
                user.last_active = user_last_active.get(user_id, time.time())
        
        # Load data into feature store if provided
        if feature_store:
            logger.info("Loading data into feature store...")
            
            # Store users
            for user in users:
                await feature_store._set_user_features(user.user_id, user)
            
            # Store interactions
            for interaction in interactions:
                await feature_store.log_user_interaction(
                    interaction['user_id'],
                    interaction['product_id'],
                    interaction['action'],
                    interaction.get('context', {})
                )
            
            # Store content features
            for content in content_items:
                await feature_store.store_content_features(content.content_id, content)
        
        # Load product embeddings into vector search if provided
        if vector_search and products:
            logger.info("Loading product embeddings into vector search...")
            
            # Create embeddings for products (use category-based embeddings)
            for product in products:
                # Generate category-based embedding
                category_embedding = np.random.normal(0, 0.1, vector_search.embedding_dim)
                
                # Add category-specific patterns
                category_hash = hash(product.category) % vector_search.embedding_dim
                category_embedding[category_hash:category_hash+20] += 0.3
                
                # Normalize
                category_embedding = category_embedding / np.linalg.norm(category_embedding)
                
                # Add to vector search
                metadata = {
                    'title': product.title,
                    'category': product.category,
                    'price': product.price,
                    'rating': product.rating,
                    'brand': product.brand
                }
                
                await vector_search.add_product_embedding(
                    product.product_id,
                    category_embedding.astype(np.float32),
                    metadata
                )
            
            # Save vector index
            await vector_search.save_index()
            logger.info("Vector search index saved")
        
        # Generate summary
        summary = {
            'users_loaded': len(users),
            'products_loaded': len(products),
            'interactions_loaded': len(interactions),
            'content_items_loaded': len(content_items),
            'loaded_into_feature_store': feature_store is not None,
            'loaded_into_vector_search': vector_search is not None,
            'loading_timestamp': time.time()
        }
        
        logger.info("Dataset loading completed successfully")
        logger.info(f"Summary: {summary}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error loading dataset from CSV: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for video commerce recommender")
    parser.add_argument("--users", type=int, default=1000, help="Number of users to generate")
    parser.add_argument("--products", type=int, default=2000, help="Number of products to generate")
    parser.add_argument("--interactions", type=int, default=10000, help="Number of interactions to generate")
    parser.add_argument("--output-dir", default="sample_data", help="Output directory for data files")
    parser.add_argument("--no-save", action="store_true", help="Don't save data to files")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate data
    results = generate_and_validate_sample_data(
        num_users=args.users,
        num_products=args.products,
        num_interactions=args.interactions,
        save_to_files=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Print summary
    print(json.dumps(results, indent=2, default=str))