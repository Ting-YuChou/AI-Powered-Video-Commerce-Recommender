"""
AI-Powered Video Commerce Recommender - Content Processor
=========================================================

This module handles multi-modal video content processing using CLIP for visual understanding,
OCR for text extraction, and basic audio processing. It extracts rich features from video
content that can be used for product recommendations.
"""

import cv2
import numpy as np
import torch
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageEnhance
import io
import base64
import time
from pathlib import Path
import tempfile
import os

# ML/AI imports
from transformers import CLIPModel, CLIPProcessor
import pytesseract
from pytesseract import Output

# Local imports
from models import ContentFeatures
from config import ModelConfig

logger = logging.getLogger(__name__)

class ContentProcessor:
    """
    Multi-modal content processor for video commerce recommendations.
    
    Processes video content to extract:
    - Visual features using CLIP
    - Text features using OCR
    - Basic audio features
    - Object detection and scene analysis
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the content processor with configuration."""
        self.config = config
        self.clip_model = None
        self.clip_processor = None
        self.device = torch.device(self._get_device())
        self.is_initialized = False
        self.lazy_load = os.getenv("MODEL_LAZY_LOAD", "false").lower() == "true"
        
        # Processing parameters
        self.max_keyframes = config.num_keyframes
        self.max_video_length = config.max_video_length
        self.batch_size = config.batch_size
        
        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6'
        
        # Commerce-related keywords for filtering
        self.commerce_keywords = {
            'price_indicators': ['$', '€', '£', '¥', 'price', 'cost', 'buy', 'sale', 'discount', '%', 'off'],
            'product_keywords': ['product', 'item', 'model', 'brand', 'new', 'latest', 'review'],
            'action_words': ['buy', 'purchase', 'order', 'shop', 'get', 'available', 'stock']
        }
        
        logger.info(f"ContentProcessor initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device for processing."""
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.enable_gpu:
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    async def load_models(self):
        """Load all required ML models."""
        try:
            logger.info(f"Loading CLIP model: {self.config.clip_model}")
            
            # Load CLIP model and processor
            self.clip_model = CLIPModel.from_pretrained(
                self.config.clip_model,
                cache_dir=self.config.cache_dir
            ).to(self.device)
            
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.config.clip_model,
                cache_dir=self.config.cache_dir
            )
            
            # Set to evaluation mode
            self.clip_model.eval()
            
            # Enable quantization if configured
            if self.config.enable_quantization and self.device == "cpu":
                self.clip_model = torch.quantization.quantize_dynamic(
                    self.clip_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Model quantization enabled")
            
            self.is_initialized = True
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    async def process_video(self, video_path: str, content_id: Optional[str] = None) -> ContentFeatures:
        """
        Process a video file and extract multi-modal features.
        
        Args:
            video_path: Path to the video file
            content_id: Optional content identifier
            
        Returns:
            ContentFeatures object with extracted features
        """
        if not self.is_initialized:
            await self.load_models()
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Generate content ID if not provided
            if content_id is None:
                content_id = f"content_{int(time.time())}"
            
            # Extract keyframes from video
            keyframes, video_info = await self._extract_keyframes(video_path)
            
            if not keyframes:
                raise ValueError("No keyframes could be extracted from video")
            
            # Process visual features with CLIP
            visual_features = await self._extract_visual_features(keyframes)
            
            # Extract text using OCR
            text_features = await self._extract_text_features(keyframes)
            
            # Basic object detection using CLIP
            detected_objects = await self._detect_objects(keyframes)
            
            # Analyze commerce-related content
            commerce_features = self._analyze_commerce_content(text_features, detected_objects)
            
            processing_time = time.time() - start_time
            
            # Create ContentFeatures object
            content_features = ContentFeatures(
                content_id=content_id,
                visual_embedding=visual_features['embedding'].tolist(),
                duration_seconds=video_info.get('duration', 0),
                detected_objects=detected_objects,
                extracted_text=text_features.get('text_blocks', []),
                product_mentions=commerce_features.get('product_mentions', []),
                category_scores=commerce_features.get('category_scores', {}),
                processing_time=processing_time,
                audio_features={
                    'has_audio': video_info.get('has_audio', False),
                    'audio_length': video_info.get('duration', 0)
                },
                text_features={
                    'total_text_regions': len(text_features.get('text_blocks', [])),
                    'commerce_score': commerce_features.get('commerce_score', 0.0),
                    'price_mentions': text_features.get('price_mentions', [])
                }
            )
            
            logger.info(f"Video processing completed in {processing_time:.2f}s for {content_id}")
            return content_features
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise
    
    async def _extract_keyframes(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """Extract keyframes from video file."""
        if self.config.ffmpeg_frame_extraction_enabled:
            try:
                keyframes, video_info = await self._extract_keyframes_ffmpeg(video_path)
                if keyframes:
                    logger.info(
                        "Extracted %s keyframes from video using FFmpeg",
                        len(keyframes),
                    )
                    return keyframes, video_info
                logger.warning(
                    "FFmpeg keyframe extraction returned no frames; falling back to OpenCV"
                )
            except Exception as exc:
                logger.warning(
                    "FFmpeg keyframe extraction failed; falling back to OpenCV: %s",
                    exc,
                )

        return await self._extract_keyframes_opencv(video_path)

    async def _extract_keyframes_ffmpeg(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """Extract uniformly sampled RGB keyframes using ffprobe and ffmpeg."""
        video_info = await self._probe_video_metadata(video_path)
        frame_count = int(video_info.get("frame_count") or 0)
        fps = float(video_info.get("fps") or 0)
        duration = float(video_info.get("duration") or 0)

        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)

        if duration > self.max_video_length and fps > 0:
            logger.warning(
                "Video duration %ss exceeds limit %ss",
                duration,
                self.max_video_length,
            )
            frame_count = min(frame_count, int(self.max_video_length * fps))
            video_info["frame_count"] = frame_count

        frame_indices = self._keyframe_indices(frame_count)
        if not frame_indices:
            return [], video_info

        select_filter = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
        target_width = max(1, int(self.config.ffmpeg_target_width))
        filter_graph = (
            f"select={select_filter},"
            f"scale=min({target_width}\\,iw):-2,"
            "format=rgb24"
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            video_path,
            "-vf",
            filter_graph,
            "-vsync",
            "vfr",
            "-frames:v",
            str(len(frame_indices)),
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "pipe:1",
        ]
        output = await self._run_media_command(command)
        keyframes = self._parse_ppm_frames(output)
        if len(keyframes) != len(frame_indices):
            raise ValueError(
                f"FFmpeg extracted {len(keyframes)} of {len(frame_indices)} requested frames"
            )
        return keyframes, video_info

    async def _probe_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Read video metadata using ffprobe."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            video_path,
        ]
        output = await self._run_media_command(command)
        payload = json.loads(output.decode("utf-8"))
        streams = payload.get("streams") or []
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise ValueError("ffprobe did not report a video stream")

        format_info = payload.get("format") or {}
        fps = self._parse_frame_rate(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        )
        duration = self._parse_float(
            video_stream.get("duration"),
            default=self._parse_float(format_info.get("duration"), default=0.0),
        )
        frame_count = self._parse_int(video_stream.get("nb_frames"), default=0)
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
            "width": self._parse_int(video_stream.get("width"), default=0),
            "height": self._parse_int(video_stream.get("height"), default=0),
        }

    async def _run_media_command(self, command: List[str]) -> bytes:
        """Run an FFmpeg/ffprobe command and return stdout."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.ffmpeg_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise TimeoutError(f"{command[0]} timed out") from exc

        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{command[0]} failed: {message}")
        return stdout

    def _parse_ppm_frames(self, payload: bytes) -> List[np.ndarray]:
        """Parse concatenated binary PPM frames from FFmpeg image2pipe output."""
        frames: List[np.ndarray] = []
        index = 0
        length = len(payload)

        while index < length:
            index = self._skip_ppm_whitespace_and_comments(payload, index)
            if index >= length:
                break
            if payload[index : index + 2] != b"P6":
                raise ValueError("Unexpected FFmpeg frame format")
            index += 2

            width_token, index = self._read_ppm_token(payload, index)
            height_token, index = self._read_ppm_token(payload, index)
            max_value_token, index = self._read_ppm_token(payload, index)
            width = int(width_token)
            height = int(height_token)
            max_value = int(max_value_token)
            if width <= 0 or height <= 0 or max_value != 255:
                raise ValueError("Unsupported FFmpeg PPM frame")
            if index >= length or payload[index] not in b" \t\r\n":
                raise ValueError("Invalid FFmpeg PPM frame payload")
            index += 1

            frame_size = width * height * 3
            if index + frame_size > length:
                raise ValueError("Truncated FFmpeg PPM frame")
            frame = np.frombuffer(
                payload[index : index + frame_size],
                dtype=np.uint8,
            ).copy()
            frames.append(frame.reshape((height, width, 3)))
            index += frame_size

        return frames

    def _skip_ppm_whitespace_and_comments(self, payload: bytes, index: int) -> int:
        while index < len(payload):
            if payload[index] in b" \t\r\n":
                index += 1
                continue
            if payload[index] == ord("#"):
                while index < len(payload) and payload[index] not in b"\r\n":
                    index += 1
                continue
            break
        return index

    def _read_ppm_token(self, payload: bytes, index: int) -> Tuple[str, int]:
        index = self._skip_ppm_whitespace_and_comments(payload, index)
        start = index
        while index < len(payload) and payload[index] not in b" \t\r\n":
            index += 1
        if start == index:
            raise ValueError("Invalid FFmpeg PPM header")
        return payload[start:index].decode("ascii"), index

    def _keyframe_indices(self, frame_count: int) -> List[int]:
        if frame_count <= 0 or self.max_keyframes <= 0:
            return []
        sample_count = min(self.max_keyframes, frame_count)
        return np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()

    def _parse_frame_rate(self, value: Any) -> float:
        if not value:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        return self._parse_float(value, default=0.0)

    def _parse_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    async def _extract_keyframes_opencv(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """Extract keyframes from video file using OpenCV."""
        keyframes = []
        video_info = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Limit video length
            if duration > self.max_video_length:
                logger.warning(f"Video duration {duration}s exceeds limit {self.max_video_length}s")
                frame_count = int(self.max_video_length * fps)
            
            video_info = {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'has_audio': True  # Assume audio exists (basic check)
            }
            
            # Calculate frame indices for uniform sampling
            frame_indices = self._keyframe_indices(frame_count)
            
            # Extract keyframes
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append(frame_rgb)
                else:
                    logger.warning(f"Could not read frame {frame_idx}")
            
            cap.release()
            
            logger.info(f"Extracted {len(keyframes)} keyframes from video")
            return keyframes, video_info
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
            return [], {}
    
    async def _extract_visual_features(self, keyframes: List[np.ndarray]) -> Dict[str, Any]:
        """Extract visual features using CLIP."""
        try:
            # Convert numpy arrays to PIL Images
            pil_images = []
            for frame in keyframes:
                # Enhance image quality
                img = Image.fromarray(frame)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.2)  # Slight sharpness increase
                pil_images.append(img)
            
            # Process images in batches
            all_embeddings = []
            
            for i in range(0, len(pil_images), self.batch_size):
                batch_images = pil_images[i:i + self.batch_size]
                
                # Preprocess images
                inputs = self.clip_processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    all_embeddings.append(image_features.cpu())
            
            # Combine all embeddings
            if all_embeddings:
                all_embeddings = torch.cat(all_embeddings, dim=0)
                # Average pool across frames
                video_embedding = all_embeddings.mean(dim=0)
            else:
                video_embedding = torch.zeros(self.config.embedding_dim)
            
            return {
                'embedding': video_embedding,
                'frame_count': len(keyframes),
                'individual_embeddings': [emb.numpy() for emb in all_embeddings] if len(all_embeddings) <= 10 else []
            }
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {'embedding': torch.zeros(self.config.embedding_dim), 'frame_count': 0}
    
    async def _extract_text_features(self, keyframes: List[np.ndarray]) -> Dict[str, Any]:
        """Extract text using OCR from keyframes."""
        all_text_blocks = []
        price_mentions = []
        
        try:
            for i, frame in enumerate(keyframes):
                # Convert to PIL Image for OCR
                pil_image = Image.fromarray(frame)
                
                # Enhance image for better OCR
                enhancer = ImageEnhance.Contrast(pil_image)
                enhanced_image = enhancer.enhance(1.5)
                
                # Perform OCR
                try:
                    ocr_data = pytesseract.image_to_data(
                        enhanced_image, 
                        config=self.ocr_config,
                        output_type=Output.DICT
                    )
                    
                    # Extract text blocks with confidence > 30
                    frame_text = []
                    for j, confidence in enumerate(ocr_data['conf']):
                        if int(confidence) > 30:
                            text = ocr_data['text'][j].strip()
                            if text and len(text) > 1:
                                frame_text.append(text)
                                
                                # Check for price patterns
                                if self._is_price_mention(text):
                                    price_mentions.append({
                                        'text': text,
                                        'frame': i,
                                        'confidence': confidence
                                    })
                    
                    if frame_text:
                        all_text_blocks.extend(frame_text)
                        
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for frame {i}: {ocr_error}")
            
            # Remove duplicates and clean text
            unique_text_blocks = list(set([
                text for text in all_text_blocks 
                if len(text) > 2 and text.replace(' ', '').isalnum()
            ]))
            
            return {
                'text_blocks': unique_text_blocks,
                'price_mentions': price_mentions,
                'total_text_regions': len(all_text_blocks)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {'text_blocks': [], 'price_mentions': [], 'total_text_regions': 0}
    
    def _is_price_mention(self, text: str) -> bool:
        """Check if text contains price information."""
        text_lower = text.lower()
        
        # Check for currency symbols and price patterns
        price_patterns = ['$', '€', '£', '¥', 'usd', 'eur', 'gbp']
        for pattern in price_patterns:
            if pattern in text_lower:
                return True
        
        # Check for price-like numbers
        import re
        if re.search(r'\d+\.?\d*\s*(dollars?|euros?|pounds?)', text_lower):
            return True
        
        return False
    
    async def _detect_objects(self, keyframes: List[np.ndarray]) -> List[str]:
        """Detect objects in keyframes using CLIP zero-shot classification."""
        try:
            # Common product categories for e-commerce
            candidate_objects = [
                'clothing', 'shoes', 'electronics', 'phone', 'laptop', 'headphones',
                'watch', 'jewelry', 'bag', 'furniture', 'book', 'toy', 'food',
                'cosmetics', 'skincare', 'perfume', 'glasses', 'hat', 'shirt',
                'dress', 'jacket', 'pants', 'sneakers', 'boots', 'camera',
                'tablet', 'speaker', 'gaming', 'kitchen', 'home decor'
            ]
            
            detected_objects = set()
            
            # Sample a few representative frames
            sample_frames = keyframes[::max(1, len(keyframes) // 3)][:3]
            
            for frame in sample_frames:
                pil_image = Image.fromarray(frame)
                
                # Prepare text queries for zero-shot classification
                text_queries = [f"a photo of {obj}" for obj in candidate_objects]
                
                # Process with CLIP
                inputs = self.clip_processor(
                    images=[pil_image],
                    text=text_queries,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get top predictions above threshold
                top_probs, top_indices = torch.topk(probs[0], k=5)
                
                for prob, idx in zip(top_probs, top_indices):
                    if prob.item() > 0.1:  # Threshold for object detection
                        detected_objects.add(candidate_objects[idx.item()])
            
            return list(detected_objects)
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def _analyze_commerce_content(self, text_features: Dict, detected_objects: List[str]) -> Dict[str, Any]:
        """Analyze commerce-related aspects of the content."""
        try:
            text_blocks = text_features.get('text_blocks', [])
            price_mentions = text_features.get('price_mentions', [])
            
            # Calculate commerce score based on various factors
            commerce_score = 0.0
            
            # Text-based commerce indicators
            commerce_text_count = 0
            for text in text_blocks:
                text_lower = text.lower()
                for keyword_category in self.commerce_keywords.values():
                    if any(keyword in text_lower for keyword in keyword_category):
                        commerce_text_count += 1
                        break
            
            if text_blocks:
                commerce_score += (commerce_text_count / len(text_blocks)) * 0.4
            
            # Price mentions boost
            if price_mentions:
                commerce_score += min(len(price_mentions) * 0.1, 0.3)
            
            # Product objects boost
            product_objects = [obj for obj in detected_objects 
                             if obj in ['clothing', 'shoes', 'electronics', 'phone', 
                                      'laptop', 'watch', 'jewelry', 'bag']]
            if product_objects:
                commerce_score += min(len(product_objects) * 0.05, 0.3)
            
            # Normalize score
            commerce_score = min(commerce_score, 1.0)
            
            # Categorize content
            category_scores = {}
            category_mapping = {
                'electronics': ['phone', 'laptop', 'headphones', 'camera', 'tablet', 'speaker'],
                'fashion': ['clothing', 'shoes', 'jewelry', 'watch', 'bag', 'hat'],
                'home': ['furniture', 'kitchen', 'home decor'],
                'beauty': ['cosmetics', 'skincare', 'perfume'],
                'books': ['book'],
                'toys': ['toy', 'gaming']
            }
            
            for category, keywords in category_mapping.items():
                score = len([obj for obj in detected_objects if obj in keywords]) / max(len(detected_objects), 1)
                if score > 0:
                    category_scores[category] = score
            
            # Extract potential product mentions from text
            product_mentions = []
            for text in text_blocks:
                if any(keyword in text.lower() for keyword in self.commerce_keywords['product_keywords']):
                    if len(text.split()) <= 5:  # Likely product names are short
                        product_mentions.append(text)
            
            return {
                'commerce_score': commerce_score,
                'category_scores': category_scores,
                'product_mentions': product_mentions[:10],  # Limit to top 10
                'has_pricing_info': len(price_mentions) > 0,
                'commerce_text_ratio': commerce_text_count / max(len(text_blocks), 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing commerce content: {e}")
            return {
                'commerce_score': 0.0,
                'category_scores': {},
                'product_mentions': [],
                'has_pricing_info': False,
                'commerce_text_ratio': 0.0
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the content processor."""
        try:
            status = {
                'initialized': self.is_initialized,
                'lazy_load_enabled': self.lazy_load,
                'device': str(self.device),
                'clip_model_loaded': self.clip_model is not None,
                'clip_processor_loaded': self.clip_processor is not None,
            }
            
            # Test basic functionality if initialized
            if self.is_initialized:
                try:
                    # Create dummy image and test processing
                    dummy_image = Image.new('RGB', (224, 224), color='white')
                    inputs = self.clip_processor(
                        images=[dummy_image], 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.clip_model.get_image_features(**inputs)
                    
                    status['clip_test_passed'] = True
                    status['output_shape'] = list(outputs.shape)
                    
                except Exception as test_error:
                    status['clip_test_passed'] = False
                    status['test_error'] = str(test_error)
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'error': str(e), 'initialized': False}
    
    async def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image (useful for testing or single-frame processing)."""
        if not self.is_initialized:
            await self.load_models()
        
        try:
            # Process single image
            inputs = self.clip_processor(
                images=[image], 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                normalized_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return {
                'embedding': normalized_features[0].cpu().numpy(),
                'shape': list(normalized_features.shape)
            }
            
        except Exception as e:
            logger.error(f"Error processing single image: {e}")
            return {'embedding': np.zeros(self.config.embedding_dim), 'error': str(e)}
