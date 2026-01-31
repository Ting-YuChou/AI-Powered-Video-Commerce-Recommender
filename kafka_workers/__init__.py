"""
Kafka Workers Module
====================

This module contains Kafka consumer workers for processing various event streams:
- video_processor: Processes video content from the video-processing-tasks topic
- feature_updater: Updates user/content features from the user-interactions topic
"""

from .video_processor import VideoProcessorWorker
from .feature_updater import FeatureUpdaterWorker

__all__ = ['VideoProcessorWorker', 'FeatureUpdaterWorker']
