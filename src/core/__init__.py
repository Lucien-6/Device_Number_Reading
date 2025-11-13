"""
Core Functionality Module

This module contains core functionality including:
- Image processing
- Digit recognition (PaddleOCR)
"""

from .image_processor import ImageProcessor
from .digit_recognizer import DigitRecognizer

__all__ = [
    'ImageProcessor',
    'DigitRecognizer'
]

