"""
Digit Recognizer Module

Handles digit recognition using PaddleOCR v3.x with official PP-OCRv5 pretrained model.

Author: Lucien
Email: lucien-6@qq.com
License: MIT License
Date: 2025-12-26
"""

import os
import cv2
import numpy as np
import re
import tempfile
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Disable OneDNN/MKL-DNN to avoid compatibility issues
# Must be set BEFORE importing PaddlePaddle/PaddleOCR
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'

# Disable fused operations that may cause OneDNN errors
os.environ['FLAGS_conv_workspace_size_limit'] = '4096'
os.environ['FLAGS_cudnn_exhaustive_search'] = '1'
os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = '1'

# Disable PaddleOCR debug log output
os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'

# CRITICAL: Import paddle FIRST, set flags, THEN import PaddleOCR
# This order is essential to prevent OneDNN from being initialized
import paddle

# Force disable OneDNN at Paddle runtime level (more reliable than env vars alone)
# This MUST be called BEFORE importing PaddleOCR
# Critical for PaddlePaddle 2.6.1+ with custom SVTR models
paddle.set_flags({'FLAGS_use_mkldnn': False})

# NOW import PaddleOCR after flags are set
from paddleocr import TextRecognition


class DigitRecognizer:
    """Recognizes digits using PaddleOCR v3.x with official PP-OCRv5_server_rec pretrained model"""
    
    # Preprocessing constants
    MIN_OCR_HEIGHT = 32  # Minimum height for OCR processing (pixels) - PaddleOCR recommended
    COMPONENT_SPACING_WITH_CLOSING = 50  # Spacing between components when closing enabled (pixels)
    COMPONENT_SPACING_DEFAULT = 20  # Default spacing between components (pixels)
    
    # Recognition constants
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold for digital displays
    DEFAULT_CHAR_WHITELIST = '-0123456789. '  # Allowed characters for digit recognition
    
    def __init__(self):
        """
        Initialize digit recognizer with PaddleOCR
        """
        self.ocr = None
        self.confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
        self.char_whitelist = self.DEFAULT_CHAR_WHITELIST
        self._initialized = False
        self.closing_size = 0  # Closing operation kernel size (0 = disabled)
        self.last_confidence = 0.0  # Store last recognition confidence
    
    def _lazy_init_ocr(self):
        r"""
        Lazy initialization of PaddleOCR v3.x TextRecognition with PP-OCRv5 Official Model
        
        Uses official pretrained PP-OCRv5_server_rec model from PaddleOCR:
        - Pretrained by PaddleOCR official team (no training required)
        - Server-level model optimized for high accuracy
        - Supports Chinese and English text recognition
        - CPU-friendly inference for production deployment
        
        Model Files (located in ./PP-OCRv5_server_rec/):
        - inference.json: Model structure (Inference format)
        - inference.pdiparams: Model weights
        - inference.yml: Model configuration
        
        Character Filtering:
        - Uses model's built-in dictionary (supports full Chinese/English)
        - Application-level whitelist filtering: 0-9, minus sign (-), decimal point (.), space
        - Whitelist reduces false positives for digit-only scenarios
        
        Model Architecture:
        - PP-OCRv5: Latest version of PaddleOCR recognition model
        - High accuracy and robust performance
        - Optimized for various text scenarios
        
        Reference (v3.x API):
        https://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html
        
        Model Download:
        https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_server_rec.tar
        """
        if not self._initialized:
            # Check if PP-OCRv5 official pretrained model files exist
            model_dir = './PP-OCRv5_server_rec'
            
            # Check if official PP-OCRv5 Inference model files exist
            inference_params = os.path.join(model_dir, 'inference.pdiparams')
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(
                    f"PP-OCRv5 model directory not found: {model_dir}\n"
                    f"Please download the official PP-OCRv5_server_rec model from:\n"
                    f"https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_server_rec.tar"
                )
            
            if not os.path.exists(inference_params):
                raise FileNotFoundError(
                    f"PP-OCRv5 model files not found in: {model_dir}\n"
                    f"Expected Inference model files: inference.pdiparams + inference.json\n"
                    f"Please download the official model or check the model directory."
                )
            
            logger.info(f"Loading PP-OCRv5 official pretrained model from: {model_dir}")
            logger.info(f"Model files: inference.pdiparams + inference.json")
            logger.info(f"Model: PP-OCRv5_server_rec (official pretrained, no training required)")
            logger.info(f"Using model's built-in dictionary (character filtering via whitelist)")
            
            # Initialize PaddleOCR with Inference model
            
            logger.info("Initializing PaddleOCR v3.x TextRecognition with PP-OCRv5_server_rec model...")
            self.ocr = TextRecognition(
                # Core parameters for Inference model (v3.x API)
                model_dir=model_dir,                  # Path to PP-OCRv5 official model directory
                
                # Device selection (v3.x: use_gpu=False -> device="cpu")
                device="cpu",                         # Use CPU for maximum compatibility
                
                # Note: Model uses built-in dictionary (full Chinese/English support)
                # Character filtering is done at application level via whitelist
            )
            logger.info("PaddleOCR v3.x TextRecognition initialized successfully")
            logger.info("Using official PP-OCRv5_server_rec pretrained model")
            logger.info("Using CPU inference for maximum compatibility")
            
            self._initialized = True
            logger.info("PP-OCRv5_server_rec model initialized successfully!")
            logger.info("Ready for high-accuracy text recognition (no training required)")
    
    def set_confidence_threshold(self, threshold):
        """
        Set confidence threshold for OCR results
        
        Args:
            threshold: Confidence threshold (0.0 - 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_closing_size(self, closing_size):
        """
        Set closing operation kernel size
        
        Args:
            closing_size: Kernel size for closing operation (0 = disabled, >0 = enabled)
        """
        self.closing_size = int(closing_size)
    
    def preprocess_for_ocr(self, thresh_img):
        """
        Additional preprocessing to improve OCR accuracy
        (Keep existing preprocessing logic)
        
        Args:
            thresh_img: Binary image from ImageProcessor or original ROI image (BGR)
        
        Returns:
            numpy.ndarray: Enhanced image for OCR
        """
        # Check if input is color image (3 channels) or grayscale/binary (2 channels)
        if len(thresh_img.shape) == 3:
            # Color image (BGR format), return as-is without binary-specific processing
            return thresh_img
        
        # Grayscale/binary image processing
        # Expand horizontal spacing between components only if closing is enabled
        if self.closing_size > 0:
            thresh_img = self._expand_component_spacing(
                thresh_img, 
                spacing=self.COMPONENT_SPACING_WITH_CLOSING
            )
        
        # Resize image to improve OCR accuracy (height should be at least MIN_OCR_HEIGHT pixels)
        h, w = thresh_img.shape
        scale_factor = max(1.0, self.MIN_OCR_HEIGHT / h)
        
        if scale_factor > 1:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            thresh_img = cv2.resize(thresh_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return thresh_img
    
    def _expand_component_spacing(self, image, spacing=None):
        """
        Expand horizontal spacing between connected components
        (Keep existing implementation unchanged)
        
        This method identifies all connected components in the image,
        sorts them by x-coordinate (left to right), and inserts
        additional horizontal spacing between adjacent components
        to improve OCR recognition accuracy.
        
        Args:
            image: Binary image (white digits on black background)
            spacing: Number of pixels to insert between components 
                (default: COMPONENT_SPACING_DEFAULT)
        
        Returns:
            numpy.ndarray: Image with expanded component spacing
        
        Note:
            - Original left and right margins are preserved
            - Components are sorted by horizontal position (x-coordinate)
            - Black background (0) is inserted as spacing
        """
        if spacing is None:
            spacing = self.COMPONENT_SPACING_DEFAULT
        # Find connected components
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(image, connectivity=8)
        
        # Need at least 2 components (excluding background)
        if num_labels <= 2:
            return image
        
        # Extract component information and sort by x-coordinate
        components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            components.append({
                'label': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        
        # Sort components by x-coordinate (left to right)
        components.sort(key=lambda c: c['x'])
        
        # Calculate margins
        left_margin = components[0]['x']
        last_comp = components[-1]
        right_margin = image.shape[1] - (last_comp['x'] + last_comp['w'])
        
        # Calculate total component width
        total_component_width = sum(c['w'] for c in components)
        
        # Calculate new width
        num_gaps = len(components) - 1
        new_width = (left_margin + total_component_width + 
                     (num_gaps * spacing) + right_margin)
        
        # Create new image with black background
        new_image = np.zeros((image.shape[0], new_width), dtype=np.uint8)
        
        # Copy each component to new position
        current_x = left_margin
        for comp in components:
            # Extract component ROI
            roi = image[comp['y']:comp['y']+comp['h'], 
                       comp['x']:comp['x']+comp['w']]
            
            # Extract component mask
            mask_roi = labels[comp['y']:comp['y']+comp['h'], 
                             comp['x']:comp['x']+comp['w']]
            component_mask = (mask_roi == comp['label'])
            
            # Copy component to new position using mask
            target_roi = new_image[comp['y']:comp['y']+comp['h'], 
                                  current_x:current_x+comp['w']]
            target_roi[component_mask] = roi[component_mask]
            
            # Update position for next component
            current_x += comp['w'] + spacing
        
        return new_image
    
    def recognize(self, thresh_img, original_roi=None, log_callback=None, result_img_path=None):
        """
        Recognize digits from preprocessed image using PaddleOCR v3.x TextRecognition with PP-OCRv5 Model
        
        Args:
            thresh_img: Preprocessed binary image or original ROI image (BGR)
            original_roi: Original ROI image (optional, for fallback)
            log_callback: Optional callback function for logging details.
                         Called as log_callback(message, level) where level is 'info', 'warning', or 'error'
            result_img_path: Optional path to save OCR result visualization image.
                            If provided, saves the result image with detection boxes and text annotations.
        
        Returns:
            str: Recognized number string or None
        """
        def log(message, level='info'):
            """Internal logging helper"""
            if log_callback:
                log_callback(message, level)
        
        try:
            # Lazy initialize OCR
            self._lazy_init_ocr()
            
            # Try with preprocessed image first
            ocr_img = self.preprocess_for_ocr(thresh_img)
            
            # Convert image to RGB for PaddleOCR
            if len(ocr_img.shape) == 2:
                # Grayscale/binary image: convert to RGB
                ocr_img_rgb = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2RGB)
            elif len(ocr_img.shape) == 3:
                # Color image: convert from BGR to RGB
                ocr_img_rgb = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2RGB)
            else:
                ocr_img_rgb = ocr_img
            
            # Save image to temporary file for predict() API
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, ocr_img_rgb)
            
            try:
                # Perform OCR using PP-OCRv5 official model with v3.x API
                # Official API usage (from PaddleOCR v3.x documentation):
                # https://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html
                # result = model.predict(input='image.jpg', batch_size=1)
                # for res in result:
                #     text = res['rec_text']
                #     confidence = res['rec_score']
                #
                # v3.x TextRecognition is recognition-only by default (no detection or classification)
                result = self.ocr.predict(input=temp_path, batch_size=1)
                
                # Parse result according to v3.x official format
                # Expected format: [ResultObject, ...]
                # Each ResultObject is dict-like with keys: 'rec_text', 'rec_score', etc.
                
                if not result:
                    log("PaddleOCR returned None (no text detected)", "warning")
                    return None
                
                # Log only key fields (rec_text and rec_score) for readability
                simplified_results = [
                    {'rec_text': r.get('rec_text', ''), 'rec_score': r.get('rec_score', 0.0)}
                    for r in result if isinstance(r, dict)
                ]
                log(f"OCR result: {simplified_results}", "info")
                
                # Extract recognition results according to v3.x API
                # Format: result[i] = {'rec_text': 'text', 'rec_score': confidence, ...}
                try:
                    # v3.x returns a list of ResultObject instances
                    if not isinstance(result, list) or len(result) == 0:
                        log(f"Unexpected result structure: {type(result)}", "warning")
                        return None
                    
                    # Convert v3.x TextRecResult format to internal format
                    # TextRecResult objects are dict-like, directly access keys
                    all_detections = []
                    for res_obj in result:
                        # Access result attributes from TextRecResult (dict-like object)
                        if isinstance(res_obj, dict):
                            text = res_obj.get('rec_text', '')
                            score = res_obj.get('rec_score', 0.0)
                            all_detections.append((text, score))
                        else:
                            log(f"Unexpected result structure: {type(res_obj)}", "warning")
                            continue
                            
                except (AttributeError, KeyError, TypeError) as e:
                    log(f"Error parsing result: {e}", "error")
                    return None
                
                if not all_detections:
                    log("Empty recognition result", "warning")
                    return None
                
                log(f"Successfully recognized {len(all_detections)} result(s)", "info")
                
                # Extract text with highest confidence
                best_text = None
                best_confidence = 0.0
                all_texts = []  # Store all detected texts for logging
                filtered_out_by_confidence = []  # Texts filtered by confidence threshold
                filtered_out_by_whitelist = []  # Texts filtered by whitelist
                
                # Iterate through recognition results
                # Official format with det=False: each result is a tuple (text, confidence)
                for detection in all_detections:
                    # Expected format: (text, confidence)
                    if isinstance(detection, tuple) and len(detection) == 2:
                        text, score = detection
                        log(f"Recognized: '{text}' with confidence {score:.3f}", "info")
                    else:
                        log(f"Unexpected detection format: {type(detection)} - {detection}", "warning")
                        continue
                    
                    all_texts.append((text, score))
                    
                    # Filter by confidence threshold
                    if score < self.confidence_threshold:
                        filtered_out_by_confidence.append((text, score))
                        continue
                    
                    # Filter characters using whitelist
                    filtered_text = self._filter_text(text)
                    
                    if not filtered_text:
                        filtered_out_by_whitelist.append((text, score))
                        continue
                    
                    if filtered_text and score > best_confidence:
                        best_text = filtered_text
                        best_confidence = score
                
                # Log all detected texts
                if all_texts:
                    text_list = [f"'{t}' (conf={s:.3f})" for t, s in all_texts]
                    log(f"All detected texts: {', '.join(text_list)}", "info")
                
                # Log filtered texts
                if filtered_out_by_confidence:
                    conf_list = [f"'{t}' (conf={s:.3f})" for t, s in filtered_out_by_confidence]
                    log(f"Filtered by confidence threshold ({self.confidence_threshold:.3f}): {', '.join(conf_list)}", "info")
                
                if filtered_out_by_whitelist:
                    whitelist_list = [f"'{t}' (conf={s:.3f})" for t, s in filtered_out_by_whitelist]
                    log(f"Filtered by whitelist (allowed: {self.char_whitelist}): {', '.join(whitelist_list)}", "info")
                
                # Save result image if path provided
                if result_img_path and result:
                    try:
                        # Draw recognition results on image for visualization
                        from PIL import Image, ImageDraw, ImageFont
                        
                        # Load original image
                        vis_img = Image.open(temp_path).convert('RGB')
                        draw = ImageDraw.Draw(vis_img)
                        
                        # When det=False, we don't have bounding boxes
                        # Just add text annotation at the top of the image
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw recognition results as text overlay
                        y_pos = 10
                        for i, (text, score) in enumerate(all_texts):
                            text_str = f"Result {i+1}: '{text}' (conf={score:.3f})"
                            draw.text((10, y_pos), text_str, fill='red', font=font)
                            y_pos += 30
                        
                        # Highlight the best result
                        if best_text:
                            best_str = f"BEST: '{best_text}' (conf={best_confidence:.3f})"
                            draw.text((10, y_pos), best_str, fill='green', font=font)
                        
                        # Create directory if needed
                        result_dir = os.path.dirname(result_img_path)
                        if result_dir and not os.path.exists(result_dir):
                            os.makedirs(result_dir, exist_ok=True)
                        
                        # Save visualization image
                        vis_img.save(result_img_path)
                        log(f"OCR result image saved to: {result_img_path}", "info")
                    except Exception as e:
                        log(f"Failed to save result image: {str(e)}", "warning")
                
                # Save confidence to instance variable for external access
                self.last_confidence = best_confidence
                
                # Validate result format
                if best_text:
                    if self._validate_number_format(best_text):
                        # Remove spaces for numeric processing
                        best_text_clean = best_text.replace(' ', '')
                        log(f"Valid number format detected: '{best_text_clean}' (confidence: {best_confidence:.3f})", "info")
                        return best_text_clean
                    else:
                        log(f"Text '{best_text}' does not match number format pattern (allowed: digits, minus sign, decimal point)", "warning")
                        return None
                else:
                    if all_texts:
                        log("No text passed all filters (confidence threshold and whitelist)", "warning")
                    else:
                        log("No valid text detected in OCR results", "warning")
                    return None
                    
            finally:
                # Clean up temporary file with proper error handling
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except PermissionError:
                        log(f"Failed to delete temporary file (permission denied): {temp_path}", "warning")
                    except OSError as e:
                        log(f"Failed to delete temporary file (OS error): {temp_path} - {str(e)}", "warning")
                
        except FileNotFoundError as e:
            error_msg = f"Model file not found: {str(e)}"
            log(error_msg, "error")
            logger.error(error_msg)
            self.last_confidence = 0.0
            return None
        except ValueError as e:
            error_msg = f"Invalid parameter or data: {str(e)}"
            log(error_msg, "error")
            logger.error(error_msg)
            self.last_confidence = 0.0
            return None
        except RuntimeError as e:
            error_msg = f"Runtime error during recognition: {str(e)}"
            log(error_msg, "error")
            logger.error(error_msg)
            self.last_confidence = 0.0
            return None
        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected error during recognition: {type(e).__name__}: {str(e)}"
            log(error_msg, "error")
            logger.error(error_msg)
            self.last_confidence = 0.0
            return None
    
    def _filter_text(self, text):
        """
        Filter text to only include whitelisted characters
        
        Args:
            text: Original recognized text
        
        Returns:
            str: Filtered text containing only whitelisted characters
        """
        if not text:
            return ""
        
        # Only keep characters in whitelist
        filtered = ''.join(c for c in text if c in self.char_whitelist)
        return filtered
    
    def _validate_number_format(self, text):
        """
        Validate if recognized text is a valid number format
        Supports spaces in numbers (e.g., "1 234.56") which will be removed for numeric conversion
        
        Args:
            text: Recognized text string (may contain spaces)
        
        Returns:
            bool: True if valid number format (after removing spaces)
        """
        if not text:
            return False
        
        # Remove spaces for validation
        text_no_space = text.replace(' ', '')
        
        if not text_no_space:
            return False
        
        # Allow formats: "123", "-123", "12.34", "-12.34", ".5", "-.5"
        # Spaces in original text are acceptable and will be removed
        pattern = r'^-?\d*\.?\d+$'
        return bool(re.match(pattern, text_no_space))
