"""
Digit Recognizer Module

Handles digit recognition using PaddleOCR.

Author: Lucien
Email: lucien-6@qq.com
License: MIT License
Date: 2025-11-13
"""

import os
import cv2
import numpy as np
import re
import tempfile

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
from paddleocr import PaddleOCR


class DigitRecognizer:
    """Recognizes digits using PaddleOCR"""
    
    def __init__(self):
        """
        Initialize digit recognizer with PaddleOCR
        """
        self.ocr = None
        self.confidence_threshold = 0.3  # Lower threshold to see more results
        self.char_whitelist = '-0123456789.'
        self._initialized = False
        self.closing_size = 0  # Closing operation kernel size (0 = disabled)
        self.last_confidence = 0.0  # Store last recognition confidence
    
    def _lazy_init_ocr(self):
        r"""
        Lazy initialization of PaddleOCR with SVTR_Tiny Digital Model
        
        Uses custom-trained SVTR_Tiny model specifically optimized for digital display recognition:
        - Trained on digital display/LED segment datasets
        - Lightweight architecture (~24MB model size)
        - High accuracy for digit recognition (0-9, minus sign, decimal point)
        - Faster inference compared to general-purpose models
        
        Model Files (located in ./svtr_tiny_digital/):
        - best_accuracy.pdparams: Model weights (~24MB)
        - config.yml: Model configuration
        
        Custom Dictionary (./digital_dict.txt):
        - Contains only: 0-9, minus sign (-), decimal point (.)
        - Reduces confusion and improves accuracy
        
        Model Architecture:
        - Algorithm: SVTR (Scene Text Recognition with a Single Visual Model)
        - Input shape: 3 x 64 x 256 (H x W)
        - Decoder: CTC (Connectionist Temporal Classification)
        - Transform: STN (Spatial Transformer Network) for perspective correction
        
        Reference:
        https://www.paddleocr.ai/v2.9.1/applications/%E5%85%89%E5%8A%9F%E7%8E%87%E8%AE%A1%E6%95%B0%E7%A0%81%E7%AE%A1%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB.html
        """
        if not self._initialized:
            # Check if SVTR_Tiny Inference model files exist
            model_dir = './svtr_tiny_digital'
            
            # Prefer Inference model over training model for better performance
            inference_model = os.path.join(model_dir, 'inference.pdmodel')
            inference_params = os.path.join(model_dir, 'inference.pdiparams')
            training_model = os.path.join(model_dir, 'best_accuracy.pdparams')
            dict_file = './digital_dict.txt'
            
            # Check which model is available
            use_inference_model = os.path.exists(inference_model) and os.path.exists(inference_params)
            use_training_model = os.path.exists(training_model)
            
            if not use_inference_model and not use_training_model:
                raise FileNotFoundError(
                    f"SVTR_Tiny model files not found in: {model_dir}\n"
                    f"Expected either:\n"
                    f"  - Inference model: inference.pdmodel + inference.pdiparams (recommended)\n"
                    f"  - Training model: best_accuracy.pdparams\n"
                    f"Please ensure the model files are present or refer to MODEL_TRAINING_GUIDE.md"
                )
            
            if not os.path.exists(dict_file):
                raise FileNotFoundError(
                    f"Dictionary file not found: {dict_file}\n"
                    f"Please ensure the digital_dict.txt file exists in the project root."
                )
            
            # Determine which model to use
            if use_inference_model:
                model_type = "Inference model (optimized for deployment)"
                print(f"[INFO] Loading SVTR_Tiny Inference model from: {model_dir}")
                print(f"[INFO] Model files: inference.pdmodel + inference.pdiparams")
            else:
                model_type = "Training model (will be slower)"
                print(f"[INFO] Loading SVTR_Tiny training model from: {model_dir}")
                print(f"[INFO] Model file: best_accuracy.pdparams")
                print(f"[WARNING] Consider converting to Inference model for better performance")
            
            print(f"[INFO] Model type: {model_type}")
            print(f"[INFO] Model size: ~24 MB")
            print(f"[INFO] Custom dictionary: {dict_file} (0-9, -, .)")
            
            # Initialize PaddleOCR with Inference model
            # API Reference from official documentation:
            # https://www.paddleocr.ai/v2.9.1/applications/%E5%85%89%E5%8A%9F%E7%8E%87%E8%AE%A1%E6%95%B0%E7%A0%81%E7%AE%A1%E5%AD%97%E7%AC%A6%E8%AF%86%E5%88%AB.html
            # https://aistudio.baidu.com/projectdetail/4049044
            #
            # Correct usage for Inference model:
            # ocr = PaddleOCR(rec_model_dir='digital_infer', rec_char_dict_path='digital_dict.txt')
            # result = ocr.ocr('image.jpg', det=False)
            # text, confidence = result[0][0]  # Output: ('-70.00', 0.9998)
            
            print("[INFO] Initializing PaddleOCR with SVTR_Tiny Inference model...")
            self.ocr = PaddleOCR(
                # Core parameters for Inference model
                rec_model_dir=model_dir,              # Path to Inference model directory
                # NOTE: Do NOT specify rec_char_dict_path - let model use its internal dictionary
                # The model was trained with a specific dictionary embedded in it
                
                # Optional: Use GPU for faster inference (set to False if encountering issues)
                use_gpu=False,                        # Use CPU for maximum compatibility
                
                # Suppress verbose logging
                show_log=False,
            )
            print("[INFO] PaddleOCR initialized successfully")
            print("[INFO] Using model's internal dictionary (trained with specific character set)")
            print("[INFO] Using CPU inference for maximum compatibility")
            
            self._initialized = True
            print("[INFO] SVTR_Tiny digital recognition model initialized successfully!")
            print("[INFO] Ready for high-accuracy digit recognition")
    
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
            thresh_img = self._expand_component_spacing(thresh_img, spacing=50)
        
        # Resize image to improve OCR accuracy (height should be at least 32 pixels)
        h, w = thresh_img.shape
        scale_factor = max(1.0, 32.0 / h)
        
        if scale_factor > 1:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            thresh_img = cv2.resize(thresh_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return thresh_img
    
    def _expand_component_spacing(self, image, spacing=20):
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
                (default: 20)
        
        Returns:
            numpy.ndarray: Image with expanded component spacing
        
        Note:
            - Original left and right margins are preserved
            - Components are sorted by horizontal position (x-coordinate)
            - Black background (0) is inserted as spacing
        """
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
        Recognize digits from preprocessed image using PaddleOCR SVTR_Tiny Model
        
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
                # Perform OCR using SVTR_Tiny Inference model
                # Official API usage (from PaddleOCR documentation):
                # result = ocr.ocr('image.jpg', det=False)
                # text, confidence = result[0][0]  # Output: ('-70.00', 0.9998)
                #
                # det=False: Only recognition, no text detection (input is already cropped ROI)
                # cls=False: No angle classification (not needed for horizontal digits)
                result = self.ocr.ocr(temp_path, det=False, cls=False)
                
                # Parse result according to official format
                # Expected format: [[('text', confidence), ...]] or [('text', confidence)]
                log(f"Raw OCR result: {result}", "info")
                
                if not result:
                    log("PaddleOCR returned None (no text detected)", "warning")
                    return None
                
                # Extract recognition results according to official API
                # Format: result[0][0] = (text, confidence)
                try:
                    if isinstance(result[0], list) and len(result[0]) > 0:
                        # Standard format: [[('text', conf)]]
                        all_detections = result[0]
                    elif isinstance(result[0], tuple):
                        # Simplified format: [('text', conf)]
                        all_detections = result
                    else:
                        log(f"Unexpected result structure: {type(result[0])}", "warning")
                        return None
                except (IndexError, TypeError) as e:
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
                        log(f"Valid number format detected: '{best_text}' (confidence: {best_confidence:.3f})", "info")
                        return best_text
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
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
        except Exception as e:
            # Log error but don't print stack trace in production
            error_msg = f"Recognition failed with exception: {type(e).__name__}: {str(e)}"
            log(error_msg, "error")
            print(f"[ERROR] {error_msg}")
            self.last_confidence = 0.0  # Reset confidence on error
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
        
        Args:
            text: Recognized text string
        
        Returns:
            bool: True if valid number format
        """
        if not text:
            return False
        
        # Allow formats: "123", "-123", "12.34", "-12.34", ".5", "-.5"
        pattern = r'^-?\d*\.?\d+$'
        return bool(re.match(pattern, text))
