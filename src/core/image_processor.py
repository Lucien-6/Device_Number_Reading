"""
Image Processor Module

Handles image preprocessing and batch processing operations.

Author: Lucien
Email: lucien-6@qq.com
License: MIT License
"""

import cv2
import numpy as np


class ImageProcessor:
    """Processes images for digit recognition"""
    
    def __init__(self):
        """Initialize image processor"""
        self.erosion_size = 0
        self.closing_size = 0
    
    def set_preprocessing_params(self, erosion_size=None, closing_size=None):
        """Set preprocessing parameters"""
        if erosion_size is not None:
            self.erosion_size = int(erosion_size)
        if closing_size is not None:
            self.closing_size = int(closing_size)
    
    def preprocess_roi(self, roi_image):
        """
        Preprocess ROI image for recognition
        
        Args:
            roi_image: ROI region from original image
        
        Returns:
            numpy.ndarray: Preprocessed binary image or original ROI image (BGR)
        """
        # Only perform preprocessing if erosion or closing is enabled
        if self.erosion_size > 0 or self.closing_size > 0:
            # Convert to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Binarize using Otsu's method
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Filter small objects based on bounding box size
            roi_height = roi_image.shape[0]
            min_length = roi_height / 40
            
            # Find connected components (8-connectivity)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, connectivity=8
            )
            
            # Create filtered image
            filtered_thresh = np.zeros_like(thresh)
            
            # Process each component (skip background label 0)
            for i in range(1, num_labels):
                # Get bounding box dimensions
                x, y, w, h, area = stats[i]
                min_length_component = min(w, h)
                
                # Filter out objects with minimum dimension < threshold
                # Keep objects with min length >= threshold
                if min_length_component >= min_length:
                    filtered_thresh[labels == i] = 255
            
            # Replace thresh with filtered version
            thresh = filtered_thresh
            
            # Add border to improve recognition (after filtering)
            border_size = 15
            thresh = cv2.copyMakeBorder(
                thresh, 
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT, 
                value=0
            )
            
            # Apply erosion operation if enabled (erosion_size > 0)
            if self.erosion_size > 0:
                # Apply erosion to entire image (no decimal point protection)
                thresh = self._apply_erosion(thresh, self.erosion_size)
                
                # Detect decimal point component after erosion
                decimal_info = self._detect_decimal_point_component(thresh)
                
                if decimal_info is not None:
                    # Expand spacing between other components and decimal point
                    thresh, decimal_mask, decimal_center = self._expand_spacing_from_decimal(
                        thresh, decimal_info, self.erosion_size
                    )
                    
                    # Dilate other components to restore original state
                    thresh = self._dilate_other_components(
                        thresh, decimal_mask, self.erosion_size
                    )
                    
                    # Replace decimal point with solid circle
                    thresh = self._replace_decimal_with_circle(
                        thresh, decimal_mask, decimal_center, self.erosion_size, roi_height
                    )
            
            # Apply closing operation if enabled (closing_size > 0)
            if self.closing_size > 0:
                # Identify decimal point
                decimal_mask = self._identify_decimal_point(thresh)
                
                if decimal_mask is not None:
                    # Exclude decimal point from closing operation
                    thresh_without_decimal = cv2.bitwise_and(
                        thresh, cv2.bitwise_not(decimal_mask)
                    )
                    
                    # Apply closing to non-decimal regions
                    closed = self._apply_closing(
                        thresh_without_decimal, self.closing_size
                    )
                    
                    # Merge decimal point back
                    thresh = cv2.bitwise_or(closed, decimal_mask)
                else:
                    # No decimal point detected, apply closing to entire image
                    thresh = self._apply_closing(thresh, self.closing_size)
            
            return thresh
        else:
            # No preprocessing enabled, return original ROI image (BGR format)
            return roi_image
    
    def _detect_decimal_point_component(self, thresh):
        """
        Detect decimal point component after erosion operation
        
        The decimal point is identified as the component with minimum area
        that is less than 1/4 of the second smallest component's area.
        
        Args:
            thresh: Binary image after erosion operation
        
        Returns:
            dict or None: Dictionary containing decimal point information:
                - 'label': Component label index
                - 'area': Component area
                - 'x': Bounding box left coordinate
                - 'y': Bounding box top coordinate
                - 'w': Bounding box width
                - 'h': Bounding box height
                - 'mask': Full-size mask with only decimal point (255 for decimal, 0 otherwise)
            None if no decimal point is detected
        """
        # Find connected components
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        # Need at least 2 non-background components (minimum and second minimum)
        if num_labels <= 2:
            return None
        
        # Collect all component areas
        component_areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            component_areas.append({'label': i, 'area': area})
        
        # Sort by area (ascending)
        component_areas.sort(key=lambda x: x['area'])
        
        # Get minimum and second minimum areas
        min_area_info = component_areas[0]
        second_min_area = component_areas[1]['area']
        
        # Check if minimum area is less than 1/4 of second minimum area
        if min_area_info['area'] >= second_min_area / 4.0:
            return None
        
        # Found decimal point component
        decimal_label = min_area_info['label']
        
        # Get bounding box information
        x = stats[decimal_label, cv2.CC_STAT_LEFT]
        y = stats[decimal_label, cv2.CC_STAT_TOP]
        w = stats[decimal_label, cv2.CC_STAT_WIDTH]
        h = stats[decimal_label, cv2.CC_STAT_HEIGHT]
        
        # Create full-size mask with only decimal point
        decimal_mask = np.zeros_like(thresh)
        decimal_mask[labels == decimal_label] = 255
        
        return {
            'label': decimal_label,
            'area': min_area_info['area'],
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'mask': decimal_mask
        }
    
    def _expand_spacing_from_decimal(self, thresh, decimal_info, kernel_size):
        """
        Expand horizontal spacing between other components and decimal point
        
        After identifying the decimal point, increase the horizontal distance
        between all other components and the decimal point by 4 times the
        erosion kernel size. The decimal point position remains unchanged.
        
        Args:
            thresh: Binary image after erosion
            decimal_info: Dictionary containing decimal point information
            kernel_size: Erosion kernel size (radius)
        
        Returns:
            tuple: (new_image, decimal_mask, decimal_center)
                - new_image: Image with expanded spacing
                - decimal_mask: Updated decimal point mask (position unchanged)
                - decimal_center: Tuple (center_x, center_y) of decimal point center
        """
        if kernel_size <= 0:
            # Calculate decimal point center coordinates
            decimal_center_x = decimal_info['x'] + decimal_info['w'] // 2
            decimal_center_y = decimal_info['y'] + decimal_info['h'] // 2
            return thresh, decimal_info['mask'], (decimal_center_x, decimal_center_y)
        
        # Calculate offset distance (4 times kernel size)
        offset = kernel_size * 4
        
        # Find all connected components
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        # Get decimal point center x coordinate
        decimal_x = decimal_info['x'] + decimal_info['w'] // 2
        
        # Collect all non-decimal components with their information
        components = []
        for i in range(1, num_labels):
            if i == decimal_info['label']:
                continue  # Skip decimal point
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            center_x = x + w // 2
            
            components.append({
                'label': i,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': center_x
            })
        
        # If no other components, return original image
        if len(components) == 0:
            # Calculate decimal point center coordinates
            decimal_center_x = decimal_info['x'] + decimal_info['w'] // 2
            decimal_center_y = decimal_info['y'] + decimal_info['h'] // 2
            return thresh, decimal_info['mask'], (decimal_center_x, decimal_center_y)
        
        # Calculate maximum offset needed
        max_left_offset = 0
        max_right_offset = 0
        
        for comp in components:
            if comp['center_x'] < decimal_x:
                # Component is to the left of decimal point, move left
                max_left_offset = max(max_left_offset, offset)
            else:
                # Component is to the right of decimal point, move right
                max_right_offset = max(max_right_offset, offset)
        
        # Calculate new image dimensions
        h, w = thresh.shape
        new_w = w + max_left_offset + max_right_offset
        
        # Create new image with black background
        new_image = np.zeros((h, new_w), dtype=np.uint8)
        
        # Copy decimal point to new position (centered with left offset)
        decimal_mask_new = np.zeros((h, new_w), dtype=np.uint8)
        decimal_x_new = decimal_info['x'] + max_left_offset
        decimal_y = decimal_info['y']
        decimal_w = decimal_info['w']
        decimal_h = decimal_info['h']
        
        # Extract decimal point from original image
        decimal_roi = thresh[decimal_y:decimal_y+decimal_h,
                           decimal_info['x']:decimal_info['x']+decimal_w]
        decimal_mask_roi = decimal_info['mask'][decimal_y:decimal_y+decimal_h,
                                               decimal_info['x']:decimal_info['x']+decimal_w]
        
        # Copy decimal point to new position
        new_image[decimal_y:decimal_y+decimal_h,
                 decimal_x_new:decimal_x_new+decimal_w] = decimal_roi
        decimal_mask_new[decimal_y:decimal_y+decimal_h,
                        decimal_x_new:decimal_x_new+decimal_w] = decimal_mask_roi
        
        # Process each non-decimal component
        for comp in components:
            # Calculate new x position
            if comp['center_x'] < decimal_x:
                # Move left by offset
                new_x = comp['x'] - offset + max_left_offset
            else:
                # Move right by offset
                new_x = comp['x'] + offset + max_left_offset
            
            # Extract component from original image
            comp_roi = thresh[comp['y']:comp['y']+comp['h'],
                            comp['x']:comp['x']+comp['w']]
            
            # Extract component mask
            mask_roi = labels[comp['y']:comp['y']+comp['h'],
                            comp['x']:comp['x']+comp['w']]
            component_mask = (mask_roi == comp['label']).astype(np.uint8) * 255
            
            # Copy component to new position using mask
            target_roi = new_image[comp['y']:comp['y']+comp['h'],
                                 new_x:new_x+comp['w']]
            target_mask = component_mask > 0
            target_roi[target_mask] = comp_roi[target_mask]
        
        # Calculate decimal point center coordinates in new image
        decimal_center_x = decimal_x_new + decimal_w // 2
        decimal_center_y = decimal_y + decimal_h // 2
        
        return new_image, decimal_mask_new, (decimal_center_x, decimal_center_y)
    
    def _dilate_other_components(self, thresh, decimal_mask, kernel_size):
        """
        Dilate all components except the decimal point by 1 times the erosion kernel size
        
        This restores the other objects to their original state after erosion.
        
        Args:
            thresh: Binary image with expanded spacing
            decimal_mask: Mask of decimal point (255 for decimal, 0 otherwise)
            kernel_size: Erosion kernel size (radius)
        
        Returns:
            numpy.ndarray: Image with dilated other components
        """
        if kernel_size <= 0:
            return thresh
        
        # Remove decimal point from image
        thresh_without_decimal = cv2.bitwise_and(
            thresh, cv2.bitwise_not(decimal_mask)
        )
        
        # Create dilation kernel (rectangular with same size as erosion kernel)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size * 2 + 1, kernel_size * 2 + 1)
        )
        
        # Dilate all non-decimal components
        dilated_others = cv2.dilate(thresh_without_decimal, kernel, iterations=1)
        
        # Merge decimal point back (unchanged)
        result = cv2.bitwise_or(dilated_others, decimal_mask)
        
        return result
    
    def _replace_decimal_with_circle(self, thresh, decimal_mask, decimal_center, kernel_size, roi_height):
        """
        Replace decimal point with a solid circle
        
        Draws a solid circle with diameter equal to the maximum of
        (4 times the erosion kernel size, ROI height / 15), rounded to integer.
        The original decimal point is removed before drawing the new circle.
        
        Args:
            thresh: Binary image after spacing expansion and other components dilation
            decimal_mask: Mask of original decimal point (255 for decimal, 0 otherwise)
            decimal_center: Tuple (center_x, center_y) of decimal point center
            kernel_size: Erosion kernel size (radius)
            roi_height: Original ROI region height
        
        Returns:
            numpy.ndarray: Image with decimal point replaced by solid circle
        """
        if kernel_size <= 0:
            return thresh
        
        # Remove original decimal point from image
        result = cv2.bitwise_and(thresh, cv2.bitwise_not(decimal_mask))
        
        # Calculate circle diameter: max(4 * kernel_size, roi_height / 15), rounded to integer
        diameter_1 = 4 * kernel_size
        diameter_2 = roi_height / 15.0
        diameter = int(max(diameter_1, diameter_2))
        
        # Calculate circle radius (radius = diameter / 2)
        radius = diameter // 2
        
        # Get center coordinates
        center_x, center_y = decimal_center
        
        # Ensure coordinates are within image bounds
        h, w = result.shape
        center_x = max(radius, min(w - radius - 1, int(center_x)))
        center_y = max(radius, min(h - radius - 1, int(center_y)))
        
        # Draw solid circle (thickness=-1 means filled)
        cv2.circle(result, (center_x, center_y), radius, 255, thickness=-1)
        
        return result
    
    def _dilate_decimal_point(self, thresh, decimal_mask, kernel_size):
        """
        Dilate decimal point by 1 times the erosion kernel size
        
        Args:
            thresh: Binary image with expanded spacing
            decimal_mask: Mask of decimal point (255 for decimal, 0 otherwise)
            kernel_size: Erosion kernel size (radius)
        
        Returns:
            numpy.ndarray: Image with dilated decimal point
        """
        if kernel_size <= 0:
            return thresh
        
        # Create dilation kernel (rectangular with same size as erosion kernel)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size * 2 + 1, kernel_size * 2 + 1)
        )
        
        # Dilate only the decimal point mask
        dilated_decimal = cv2.dilate(decimal_mask, kernel, iterations=1)
        
        # Remove original decimal point from image
        thresh_without_decimal = cv2.bitwise_and(
            thresh, cv2.bitwise_not(decimal_mask)
        )
        
        # Merge dilated decimal point back
        result = cv2.bitwise_or(thresh_without_decimal, dilated_decimal)
        
        return result
    
    def _identify_decimal_point(self, thresh):
        """
        Identify decimal point in binary image
        
        Args:
            thresh: Binary image after filtering
        
        Returns:
            numpy.ndarray or None: Mask of decimal point 
                (only decimal point pixels are 255),
                None if no decimal point is identified
        
        Note:
            Decimal point is identified as the smallest connected 
            component with aspect ratio (long/short axis) < 1.25
        """
        # Find connected components
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        # Need at least one non-background component
        if num_labels <= 1:
            return None
        
        # Find the component with minimum area
        min_area = float('inf')
        min_area_idx = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                min_area = area
                min_area_idx = i
        
        # Get bounding box of the smallest component
        x = stats[min_area_idx, cv2.CC_STAT_LEFT]
        y = stats[min_area_idx, cv2.CC_STAT_TOP]
        w = stats[min_area_idx, cv2.CC_STAT_WIDTH]
        h = stats[min_area_idx, cv2.CC_STAT_HEIGHT]
        
        # Extract component mask for contour analysis
        component_mask = (labels[y:y+h, x:x+w] == min_area_idx).astype(
            np.uint8
        ) * 255
        
        # Find contours to get minimum area rectangle
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contours[0])
        width, height = rect[1]
        
        # Calculate aspect ratio (long axis / short axis)
        if min(width, height) < 1e-6:
            return None
        
        aspect_ratio = max(width, height) / min(width, height)
        
        # Check if it's a decimal point
        if aspect_ratio < 1.25:
            # Create full-size mask with only decimal point
            decimal_mask = np.zeros_like(thresh)
            decimal_mask[labels == min_area_idx] = 255
            return decimal_mask
        
        return None
    
    def _apply_erosion(self, image, kernel_size):
        """
        Apply morphological erosion operation with rectangular kernel
        
        Args:
            image: Binary image
            kernel_size: Size of structuring element (half of side length)
        
        Returns:
            numpy.ndarray: Image after erosion operation
        """
        # Use MORPH_RECT as rectangular kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size * 2 + 1, kernel_size * 2 + 1)
        )
        result = cv2.erode(image, kernel, iterations=1)
        return result
    
    def _apply_closing(self, image, kernel_size):
        """
        Apply morphological closing operation with rectangular kernel
        
        Args:
            image: Binary image
            kernel_size: Size of structuring element (width and height of rectangle)
        
        Returns:
            numpy.ndarray: Image after closing operation
        """
        # Use MORPH_RECT for rectangular kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result
    
    def extract_roi(self, image, roi_coords):
        """
        Extract ROI region from image
        
        Args:
            image: Full image
            roi_coords: (start_x, start_y, end_x, end_y)
        
        Returns:
            numpy.ndarray: ROI region
        """
        start_x, start_y, end_x, end_y = roi_coords
        return image[start_y:end_y, start_x:end_x]

