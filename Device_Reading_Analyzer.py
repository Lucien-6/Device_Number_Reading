
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Device Readings Analyzer - Main Entry Point

A professional tool for recognizing digital readings from device displays
using computer vision and PaddleOCR SVTR_Tiny technology.

Author: Lucien
Email: lucien-6@qq.com
License: MIT License
Version: 4.0.0
Date: 2025-11-12
"""

import sys
import os
import tkinter as tk

# Disable PaddleOCR debug log output (must be set before importing)
os.environ['DISABLE_AUTO_LOGGING_CONFIG'] = '1'

# Check dependencies
try:
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib
except ImportError as e:
    print(f"Required package missing: {str(e)}")
    print("Please install required packages using pip:")
    print("pip install opencv-contrib-python numpy pandas matplotlib Pillow openpyxl paddlepaddle paddleocr")
    sys.exit(1)

# Check PaddleOCR
try:
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"Required package missing: {str(e)}")
    print("Please install PaddleOCR:")
    print("pip install paddlepaddle paddleocr")
    sys.exit(1)

# Import from modular structure
try:
    from src.main_window import DeviceReadingsAnalyzer
except ImportError as e:
    print(f"Error importing application: {e}")
    print("Please ensure the src/ directory and all modules are present.")
    sys.exit(1)


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = DeviceReadingsAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

