"""
Main Window Module

This module contains the main application window with full functionality.

Author: Lucien
Email: lucien-6@qq.com
License: MIT License
"""

import os
import cv2
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import datetime
import time
import threading
import tempfile
import numpy as np
from src.core.image_processor import ImageProcessor
from src.core.digit_recognizer import DigitRecognizer

class DeviceReadingsAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Device Readings Analyzer")
        self.root.geometry("1200x1100")
        self.root.minsize(800, 800)
        
        # State variables
        self.image_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.total_images = 0
        self.roi_selecting = False
        self.roi_selected = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.roi_end_x = 0
        self.roi_end_y = 0
        self.readings = []
        self.time_values = []
        self.confidences = []
        self.processing_thread = None
        self.stop_processing = False
        
        # Image processor
        self.image_processor = ImageProcessor()
        
        # Digit recognizer
        self.digit_recognizer = DigitRecognizer()
        
        # Configuration
        self.time_interval = tk.DoubleVar(value=1.0)
        self.time_unit = tk.StringVar(value="s")
        self.reading_unit = tk.StringVar(value="")
        self.start_time = tk.DoubleVar(value=0.0)
        self.erosion_size = tk.IntVar(value=0)
        self.closing_size = tk.IntVar(value=0)
        self.decimal_position = tk.StringVar(value="Keep")
        
        # Application metadata
        self.version = "4.0.1"
        self.author = "Lucien"
        self.email = "lucien-6@qq.com"
        self.license = "MIT License"
        
        # Preview photo reference
        self.preview_photo = None
        
        # Create the main UI
        self.create_ui()
        
        # Initialize log
        self.log("Application started. Please load an image sequence.")
    
    def create_ui(self):
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Image Sequence", command=self.load_image_sequence, accelerator="Ctrl+L")
        self.file_menu.add_command(label="Export to Excel", command=self.export_to_excel, accelerator="Ctrl+E", state=tk.DISABLED)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help_window, accelerator="Ctrl+H")
        help_menu.add_command(label="About", command=self.show_about_dialog)
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for image display and control panels
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create image panel (left side of top frame)
        self.image_panel_frame = ttk.LabelFrame(top_frame, text="Image Preview")
        self.image_panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas for image display
        self.canvas_frame = ttk.Frame(self.image_panel_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.resize_image)
        
        # Bind mouse events for ROI selection
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Image navigation controls
        nav_frame = ttk.Frame(self.image_panel_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create image slider
        
        self.image_slider = ttk.Scale(
            nav_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.image_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.image_counter_label = ttk.Label(nav_frame, text="0/0")
        self.image_counter_label.pack(side=tk.LEFT, padx=5)
        
        # Create control panel (right side of top frame)
        control_panel = ttk.LabelFrame(top_frame, text="Controls", width=300)
        control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5, expand=False)
        
        # ROI selection
        roi_section = ttk.LabelFrame(control_panel, text="ROI Selection")
        roi_section.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(roi_section, text="Select reading area on image").pack(pady=5)
        
        self.select_roi_button = ttk.Button(roi_section, text="Select ROI", command=self.start_roi_selection)
        self.select_roi_button.pack(fill=tk.X, pady=5)
        
        self.clear_roi_button = ttk.Button(roi_section, text="Clear ROI", command=self.clear_roi)
        self.clear_roi_button.pack(fill=tk.X, pady=5)

        # Parameters section
        params_section = ttk.LabelFrame(control_panel, text="Parameters")
        params_section.pack(fill=tk.X, padx=5, pady=10)

        ttk.Label(params_section, text="Time Interval:").pack(anchor=tk.W, pady=2)
        ttk.Entry(params_section, textvariable=self.time_interval, justify=tk.CENTER).pack(fill=tk.X, pady=2)
        
        ttk.Label(params_section, text="Time Unit:").pack(anchor=tk.W, pady=2)
        unit_combo = ttk.Combobox(params_section, textvariable=self.time_unit, 
                                 values=["μs", "ms", "s", "mins", "hours", "days"], state="readonly")
        unit_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(params_section, text="Start Time:").pack(anchor=tk.W, pady=2)
        ttk.Entry(params_section, textvariable=self.start_time, justify=tk.CENTER).pack(fill=tk.X, pady=2)

        ttk.Label(params_section, text="Reading Unit:").pack(anchor=tk.W, pady=2)
        unit_entry = ttk.Entry(params_section, textvariable=self.reading_unit, justify=tk.CENTER)
        unit_entry.pack(fill=tk.X, pady=2)

        # Pre-processing section
        preproc_section = ttk.LabelFrame(control_panel, text="Pre-processing")
        preproc_section.pack(fill=tk.X, padx=5, pady=10)

        preproc_section.grid_columnconfigure(0, weight=1)
        preproc_section.grid_columnconfigure(1, weight=1)
        preproc_section.grid_rowconfigure(0, weight=1)
        preproc_section.grid_rowconfigure(1, weight=1)
        preproc_section.grid_rowconfigure(2, weight=1)

        ttk.Label(preproc_section, text="Erosion Size:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        erosion_entry = ttk.Entry(preproc_section, textvariable=self.erosion_size, width=5, justify="center")
        erosion_entry.grid(row=0, column=1, pady=2)

        ttk.Label(preproc_section, text="Closing Size:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        closing_entry = ttk.Entry(preproc_section, textvariable=self.closing_size, width=5, justify="center")
        closing_entry.grid(row=1, column=1, pady=2)

        preview_button = ttk.Button(preproc_section, text="Preview Preprocessing", command=self.preview_preprocessing)
        preview_button.grid(row=2, column=0, columnspan=2, pady=2)
        
        # Process button
        process_section = ttk.Frame(control_panel)
        process_section.pack(fill=tk.X, padx=5, pady=10)
        
        # Decimal position setting
        ttk.Label(process_section, text="Decimal Position:").pack(anchor=tk.W, pady=2)
        decimal_combo = ttk.Combobox(process_section, textvariable=self.decimal_position,
                                     values=["Keep", "None", "0.1", "0.01", "0.001", "0.0001"],
                                     state="readonly", width=15)
        decimal_combo.pack(fill=tk.X, pady=2)
        
        self.preview_button = ttk.Button(process_section, text="Preview Recognition", command=self.preview_recognition)
        self.preview_button.pack(fill=tk.X, pady=5)
        
        self.process_button = ttk.Button(process_section, text="Process Images", command=self.start_processing)
        self.process_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(process_section, text="Stop Processing", command=self.stop_processing_command, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        # Create results frame (graph and log)
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create graph panel
        graph_frame = ttk.LabelFrame(results_frame, text="Readings Graph")
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.plot.set_xlabel("Time")
        self.plot.set_ylabel("Reading")
        self.plot.grid(True)
        self.figure.tight_layout()
        
        self.canvas_plot = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create log panel
        log_frame = ttk.LabelFrame(results_frame, text="Log", width=400)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Configure log tags for different colors
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("error", foreground="red")
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-l>', lambda e: self.load_image_sequence())
        self.root.bind('<Control-L>', lambda e: self.load_image_sequence())
        self.root.bind('<Control-e>', lambda e: self.export_to_excel())
        self.root.bind('<Control-E>', lambda e: self.export_to_excel())
        self.root.bind('d', lambda e: self.previous_image())
        self.root.bind('D', lambda e: self.previous_image())
        self.root.bind('f', lambda e: self.next_image())
        self.root.bind('F', lambda e: self.next_image())
        self.root.bind('<Control-z>', lambda e: self.clear_roi())
        self.root.bind('<Control-Z>', lambda e: self.clear_roi())
        self.root.bind('<Control-Return>', lambda e: self.start_processing())
        self.root.bind('<Control-h>', lambda e: self.show_help_window())
        self.root.bind('<Control-H>', lambda e: self.show_help_window())
    
    def log(self, message, level="info"):
        """Add message to the log display with color based on level"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        # Define colors for different log levels
        if level == "info":
            tag = "info"
        elif level == "warning":
            tag = "warning"
        elif level == "error":
            tag = "error"
        else:
            tag = "info"
        
        self.log_text.insert(tk.END, full_message, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def adjust_decimal_position(self, reading_str):
        """
        Adjust decimal position based on user selection
        
        Args:
            reading_str: Original recognized reading string (e.g., "100.00")
        
        Returns:
            str: Adjusted reading string based on decimal position setting
        """
        if not reading_str:
            return reading_str
        
        decimal_setting = self.decimal_position.get()
        
        # Keep: return original result
        if decimal_setting == "Keep":
            return reading_str
        
        # None: remove decimal point
        if decimal_setting == "None":
            return reading_str.replace(".", "")
        
        # Parse the decimal precision value (e.g., "0.1" -> 1 decimal place)
        try:
            precision_value = float(decimal_setting)
            # Calculate number of decimal places
            decimal_places = len(decimal_setting.split('.')[-1]) if '.' in decimal_setting else 0
            
            # Convert reading string to float
            try:
                reading_value = float(reading_str)
            except ValueError:
                # If conversion fails, return original
                return reading_str
            
            # Determine current decimal places in reading_str
            if '.' in reading_str:
                current_decimal_places = len(reading_str.split('.')[-1])
            else:
                current_decimal_places = 0
            
            # Calculate adjustment: move decimal point to match target precision
            # Example: "100.00" (value=100.0, 2 decimal places) -> "0.001" (3 decimal places)
            # We need: 100.0 * (0.001 / 0.01) = 100.0 * 0.1 = 10.0 -> "10.001"
            # Actually, the logic is: multiply by precision_value and adjust decimal places
            
            # Get the implied precision from the reading string
            if '.' in reading_str:
                implied_precision = 10 ** (-current_decimal_places)
            else:
                implied_precision = 1.0
            
            # Adjust the value: multiply by precision_value and divide by implied precision
            adjusted_value = reading_value * (precision_value / implied_precision)
            
            # Format to target decimal places
            formatted_str = f"{adjusted_value:.{decimal_places}f}"
            
            return formatted_str
            
        except (ValueError, ZeroDivisionError):
            # If parsing fails, return original
            return reading_str
    
    def load_image_sequence(self):
        """Load a sequence of images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Image Sequence")
        if not folder_path:
            return
            
        self.image_folder = folder_path
        self.log(f"Loading images from: {folder_path}", "info")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        self.image_files = [f for f in sorted(os.listdir(folder_path)) 
                           if os.path.isfile(os.path.join(folder_path, f)) and 
                           f.lower().endswith(image_extensions)]
        
        self.total_images = len(self.image_files)
        
        if self.total_images == 0:
            messagebox.showwarning("No Images", "No image files found in the selected folder.")
            self.log("No image files found.", "warning")
            return
        
        # Update image slider
        self.image_slider.configure(from_=0, to=self.total_images-1)
        
        self.log(f"Found {self.total_images} images", "info")
        self.current_image_index = 0
        self.show_current_image()
        
        self.readings = []
        self.confidences = []
        self.update_plot()
        self.roi_selected = False
        self.file_menu.entryconfig("Export to Excel", state=tk.DISABLED)
    
    def show_current_image(self):
        """Display the current image on the canvas"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        self.image_counter_label.config(text=f"{self.current_image_index + 1}/{self.total_images}")
        
        image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
        self.display_image(image_path)
    
    def on_slider_change(self, value):
        """Handle image slider value change"""
        if not self.image_files:
            return
        
        # Convert slider value to integer
        index = int(float(value))
        if index != self.current_image_index:
            self.current_image_index = index
            self.show_current_image()
    
    def previous_image(self):
        """Navigate to previous image"""
        if not self.image_files:
            return
        
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_slider.set(self.current_image_index)
            self.show_current_image()
    
    def next_image(self):
        """Navigate to next image"""
        if not self.image_files:
            return
        
        if self.current_image_index < self.total_images - 1:
            self.current_image_index += 1
            self.image_slider.set(self.current_image_index)
            self.show_current_image()

    def resize_image(self, event):
        """Automatically resize the image and ROI rectangle to fit the Canvas"""
        if not hasattr(self, 'original_img'):
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Get original image dimensions
        orig_width, orig_height = self.original_img.size

        # Calculate aspect ratios
        aspect_ratio = orig_width / orig_height
        canvas_ratio = canvas_width / canvas_height

        # Scale to fit while preserving aspect ratio
        if aspect_ratio > canvas_ratio:
            # Image is wider, fit to width
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        else:
            # Image is taller or equal, fit to height
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)

        self.current_img = self.original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(self.current_img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                 image=self.current_photo, anchor=tk.CENTER, tags="image")

        self.display_width = new_width
        self.display_height = new_height

        if self.roi_selected:
            self.draw_roi_rectangle()
    
    def display_image(self, image_path):
        """Display an image on the canvas"""
        try:
            img = Image.open(image_path)
            self.original_img = img.copy()

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400
           
            # Get original image dimensions
            orig_width, orig_height = img.size

            # Calculate aspect ratios
            aspect_ratio = orig_width / orig_height
            canvas_ratio = canvas_width / canvas_height

            # Scale to fit while preserving aspect ratio
            if aspect_ratio > canvas_ratio:
                # Image is wider, fit to width
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller or equal, fit to height
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            self.current_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(self.current_img)
            
            self.display_width = new_width
            self.display_height = new_height
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                    image=self.current_photo, anchor=tk.CENTER, tags="image")
            
            if self.roi_selected:
                self.draw_roi_rectangle()
                
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}", "error")
    
    def on_mouse_down(self, event):
        """Handle mouse button press for ROI selection"""
        if not self.roi_selecting:
            return
        
        self.roi_start_x = event.x
        self.roi_start_y = event.y
        
        self.canvas.delete("roi")
    
    def on_mouse_move(self, event):
        """Handle mouse drag for ROI selection"""
        if not self.roi_selecting or self.roi_start_x is None:
            return
        
        self.canvas.delete("roi_temp")
        
        self.canvas.create_rectangle(self.roi_start_x, self.roi_start_y, 
                                    event.x, event.y, 
                                    outline="red", tags="roi_temp")
    
    def on_mouse_up(self, event):
        """Handle mouse button release for ROI selection"""
        if not self.roi_selecting or self.roi_start_x is None:
            return

        self.roi_end_x = event.x
        self.roi_end_y = event.y

        self.canvas.delete("roi_temp")

        if abs(self.roi_end_x - self.roi_start_x) > 10 and abs(self.roi_end_y - self.roi_start_y) > 10:
            # Calculate image offset in canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_x = (canvas_width - self.display_width) // 2
            img_y = (canvas_height - self.display_height) // 2
            
            # Convert canvas coordinates to image relative coordinates
            roi_start_x_relative = self.roi_start_x - img_x
            roi_start_y_relative = self.roi_start_y - img_y
            roi_end_x_relative = self.roi_end_x - img_x
            roi_end_y_relative = self.roi_end_y - img_y
            
            # Calculate scaling ratios
            x_scale = self.original_img.width / self.display_width
            y_scale = self.original_img.height / self.display_height

            # Convert relative coordinates to original image coordinates
            self.orig_roi_start_x = int(min(roi_start_x_relative, roi_end_x_relative) * x_scale)
            self.orig_roi_start_y = int(min(roi_start_y_relative, roi_end_y_relative) * y_scale)
            self.orig_roi_end_x = int(max(roi_start_x_relative, roi_end_x_relative) * x_scale)
            self.orig_roi_end_y = int(max(roi_start_y_relative, roi_end_y_relative) * y_scale)
            
            # Boundary check: ensure coordinates are within original image bounds
            self.orig_roi_start_x = max(0, min(self.orig_roi_start_x, self.original_img.width))
            self.orig_roi_start_y = max(0, min(self.orig_roi_start_y, self.original_img.height))
            self.orig_roi_end_x = max(0, min(self.orig_roi_end_x, self.original_img.width))
            self.orig_roi_end_y = max(0, min(self.orig_roi_end_y, self.original_img.height))

            self.roi_selected = True
            self.draw_roi_rectangle()
            self.roi_selecting = False
            self.select_roi_button.config(state=tk.NORMAL)
            self.log(f"ROI selected: ({self.orig_roi_start_x}, {self.orig_roi_start_y}) to ({self.orig_roi_end_x}, {self.orig_roi_end_y})", "info")
        else:
            self.log("ROI too small. Please select a larger area.", "warning")
            self.roi_selecting = False
            self.select_roi_button.config(state=tk.NORMAL)
    
    def draw_roi_rectangle(self):
        """Draw the ROI rectangle on the canvas"""
        if not self.roi_selected:
            return

        self.canvas.delete("roi")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_x = (canvas_width - self.display_width) // 2
        img_y = (canvas_height - self.display_height) // 2

        x_scale = self.display_width / self.original_img.width
        y_scale = self.display_height / self.original_img.height

        start_x = img_x + int(self.orig_roi_start_x * x_scale)
        start_y = img_y + int(self.orig_roi_start_y * y_scale)
        end_x = img_x + int(self.orig_roi_end_x * x_scale)
        end_y = img_y + int(self.orig_roi_end_y * y_scale)

        self.canvas.create_rectangle(start_x, start_y, end_x, end_y, 
                                     outline="green", width=2, tags="roi")
    
    def start_roi_selection(self):
        """Start ROI selection process on first image"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load an image sequence first.")
            return
            
        self.current_image_index = 0
        self.show_current_image()
        
        self.roi_selecting = True
        self.select_roi_button.config(state=tk.DISABLED)
        self.log("Please select the reading area on the first image.", "info")
    
    def clear_roi(self):
        """Clear the ROI selection"""
        self.roi_selected = False
        self.canvas.delete("roi")
        self.log("ROI cleared", "info")
    
    def start_processing(self):
        """Start processing the image sequence in a separate thread"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load an image sequence first.")
            return
            
        if not self.roi_selected:
            messagebox.showwarning("ROI Required", "Please select the reading area first.")
            return
        
        try:
            time_interval = float(self.time_interval.get())
            if time_interval <= 0:
                raise ValueError("Time interval must be positive")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid time interval.")
            return
            
        try:
            start_time = self.start_time.get()
            if start_time < 0:
                raise ValueError("Start time must be positive")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid start time.")
            return
            
        self.readings = []
        self.time_values = []
        self.confidences = []
        
        self.stop_processing = False
        self.process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing_command(self):
        """Stop the image processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing = True
            self.log("Stopping processing...", "warning")
            self.stop_button.config(state=tk.DISABLED)
    
    def process_images(self):
        """Process all images to extract readings (run in a separate thread)"""
        try:
            unit = self.time_unit.get()
            start_val = self.start_time.get()
            
            self.log("Starting image processing...", "info")
            self.time_values = []
            self.readings = []
            self.confidences = []
            
            for i, img_file in enumerate(self.image_files):
                if self.stop_processing:
                    break

                # Update the image slider in the main thread
                self.root.after(0, lambda idx=i: self.image_slider.set(idx))
                
                time_value = start_val + i * self.time_interval.get()
                self.time_values.append(time_value)
                
                img_path = os.path.join(self.image_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                roi = img[self.orig_roi_start_y:self.orig_roi_end_y,
                          self.orig_roi_start_x:self.orig_roi_end_x]
                
                # Set preprocessing parameters
                self.image_processor.set_preprocessing_params(
                    erosion_size=self.erosion_size.get(),
                    closing_size=self.closing_size.get()
                )
                
                # Set closing_size flag for digit recognizer
                self.digit_recognizer.set_closing_size(self.closing_size.get())
                
                # Use ImageProcessor for preprocessing
                thresh = self.image_processor.preprocess_roi(roi)

                try:
                    # Recognize using PaddleOCR SVTR_Tiny with detailed logging
                    def recognition_log(message, level='info'):
                        """Log recognition details with image index"""
                        self.log(f"Image {i+1}: {message}", level)
                    
                    reading_str = self.digit_recognizer.recognize(thresh, log_callback=recognition_log)
                    
                    # Apply decimal position adjustment
                    if reading_str:
                        reading_str = self.adjust_decimal_position(reading_str)
                    
                    if reading_str and reading_str != "":
                        try:
                            reading = float(reading_str)
                            self.readings.append(reading)
                            self.confidences.append(self.digit_recognizer.last_confidence)
                            self.log(f"Image {i+1}: {time_value:.2f} {unit} → {reading} {self.reading_unit.get()}", "info")
                        except ValueError:
                            self.readings.append(float('nan'))
                            self.confidences.append(float('nan'))
                            self.log(f"Image {i+1}: Invalid reading '{reading_str}'", "warning")
                    else:
                        self.readings.append(float('nan'))
                        self.confidences.append(float('nan'))
                        # Detailed error message is already logged by recognition_log callback
                        self.log(f"Image {i+1}: No valid reading detected", "warning")
                except Exception as e:
                    self.readings.append(float('nan'))
                    self.confidences.append(float('nan'))
                    self.log(f"Recognition error on image {i+1}: {str(e)}", "error")
                
                self.update_plot()
                
                self.root.after(0, self.display_image, os.path.join(self.image_folder, img_file))
                
            self.root.after(0, lambda: self.file_menu.entryconfig("Export to Excel", state=tk.NORMAL if self.readings else tk.DISABLED))
            self.log("Processing completed!" if not self.stop_processing else "Processing stopped.", "info")

        except Exception as e:
            self.log(f"Processing error: {str(e)}", "error")
        finally:
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

    def preview_preprocessing(self):
        """Preview the preprocessing results on the current image"""
        if not self.image_files or not self.roi_selected:
            messagebox.showwarning("Warning", "Please load an image sequence and select ROI first.")
            return
            
        img_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
        img = cv2.imread(img_path)
        if img is None:
            return
            
        roi = img[self.orig_roi_start_y:self.orig_roi_end_y,
                  self.orig_roi_start_x:self.orig_roi_end_x]
        
        # Set preprocessing parameters
        self.image_processor.set_preprocessing_params(
            erosion_size=self.erosion_size.get(),
            closing_size=self.closing_size.get()
        )
        
        # Set closing_size flag for digit recognizer (needed for spacing expansion)
        self.digit_recognizer.set_closing_size(self.closing_size.get())
        
        # Use ImageProcessor for preprocessing
        thresh = self.image_processor.preprocess_roi(roi)
        
        # Apply spacing expansion if closing is enabled (same as in OCR preprocessing)
        # Only process if image is grayscale/binary (not color)
        if self.closing_size.get() > 0 and len(thresh.shape) == 2:
            thresh = self.digit_recognizer.preprocess_for_ocr(thresh)
            
        cv2.imshow("Pre-Processing Preview", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def preview_recognition(self):
        """Preview recognition result on the current image with annotation"""
        try:
            # Check preconditions
            if not self.image_files:
                messagebox.showwarning("Warning", "Please load an image sequence first.")
                return
            
            if not self.roi_selected:
                messagebox.showwarning("Warning", "Please select ROI first.")
                return
            
            # Get current image path
            img_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                self.log("Error: Failed to read image", "error")
                return
            
            # Extract ROI region
            roi = img[self.orig_roi_start_y:self.orig_roi_end_y,
                      self.orig_roi_start_x:self.orig_roi_end_x]
            
            # Set preprocessing parameters
            self.image_processor.set_preprocessing_params(
                erosion_size=self.erosion_size.get(),
                closing_size=self.closing_size.get()
            )
            
            # Set closing_size flag for digit recognizer
            self.digit_recognizer.set_closing_size(self.closing_size.get())
            
            # Use ImageProcessor for preprocessing
            thresh = self.image_processor.preprocess_roi(roi)
            
            # Get final OCR input image (with spacing, scaling, border)
            ocr_img = self.digit_recognizer.preprocess_for_ocr(thresh)
            
            # Create temporary file for OCR result image
            temp_result_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_result_img_path = temp_result_img.name
            temp_result_img.close()
            
            # OCR recognition with detailed logging and result image saving
            def recognition_log(message, level='info'):
                """Log recognition details for preview"""
                self.log(f"Preview Recognition: {message}", level)
            
            reading_str = self.digit_recognizer.recognize(
                thresh, 
                log_callback=recognition_log,
                result_img_path=temp_result_img_path
            )
            
            # Apply decimal position adjustment
            if reading_str:
                reading_str = self.adjust_decimal_position(reading_str)
            
            # Get recognition confidence
            recognition_confidence = self.digit_recognizer.last_confidence
            
            # Copy original image for annotation
            display_img = img.copy()
            
            # Get ROI dimensions
            roi_height = self.orig_roi_end_y - self.orig_roi_start_y
            roi_width = self.orig_roi_end_x - self.orig_roi_start_x
            
            # Draw ROI rectangle (green)
            cv2.rectangle(display_img, 
                         (self.orig_roi_start_x, self.orig_roi_start_y),
                         (self.orig_roi_end_x, self.orig_roi_end_y),
                         (0, 255, 0), 2)
            
            # Use preprocessed OCR image (without text annotations)
            # Convert OCR image to BGR for display
            # Check if image is grayscale (2D) or color (3D)
            if len(ocr_img.shape) == 2:
                # Grayscale image: convert to BGR
                ocr_img_bgr = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR)
            elif len(ocr_img.shape) == 3:
                # Color image: already BGR format, use as-is
                ocr_img_bgr = ocr_img
            else:
                # Fallback: convert to BGR
                ocr_img_bgr = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR)
            
            # Get actual OCR image dimensions (no resizing)
            ocr_img_height, ocr_img_width = ocr_img_bgr.shape[:2]
            
            # Calculate position for OCR image (below original ROI)
            preview_y_start = self.orig_roi_end_y + 5
            preview_y_end = preview_y_start + ocr_img_height
            preview_x_start = self.orig_roi_start_x
            preview_x_end = self.orig_roi_start_x + ocr_img_width
            
            # Check if preview fits within image bounds
            img_height, img_width = display_img.shape[:2]
            
            if preview_y_end <= img_height and preview_x_end <= img_width:
                # Place OCR image directly on the image
                display_img[preview_y_start:preview_y_end,
                           preview_x_start:preview_x_end] = ocr_img_bgr
                
                # Draw rectangle around OCR image (blue)
                cv2.rectangle(display_img,
                             (preview_x_start, preview_y_start),
                             (preview_x_end, preview_y_end),
                             (255, 0, 0), 2)
            else:
                # Expand image if needed (vertically or horizontally)
                extra_height = max(0, preview_y_end - img_height)
                extra_width = max(0, preview_x_end - img_width)
                new_height = img_height + extra_height
                new_width = img_width + extra_width
                
                expanded_img = np.zeros((new_height, new_width, 3),
                                       dtype=np.uint8)
                expanded_img[0:img_height, 0:img_width] = display_img
                expanded_img[preview_y_start:preview_y_end,
                            preview_x_start:preview_x_end] = ocr_img_bgr
                cv2.rectangle(expanded_img,
                             (preview_x_start, preview_y_start),
                             (preview_x_end, preview_y_end),
                             (255, 0, 0), 2)
                display_img = expanded_img
            
            # Clean up temporary result image file
            try:
                if os.path.exists(temp_result_img_path):
                    os.unlink(temp_result_img_path)
            except:
                pass
            
            # Convert to PIL RGB for canvas display
            pil_display_img = Image.fromarray(
                cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            )
            
            # Draw text annotation above ROI region
            draw = ImageDraw.Draw(pil_display_img)
            
            # Calculate font size: quarter of ROI height
            font_size = int(roi_height / 4)
            
            # Load Arial Bold font
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)
            except:
                try:
                    # Try alternative Arial Bold path
                    font = ImageFont.truetype("Arial Bold.ttf", font_size)
                except:
                    try:
                        # Try system font path on Windows
                        font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arialbd.ttf')
                        font = ImageFont.truetype(font_path, font_size)
                    except:
                        # Fallback to default font
                        font = ImageFont.load_default()
                        self.log("Preview Recognition: Arial Bold font not found, using default font", "warning")
            
            # Prepare annotation text and color
            if reading_str and reading_str != "":
                # Success: green text with recognition result
                annotation_text = f"BEST: '{reading_str}' (conf={recognition_confidence:.3f})"
                text_color = (0, 255, 0)  # Green in RGB
            else:
                # Failure: red text
                annotation_text = "No valid reading"
                text_color = (255, 0, 0)  # Red in RGB
            
            # Calculate text position (above ROI, centered horizontally)
            bbox = draw.textbbox((0, 0), annotation_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text above ROI region (with 20 pixel margin)
            text_x = self.orig_roi_start_x + (roi_width - text_width) // 2
            text_y = self.orig_roi_start_y - text_height - 20
            
            # Ensure text is within image bounds
            text_x = max(0, min(text_x, pil_display_img.width - text_width))
            text_y = max(0, text_y)
            
            # Draw text with solid color (no outline)
            draw.text((text_x, text_y), annotation_text, font=font, fill=text_color)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400
            
            # Get image dimensions
            img_width, img_height = pil_display_img.size
            
            # Calculate aspect ratios
            aspect_ratio = img_width / img_height
            canvas_ratio = canvas_width / canvas_height
            
            # Scale to fit while preserving aspect ratio
            if aspect_ratio > canvas_ratio:
                # Image is wider, fit to width
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller or equal, fit to height
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            
            # Resize image to fit canvas
            resized_img = pil_display_img.resize((new_width, new_height),
                                                 Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_img)
            
            # Store reference to prevent garbage collection
            self.preview_photo = photo
            
            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                    image=photo, anchor=tk.CENTER, 
                                    tags="preview")
            
            # Update original_img and dimensions for consistency
            self.original_img = pil_display_img
            self.display_width = new_width
            self.display_height = new_height
            
            # Redraw ROI if selected
            if self.roi_selected:
                self.draw_roi_rectangle()
            
            # Log final result (detailed diagnostics already logged by recognition_log callback)
            if reading_str and reading_str != "":
                self.log(f"Preview Recognition: Final result - {reading_str} {self.reading_unit.get()}", "info")
            else:
                # Detailed error message already logged by recognition_log callback
                pass
                
        except Exception as e:
            self.log(f"Preview Recognition Error: {str(e)}", "error")
    
    def update_plot(self):
        """Update the matplotlib plot with current readings"""
        if not self.readings:
            return
            
        self.root.after(0, self._update_plot_main_thread)
    
    def _update_plot_main_thread(self):
        """Update plot (called in main thread)"""
        try:
            self.plot.clear()
            if self.time_values and self.readings:
                unit = self.time_unit.get()
                self.plot.plot(self.time_values, self.readings, 'bo-')
                self.plot.set_xlabel(f'Time ({unit})')
                self.plot.set_ylabel(f'Reading ({self.reading_unit.get()})')
                self.plot.grid(True)
                self.figure.tight_layout()
                self.canvas_plot.draw()
        except Exception as e:
            self.log(f"Error updating plot: {str(e)}", "error")
    
    def export_to_excel(self):
        """Export the readings data to an Excel file"""
        if not self.readings or not self.time_values:
            messagebox.showwarning("No Data", "No data available to export.")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Readings Data"
            )
            
            if not file_path:
                return
                 
            unit = self.time_unit.get()
            df = pd.DataFrame({
                f'Time ({unit})': self.time_values,
                f'Reading ({self.reading_unit.get()})': self.readings,
                'Confidence': self.confidences
            })
            
            df.to_excel(file_path, index=False, sheet_name='Device Readings')
            
            self.log(f"Data exported successfully to {file_path}", "info")
            messagebox.showinfo("Export Successful", f"Data exported successfully to {file_path}")
            
        except Exception as e:
            self.log(f"Error exporting data: {str(e)}", "error")
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
    
    def show_help_window(self):
        """Display help window with bilingual user guide"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Device Readings Analyzer - Help")
        help_window.geometry("900x650")
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Center the window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (help_window.winfo_screenheight() // 2) - (650 // 2)
        help_window.geometry(f"900x650+{x}+{y}")
        
        # Language switch buttons frame
        lang_frame = ttk.Frame(help_window)
        lang_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(lang_frame, text="Language / 语言：", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Create text widget first
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        help_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Consolas", 9))
        help_text.pack(fill=tk.BOTH, expand=True)
        
        # Language buttons
        btn_cn = ttk.Button(lang_frame, text="中文", width=10)
        btn_cn.pack(side=tk.LEFT, padx=2)
        
        btn_en = ttk.Button(lang_frame, text="English", width=10)
        btn_en.pack(side=tk.LEFT, padx=2)
        
        # Configure button commands
        btn_cn.config(command=lambda: self.switch_help_language('cn', help_text, btn_cn, btn_en))
        btn_en.config(command=lambda: self.switch_help_language('en', help_text, btn_cn, btn_en))
        
        # Close button
        close_frame = ttk.Frame(help_window)
        close_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(close_frame, text="Close / 关闭", command=help_window.destroy, width=15).pack()
        
        # Load Chinese content by default
        self.switch_help_language('cn', help_text, btn_cn, btn_en)
    
    def switch_help_language(self, language, text_widget, btn_cn, btn_en):
        """Switch help content language"""
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        
        if language == 'cn':
            text_widget.insert(1.0, self.get_help_content_cn())
            btn_cn.config(state=tk.DISABLED)
            btn_en.config(state=tk.NORMAL)
        else:
            text_widget.insert(1.0, self.get_help_content_en())
            btn_en.config(state=tk.DISABLED)
            btn_cn.config(state=tk.NORMAL)
        
        text_widget.config(state=tk.DISABLED)
    
    def show_about_dialog(self):
        """Display About dialog"""
        about_text = f"""Device Readings Analyzer

Version: {self.version}
Author: {self.author}
Email: {self.email}
License: {self.license}

Copyright © 2025 {self.author}
All rights reserved."""
        
        messagebox.showinfo("About", about_text)
    
    def get_help_content_cn(self):
        """Return Chinese help content"""
        try:
            # Get resource file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            help_file = os.path.join(current_dir, 'resources', 'help_content_cn.txt')
            
            # Read file content with UTF-8 encoding
            with open(help_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # Fallback: return error message
            return f"""Error loading Chinese help content.

错误：无法加载中文帮助内容。
错误信息：{str(e)}

请确保 src/resources/help_content_cn.txt 文件存在。"""
    
    def get_help_content_en(self):
        """Return English help content"""
        try:
            # Get resource file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            help_file = os.path.join(current_dir, 'resources', 'help_content_en.txt')
            
            # Read file content with UTF-8 encoding
            with open(help_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # Fallback: return error message
            return f"""Error loading English help content.

Error message: {str(e)}

Please ensure src/resources/help_content_en.txt file exists."""


# Export the class
__all__ = ['DeviceReadingsAnalyzer']
