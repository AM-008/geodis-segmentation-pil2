import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import os


class ImageSegmentationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        self.root.geometry("1000x700")
        
        self.image = None
        self.image_path = None
        self.photo_image = None
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        self.mask = None
        self.mask_photo_image = None
        self.mask_opacity = 0.3  # 30% opacity for the mask
        
        self.blue_dots = []  # Areas of interest
        self.red_dots = []   # Background areas
        self.dot_radius = 5  # Radius of drawn dots
        self.drawing_mode = None  # 'blue', 'red', or None
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        """Create the main UI widgets for image import and display."""
        # Frame for image path input
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(input_frame, text="Image Path:").pack(side=tk.LEFT, padx=5)
        self.path_entry = tk.Entry(input_frame, width=60)
        self.path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        load_button = tk.Button(input_frame, text="Load Image", command=self.load_image)
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Frame for zoom and display controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Fit to Window", command=self.fit_to_window).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Original Size", command=self.original_size).pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = tk.Label(control_frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=15)
        
        # Frame for mask controls
        mask_frame = tk.Frame(self.root)
        mask_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(mask_frame, text="Mask Opacity:").pack(side=tk.LEFT, padx=5)
        self.opacity_slider = tk.Scale(mask_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                        command=self.on_opacity_change, length=200)
        self.opacity_slider.set(int(self.mask_opacity * 100))
        self.opacity_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.clear_mask_button = tk.Button(mask_frame, text="Clear Mask", command=self.clear_mask)
        self.clear_mask_button.pack(side=tk.LEFT, padx=5)
        
        # Frame for drawing mode controls
        draw_frame = tk.Frame(self.root)
        draw_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(draw_frame, text="Drawing Mode:").pack(side=tk.LEFT, padx=5)
        
        self.blue_button = tk.Button(draw_frame, text="Blue Dots (Interest)", 
                                      command=self.set_drawing_mode_blue, bg="lightblue")
        self.blue_button.pack(side=tk.LEFT, padx=5)
        
        self.red_button = tk.Button(draw_frame, text="Red Dots (Background)", 
                                     command=self.set_drawing_mode_red, bg="lightcoral")
        self.red_button.pack(side=tk.LEFT, padx=5)
        
        self.off_button = tk.Button(draw_frame, text="Turn Off Drawing", 
                                     command=self.set_drawing_mode_off)
        self.off_button.pack(side=tk.LEFT, padx=5)
        
        tk.Label(draw_frame, text="Dot Size:").pack(side=tk.LEFT, padx=5, fill=tk.X, expand=False)
        self.dot_size_slider = tk.Scale(draw_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                                         command=self.on_dot_size_change, length=100)
        self.dot_size_slider.set(self.dot_radius)
        self.dot_size_slider.pack(side=tk.LEFT, padx=5)
        
        self.clear_dots_button = tk.Button(draw_frame, text="Clear All Dots", command=self.clear_all_dots)
        self.clear_dots_button.pack(side=tk.LEFT, padx=5)
        
        # Info label for drawing status
        self.drawing_info_label = tk.Label(draw_frame, text="Mode: OFF", fg="gray")
        self.drawing_info_label.pack(side=tk.LEFT, padx=15)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.root, bg="gray30", cursor="cross")
        self.canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.root.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind("<Button-1>", self.on_canvas_click)  # Left mouse button click
    
    def load_image(self):
        """Load an image from the provided absolute path."""
        path = self.path_entry.get().strip()
        
        if not path:
            messagebox.showerror("Error", "Please enter an image path.")
            return
        
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File not found: {path}")
            return
        
        try:
            self.image = Image.open(path)
            self.image_path = path
            self.zoom_level = 1.0
            self.create_mask()
            self.display_image()
            messagebox.showinfo("Success", f"Image loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def create_mask(self):
        """Create a transparent mask overlay the same size as the image."""
        if self.image is None:
            return
        
        # Create a transparent RGBA image
        self.mask = Image.new('RGBA', self.image.size, (255, 255, 255, 0))
    
    def display_image(self):
        """Display the image and mask overlay on the canvas with current zoom level."""
        if self.image is None:
            return
        
        # Calculate new dimensions based on zoom level
        new_width = int(self.image.width * self.zoom_level)
        new_height = int(self.image.height * self.zoom_level)
        
        # Resize the image
        resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        self.photo_image = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        canvas_center_x = self.canvas.winfo_width() // 2
        canvas_center_y = self.canvas.winfo_height() // 2
        self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.photo_image)
        
        # Resize and display mask overlay with opacity
        if self.mask is not None:
            resized_mask = self.mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Apply opacity to the mask
            mask_with_opacity = self.apply_mask_opacity(resized_mask)
            self.mask_photo_image = ImageTk.PhotoImage(mask_with_opacity)
            self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.mask_photo_image)
        
        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
    
    def apply_mask_opacity(self, mask_image):
        """Apply opacity to the mask by adjusting alpha channel."""
        if mask_image.mode != 'RGBA':
            mask_image = mask_image.convert('RGBA')
        
        # Split the image into channels
        r, g, b, a = mask_image.split()
        
        # Apply opacity to the alpha channel
        a = a.point(lambda x: int(x * self.mask_opacity))
        
        # Merge channels back together
        return Image.merge('RGBA', (r, g, b, a))
    
    def zoom_in(self):
        """Increase zoom level."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        self.zoom_level = min(self.zoom_level + 0.1, self.max_zoom)
        self.display_image()
    
    def zoom_out(self):
        """Decrease zoom level."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        self.zoom_level = max(self.zoom_level - 0.1, self.min_zoom)
        self.display_image()
    
    def fit_to_window(self):
        """Fit image to window size."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            messagebox.showwarning("Warning", "Canvas not ready. Try again.")
            return
        
        # Calculate zoom to fit image in canvas
        zoom_x = canvas_width / self.image.width
        zoom_y = canvas_height / self.image.height
        self.zoom_level = min(zoom_x, zoom_y) * 0.95  # 95% to leave margin
        self.display_image()
    
    def original_size(self):
        """Reset to original image size (zoom = 1.0)."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        self.zoom_level = 1.0
        self.display_image()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling for zoom."""
        if self.image is None:
            return
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            self.zoom_out()
        elif event.num == 4 or event.delta > 0:
            self.zoom_in()
    
    def on_opacity_change(self, value):
        """Update mask opacity based on slider value."""
        self.mask_opacity = int(value) / 100.0
        self.display_image()
    
    def clear_mask(self):
        """Clear the mask overlay."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        self.create_mask()
        self.display_image()
        messagebox.showinfo("Info", "Mask cleared.")
    
    def set_drawing_mode_blue(self):
        """Set drawing mode to blue dots (areas of interest)."""
        self.drawing_mode = 'blue'
        self.drawing_info_label.config(text="Mode: BLUE (Interest)", fg="blue")
        self.blue_button.config(relief=tk.SUNKEN)
        self.red_button.config(relief=tk.RAISED)
        self.off_button.config(relief=tk.RAISED)
    
    def set_drawing_mode_red(self):
        """Set drawing mode to red dots (background)."""
        self.drawing_mode = 'red'
        self.drawing_info_label.config(text="Mode: RED (Background)", fg="red")
        self.blue_button.config(relief=tk.RAISED)
        self.red_button.config(relief=tk.SUNKEN)
        self.off_button.config(relief=tk.RAISED)
    
    def set_drawing_mode_off(self):
        """Turn off drawing mode."""
        self.drawing_mode = None
        self.drawing_info_label.config(text="Mode: OFF", fg="gray")
        self.blue_button.config(relief=tk.RAISED)
        self.red_button.config(relief=tk.RAISED)
        self.off_button.config(relief=tk.SUNKEN)
    
    def on_dot_size_change(self, value):
        """Update dot radius based on slider value."""
        self.dot_radius = int(value)
    
    def on_canvas_click(self, event):
        """Handle canvas click to draw dots."""
        if self.image is None or self.drawing_mode is None:
            return
        
        # Get canvas center
        canvas_center_x = self.canvas.winfo_width() // 2
        canvas_center_y = self.canvas.winfo_height() // 2
        
        # Calculate image display dimensions
        display_width = int(self.image.width * self.zoom_level)
        display_height = int(self.image.height * self.zoom_level)
        
        # Calculate the top-left corner of the displayed image
        img_x = canvas_center_x - display_width // 2
        img_y = canvas_center_y - display_height // 2
        
        # Convert canvas click coordinates to image pixel coordinates
        click_x = event.x - img_x
        click_y = event.y - img_y
        
        # Check if click is within the image bounds
        if 0 <= click_x < display_width and 0 <= click_y < display_height:
            # Convert to original image coordinates (undo zoom)
            img_pixel_x = int(click_x / self.zoom_level)
            img_pixel_y = int(click_y / self.zoom_level)
            
            # Add dot to appropriate list
            if self.drawing_mode == 'blue':
                self.blue_dots.append((img_pixel_x, img_pixel_y))
            elif self.drawing_mode == 'red':
                self.red_dots.append((img_pixel_x, img_pixel_y))
            
            # Redraw the mask with the new dot
            self.draw_dots_on_mask()
            self.display_image()
    
    def draw_dots_on_mask(self):
        """Draw all dots on the mask."""
        if self.image is None:
            return
        
        # Start with a fresh transparent mask
        self.create_mask()
        
        # Create a draw object
        draw = ImageDraw.Draw(self.mask)
        
        # Draw blue dots (areas of interest)
        for x, y in self.blue_dots:
            draw.ellipse(
                [x - self.dot_radius, y - self.dot_radius, 
                 x + self.dot_radius, y + self.dot_radius],
                fill=(0, 0, 255, 200)  # Blue with semi-transparency
            )
        
        # Draw red dots (background areas)
        for x, y in self.red_dots:
            draw.ellipse(
                [x - self.dot_radius, y - self.dot_radius, 
                 x + self.dot_radius, y + self.dot_radius],
                fill=(255, 0, 0, 200)  # Red with semi-transparency
            )
    
    def clear_all_dots(self):
        """Clear all drawn dots."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        self.blue_dots.clear()
        self.red_dots.clear()
        self.create_mask()
        self.display_image()
        messagebox.showinfo("Info", "All dots cleared.")


def main():
    root = tk.Tk()
    app = ImageSegmentationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
