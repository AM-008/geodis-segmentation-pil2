# Image Segmentation Tool

A Python-based image segmentation software that allows users to mark areas of interest (blue dots) and background areas (red dots) on images, with plans for automated segmentation processing.

## Project Overview

This tool provides a user-friendly UI built with Tkinter and Pillow for interactive image segmentation. Users can import images, draw annotations, and prepare data for segmentation algorithms.

---

## Development Tutorial

### Step 1: Image Import Functionality

**What was done:**
- Created a Tkinter window with basic UI layout
- Added a text entry field where users can input the absolute path to an image file
- Implemented a "Load Image" button that validates and imports the image using Pillow
- Added error handling for missing files and invalid image formats
- Stored the loaded image for use in subsequent features

**Key Components:**
- `ImageSegmentationUI` class: Main application class that manages the UI and image data
- `create_widgets()`: Sets up the initial UI elements (path entry and load button)
- `load_image()`: Validates the file path and loads the image using PIL
- File existence check: Prevents errors from invalid paths
- User feedback: Displays success/error messages via message boxes

**Code Structure:**
```python
# The app initializes with a window, paths, and widgets
# Users enter an absolute path and click "Load Image"
# The image is stored in self.image for later use
```

**How to Test:**
1. Run `python ui.py`
2. Enter an absolute path to an image file (e.g., `C:\Users\...\photo.jpg`)
3. Click "Load Image"
4. A success message confirms the image is loaded

---

### Step 2: Image Display with Zoom and Resizing

**What was done:**
- Added a Canvas widget to display the loaded image
- Implemented zoom controls with buttons: "Zoom In", "Zoom Out", "Fit to Window", "Original Size"
- Added mouse wheel support for zooming (Windows: MouseWheel, Linux: Button-4/5)
- Display zoom level percentage in real-time
- Image resizes smoothly using Lanczos resampling for better quality
- Canvas adapted to fill the window with a dark gray background

**Key Features:**
- `zoom_level`: Tracks the current zoom percentage (0.1x to 5.0x)
- `display_image()`: Renders the image at the current zoom level on the canvas
- Zoom buttons for precise control
- Mouse wheel support for intuitive zooming
- "Fit to Window" automatically scales the image to fill the available space
- "Original Size" resets zoom to 1.0 (100%)

**How to Test:**
1. Run `python ui.py`
2. Load an image via the path entry field
3. Use buttons or mouse wheel to zoom in/out
4. Try "Fit to Window" to scale to available space
5. Click "Original Size" to reset zoom

---

### Step 3: Transparent Mask Overlay

**What was done:**
- Created a transparent RGBA mask layer that sits on top of the displayed image
- Added an opacity slider (0-100%) to control mask transparency
- Implemented "Clear Mask" button to reset the mask
- The mask is automatically created when an image is loaded
- Mask resizes with the image when zooming

**Key Features:**
- `create_mask()`: Creates a fully transparent RGBA image matching the original image size
- `apply_mask_opacity()`: Adjusts the alpha channel to control mask transparency
- Opacity slider for real-time control of mask visibility
- Mask displays on top of the image without distorting it
- Opacity slider defaults to 30% for good visibility while seeing the image

**Purpose:**
The mask layer serves as a canvas where you'll draw blue and red dots in the next step. It exists separately from the image, allowing for clean drawing while keeping the original image visible.

**How to Test:**
1. Run `python ui.py`
2. Load an image
3. Move the "Mask Opacity" slider to adjust transparency (0% = invisible, 100% = fully opaque)
4. Click "Clear Mask" to reset the mask to fully transparent
5. The mask resizes with the image as you zoom

---

### Step 4: Drawing Functionality (Blue and Red Dots)

**What was done:**
- Added drawing mode toggle buttons for blue and red dots
- Implemented mouse click detection on the canvas to place dots
- Blue dots mark areas of interest for segmentation
- Red dots mark background areas to ignore
- Adjustable dot size with a slider (1-20 pixels)
- Added "Clear All Dots" button to reset all annotations
- Real-time status display showing current drawing mode
- Buttons highlight when active using visual feedback (sunken/raised effect)

**Key Features:**
- **Drawing Modes:**
  - Blue Dots: Click while in "Blue" mode to mark areas of interest
  - Red Dots: Click while in "Red" mode to mark background areas
  - Mode OFF: Disable drawing to prevent accidental clicks
- `on_canvas_click()`: Handles mouse clicks, converts canvas coordinates to image pixel coordinates, accounting for zoom level
- `draw_dots_on_mask()`: Renders all dots on the mask with semi-transparent fill (alpha=200)
- Dot coordinates are stored in `self.blue_dots` and `self.red_dots` lists
- Coordinates automatically adjust when zooming - all stored in original image pixel space

**Coordinate System:**
The script intelligently translates canvas click positions to original image coordinates:
1. Detects click position on canvas
2. Calculates image position on canvas (accounting for zoom and centering)
3. Converts to original image pixel coordinates
4. Stores in lists for later segmentation processing

**How to Test:**
1. Run `python ui.py`
2. Load an image
3. Click "Blue Dots (Interest)" button
4. Click on the image to place blue dots
5. Click "Red Dots (Background)" button
6. Click on the image to place red dots
7. Adjust "Dot Size" slider to change dot appearance
8. Try zooming in/out - dots stay in the same positions
9. Click "Clear All Dots" to reset

**Example Workflow:**
```
1. Load image
2. Set to Blue mode
3. Click on areas you want to segment (interest areas)
4. Set to Red mode
5. Click on background areas to exclude
6. Adjust opacity and dot size for clarity
7. Ready for segmentation processing (Step 5)
```

---

### Step 5: Geodesic Segmentation

**What was done:**
- Created `segmentation.py` module implementing geodesic distance-based image segmentation
- Uses Dijkstra's algorithm to compute weighted shortest paths from seed pixels
- Integrates blue and red dots from the UI as seed markers
- Segments image into foreground (blue) and background (red) regions
- Provides visualization and post-processing capabilities
- Added "Run Segmentation" button to the UI to execute the segmentation algorithm

**Algorithm Overview:**

The geodesic segmentation algorithm works by computing distances that respect image structure rather than using simple Euclidean distance.

1. **Gradient Computation:**
   - Compute image gradient using Sobel operators: $\nabla I = \sqrt{(\frac{\partial I}{\partial x})^2 + (\frac{\partial I}{\partial y})^2}$
   - Smooth image first to reduce noise sensitivity: Gaussian filter with adjustable sigma
   - Normalize gradient to [0, 1]

2. **Edge Weighting:**
   - Define edge weight based on gradient magnitude: $w(p,q) = 1 + 5 \cdot |\nabla I|$
   - High gradient (strong edges) → high weight (expensive path)
   - Low gradient (uniform regions) → low weight (cheap path)
   - This ensures the algorithm follows object boundaries, not crossing structural edges

3. **Distance Map Computation (Dijkstra's Algorithm):**
   - For each seed set (blue foreground, red background), compute geodesic distance map
   - Distance represents "cost" to reach each pixel from the seed set
   - Uses priority queue for efficient computation
   - Solves: $D(p) = \min_{\text{paths } \gamma} \sum w(\gamma)$
   - Results in **distance_blue** (cost to reach from foreground seeds) and **distance_red** (cost from background seeds)

4. **Pixel Label Assignment:**
   - Each pixel is labeled based on minimum distance:
     - If closer to blue seeds: assign to foreground (label = 1)
     - If closer to red seeds: assign to background (label = 0)
   - Pixels equidistant to both are arbitrarily assigned (rarely occurs in practice)

5. **Optional Post-processing:**
   - Morphological operations to clean up segmentation
   - Remove small connected components (noise reduction)
   - Gaussian smoothing for soft boundaries

**Key Features:**
- **8-connectivity:** Considers diagonal neighbors for more natural segmentation
- **Efficiency:** O(n log n) complexity with Dijkstra + priority queue
- **Robustness:** Gradient-based weighting is insensitive to illumination changes
- **Multi-label:** Can be extended to handle multiple regions with multiple seed sets
- **Confidence Map:** Computes confidence of segmentation (difference between distances)

**Integration with UI:**

The UI now includes a "Run Segmentation" button that:
1. Gathers blue_dots (foreground) and red_dots (background) from user annotations
2. Calls `segmentation_from_ui()` function from segmentation.py
3. Executes geodesic segmentation algorithm
4. Creates output image with background removed (keeping only foreground)
5. Displays result and saves visualization

**Output:**
- **Segmented Image:** PNG file with background removed, keeping only the area of interest (blue-marked region)
- **Visualization:** 6-panel diagnostic image showing:
  - Original image
  - Binary segmentation result (blue/red regions)
  - Overlay on original image
  - Distance map from foreground seeds (blue)
  - Distance map from background seeds (red)
  - Confidence map showing certainty of assignments

**How to Use:**
1. Run `python ui.py`
2. Load an image
3. Set to "Blue Dots" mode and click on areas of interest
4. Set to "Red Dots" mode and click on background areas to exclude
5. Click "Run Segmentation" button
6. Wait for processing (displays status messages)
7. Segmented image saved as `segmentation_result.png` (background removed, only foreground)
8. Visualization saved as `segmentation_visualization.png` (6-panel diagnostic view)

**Parameters:**
- `gradient_sigma`: Smoothing applied before gradient computation (default: 1.0). Increase for smoother gradients.
- `connectivity`: Neighborhood type (4 or 8). Default: 8 for better quality.
- Post-processing parameters: `sigma` for smoothing, `min_size` for removing small components.

**Limitations and Notes:**
- Requires at least one seed point (blue or red)
- Performance depends on image size and number of seeds
- Works best with clear boundaries between foreground and background
- Can be extended for multi-label segmentation with more seed colors

---

## Dependencies

```
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
```

Install with:
```bash
pip install pillow numpy scipy
```

---

## File Structure

```
geodis-segmentation-pil2/
├── ui.py                 # Main Tkinter UI application
├── segmentation.py       # Geodesic segmentation algorithm
├── main.py              # Entry point (legacy)
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

---

## Next Steps

- Add more filter options (bilateral filtering, edge detection modes)
- Implement undo/redo for seed placement
- Support for batch processing multiple images
- Add more post-processing filters
- Real-time segmentation preview while drawing seeds
