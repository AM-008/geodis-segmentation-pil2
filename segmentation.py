import numpy as np
from PIL import Image
import heapq
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class GeodesicSegmentation:
    """
    Geodesic segmentation using weighted shortest path distance.
    
    This class implements geodesic distance-based image segmentation where
    the distance metric is weighted by image gradients. Seeds/markers are used
    to initialize foreground (blue) and background (red) regions.
    """
    
    def __init__(self, image, blue_seeds, red_seeds, gradient_sigma=1.0):
        """
        Initialize geodesic segmentation.
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            blue_seeds (list): List of (x, y) coordinates for foreground/interest region
            red_seeds (list): List of (x, y) coordinates for background region
            gradient_sigma (float): Sigma for Gaussian smoothing before gradient computation
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            self.image = np.array(image.convert('L') if image.mode != 'L' else image)
        else:
            self.image = image.astype(np.float32)
        
        # Normalize image to [0, 1]
        if self.image.max() > 1:
            self.image = self.image / 255.0
        
        self.height, self.width = self.image.shape[:2]
        self.blue_seeds = blue_seeds
        self.red_seeds = red_seeds
        self.gradient_sigma = gradient_sigma
        
        # Precompute gradient magnitude
        self.gradient = self._compute_gradient()
        
        # Edge weights
        self.edge_weights = None
    
    def _compute_gradient(self):
        """
        Compute gradient magnitude of the image.
        
        Returns:
            np.ndarray: Normalized gradient magnitude
        """
        # Smooth image first to reduce noise sensitivity
        smoothed = gaussian_filter(self.image, sigma=self.gradient_sigma)
        
        # Compute gradients using Sobel operators
        gy, gx = np.gradient(smoothed)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normalize to [0, 1]
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        return gradient_magnitude
    
    def _compute_edge_weight(self, p1, p2):
        """
        Compute edge weight between two adjacent pixels using gradient.
        
        High gradient (edge) → high weight (expensive path)
        Low gradient (uniform region) → low weight (cheap path)
        
        Args:
            p1 (tuple): First pixel coordinate (y, x)
            p2 (tuple): Second pixel coordinate (y, x)
        
        Returns:
            float: Edge weight
        """
        y1, x1 = p1
        y2, x2 = p2
        
        # Use exponential of gradient to create strong edge penalties
        # w = 1 + exp(alpha * gradient)
        avg_gradient = (self.gradient[y1, x1] + self.gradient[y2, x2]) / 2.0
        
        # Use exponential weighting to penalize high gradients
        # alpha=5 provides good contrast between edges and uniform regions
        weight = 1.0 + 5.0 * avg_gradient
        
        return weight
    
    def _get_neighbors(self, y, x, connectivity=8):
        """
        Get valid neighboring pixels (4 or 8 connectivity).
        
        Args:
            y (int): Y coordinate
            x (int): X coordinate
            connectivity (int): 4 or 8 connectivity
        
        Returns:
            list: List of valid neighbor coordinates
        """
        neighbors = []
        
        if connectivity == 4:
            # 4-connectivity (up, down, left, right)
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            # 8-connectivity (includes diagonals)
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dy, dx in deltas:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                neighbors.append((ny, nx))
        
        return neighbors
    
    def _dijkstra_distance(self, seeds, label):
        """
        Compute geodesic distance map from seed points using Dijkstra's algorithm.
        
        Args:
            seeds (list): List of seed coordinates (x, y)
            label (int): Label for this distance map (0=blue, 1=red)
        
        Returns:
            np.ndarray: Distance map from seed points
        """
        # Initialize distance map with infinity
        distance = np.full((self.height, self.width), np.inf, dtype=np.float32)
        visited = np.zeros((self.height, self.width), dtype=bool)
        
        # Priority queue: (distance, y, x)
        pq = []
        
        # Initialize with seed points
        if not seeds:
            return distance
        
        for x, y in seeds:
            # Ensure coordinates are within bounds
            if 0 <= y < self.height and 0 <= x < self.width:
                distance[y, x] = 0.0
                heapq.heappush(pq, (0.0, y, x))
        
        # Dijkstra's algorithm
        while pq:
            curr_dist, y, x = heapq.heappop(pq)
            
            # Skip if already visited
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            
            # Process neighbors
            for ny, nx in self._get_neighbors(y, x, connectivity=8):
                if not visited[ny, nx]:
                    # Compute edge weight based on gradient
                    weight = self._compute_edge_weight((y, x), (ny, nx))
                    
                    # Calculate new distance
                    new_dist = curr_dist + weight
                    
                    # Update if shorter path found
                    if new_dist < distance[ny, nx]:
                        distance[ny, nx] = new_dist
                        heapq.heappush(pq, (new_dist, ny, nx))
        
        return distance
    
    def segment(self, connectivity=8):
        """
        Perform geodesic segmentation.
        
        Computes geodesic distance from blue seeds and red seeds,
        then assigns each pixel to the nearest seed set.
        
        Args:
            connectivity (int): 4 or 8 connectivity (default: 8)
        
        Returns:
            dict: Dictionary with segmentation results:
                - 'label_map': Label map (0=background/red, 1=foreground/blue, 2=unlabeled)
                - 'distance_blue': Distance map from blue seeds
                - 'distance_red': Distance map from red seeds
                - 'confidence': Confidence map (difference between distances)
        """
        if not self.blue_seeds and not self.red_seeds:
            raise ValueError("At least one seed point required (blue or red)")
        
        # Compute distance maps from each seed set
        print("Computing geodesic distance from blue seeds (foreground)...")
        distance_blue = self._dijkstra_distance(self.blue_seeds, label=0)
        
        print("Computing geodesic distance from red seeds (background)...")
        distance_red = self._dijkstra_distance(self.red_seeds, label=1)
        
        # Initialize label map
        label_map = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Assign labels based on minimum distance
        # Handle pixels with valid distances from both seed sets
        valid_blue = ~np.isinf(distance_blue)
        valid_red = ~np.isinf(distance_red)
        
        # Pixels with both valid distances: assign based on minimum
        both_valid = valid_blue & valid_red
        label_map[both_valid] = (distance_blue[both_valid] < distance_red[both_valid]).astype(np.uint8)
        
        # Pixels with only blue distance: foreground
        only_blue = valid_blue & ~valid_red
        label_map[only_blue] = 1
        
        # Pixels with only red distance: background
        only_red = ~valid_blue & valid_red
        label_map[only_red] = 0
        
        # Compute confidence map (how certain is the assignment)
        confidence = np.zeros((self.height, self.width), dtype=np.float32)
        confidence[both_valid] = np.abs(distance_blue[both_valid] - distance_red[both_valid])
        
        # Normalize confidence to [0, 1]
        if confidence.max() > 0:
            confidence = confidence / confidence.max()
        
        results = {
            'label_map': label_map,
            'distance_blue': distance_blue,
            'distance_red': distance_red,
            'confidence': confidence,
            'gradient': self.gradient
        }
        
        return results
    
    def post_process_segmentation(self, label_map, sigma=2.0, min_size=50):
        """
        Post-process segmentation with morphological operations and smoothing.
        
        Args:
            label_map (np.ndarray): Binary segmentation label map
            sigma (float): Sigma for Gaussian smoothing
            min_size (int): Minimum connected component size to keep
        
        Returns:
            np.ndarray: Post-processed label map
        """
        # Apply Gaussian smoothing to soft boundaries
        label_soft = gaussian_filter(label_map.astype(float), sigma=sigma)
        
        # Apply morphological operations to clean up
        binary_seg = label_soft > 0.5
        
        # Remove small connected components
        labeled_array, num_features = ndimage.label(binary_seg)
        
        # Calculate size of each component
        sizes = ndimage.sum(binary_seg, labeled_array, range(num_features + 1))
        
        # Remove small components
        for i in range(1, num_features + 1):
            if sizes[i] < min_size:
                binary_seg[labeled_array == i] = False
        
        return binary_seg.astype(np.uint8) * 255


def segmentation_from_ui(image, blue_dots, red_dots, gradient_sigma=1.0):
    """
    Convenience function to perform segmentation from UI inputs.
    
    Args:
        image (PIL.Image): Input image from UI
        blue_dots (list): Blue seed points from UI: [(x, y), ...]
        red_dots (list): Red seed points from UI: [(x, y), ...]
        gradient_sigma (float): Gradient sigma for preprocessing
    
    Returns:
        dict: Segmentation results with 'label_map', 'distance_blue', 'distance_red', etc.
    """
    segmentor = GeodesicSegmentation(
        image=image,
        blue_seeds=blue_dots,
        red_seeds=red_dots,
        gradient_sigma=gradient_sigma
    )
    
    results = segmentor.segment(connectivity=8)
    return results


def visualize_segmentation(original_image, segmentation_results, output_path=None):
    """
    Create visualization of segmentation results.
    
    Args:
        original_image (PIL.Image or np.ndarray): Original input image
        segmentation_results (dict): Results from segment()
        output_path (str): Optional path to save visualization
    
    Returns:
        PIL.Image: Visualization image with overlay and distance maps
    """
    if isinstance(original_image, Image.Image):
        orig_array = np.array(original_image.convert('RGB'))
    else:
        orig_array = original_image
    
    label_map = segmentation_results['label_map']
    distance_blue = segmentation_results['distance_blue']
    distance_red = segmentation_results['distance_red']
    confidence = segmentation_results['confidence']
    
    height, width = label_map.shape
    
    # Create multi-panel visualization
    vis_image = Image.new('RGB', (width * 3, height * 2), color='white')
    
    # Panel 1: Original image
    if orig_array.dtype != np.uint8:
        orig_array = (orig_array * 255).astype(np.uint8)
    orig_img = Image.fromarray(orig_array if len(orig_array.shape) == 3 else 
                               np.repeat(orig_array[:, :, np.newaxis], 3, axis=2).astype(np.uint8))
    vis_image.paste(orig_img, (0, 0))
    
    # Panel 2: Segmentation result (blue=foreground, red=background)
    seg_array = np.zeros((height, width, 3), dtype=np.uint8)
    seg_array[label_map == 1] = [0, 0, 255]      # Blue for foreground
    seg_array[label_map == 0] = [255, 0, 0]      # Red for background
    seg_img = Image.fromarray(seg_array)
    vis_image.paste(seg_img, (width, 0))
    
    # Panel 3: Overlay on original
    overlay = orig_img.copy()
    overlay_array = np.array(overlay).astype(float)
    overlay_array[label_map == 1, 2] = np.minimum(overlay_array[label_map == 1, 2] + 100, 255)  # Blue highlight
    overlay_array[label_map == 0, 0] = np.minimum(overlay_array[label_map == 0, 0] + 100, 255)  # Red highlight
    overlay_img = Image.fromarray(overlay_array.astype(np.uint8))
    vis_image.paste(overlay_img, (width * 2, 0))
    
    # Panel 4: Distance map from blue seeds (normalized for visualization)
    if np.isfinite(distance_blue).any():
        dist_blue_norm = np.copy(distance_blue)
        dist_blue_norm[~np.isfinite(dist_blue_norm)] = 0
        dist_blue_norm = (dist_blue_norm / dist_blue_norm.max() * 255).astype(np.uint8) if dist_blue_norm.max() > 0 else dist_blue_norm
        dist_blue_img = Image.fromarray(dist_blue_norm, mode='L').convert('RGB')
        vis_image.paste(dist_blue_img, (0, height))
    
    # Panel 5: Distance map from red seeds (normalized for visualization)
    if np.isfinite(distance_red).any():
        dist_red_norm = np.copy(distance_red)
        dist_red_norm[~np.isfinite(dist_red_norm)] = 0
        dist_red_norm = (dist_red_norm / dist_red_norm.max() * 255).astype(np.uint8) if dist_red_norm.max() > 0 else dist_red_norm
        dist_red_img = Image.fromarray(dist_red_norm, mode='L').convert('RGB')
        vis_image.paste(dist_red_img, (width, height))
    
    # Panel 6: Confidence map (normalized for visualization)
    conf_norm = (confidence * 255).astype(np.uint8)
    conf_img = Image.fromarray(conf_norm, mode='L').convert('RGB')
    vis_image.paste(conf_img, (width * 2, height))
    
    if output_path:
        vis_image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    
    return vis_image


if __name__ == "__main__":
    # Example usage
    print("Geodesic Segmentation Module")
    print("=" * 50)
    print("Import this module and use:")
    print("  - GeodesicSegmentation class for direct control")
    print("  - segmentation_from_ui() for convenient UI integration")
    print("  - visualize_segmentation() for result visualization")
