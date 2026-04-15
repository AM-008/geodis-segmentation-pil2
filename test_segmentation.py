"""
Diagnostic test script to visualize intermediate steps of geodesic segmentation.
This helps identify where blur is being introduced in the algorithm.
"""

import numpy as np
from PIL import Image
import sys
from segmentation import GeodesicSegmentation


def visualize_algorithm_steps(image_path, blue_seeds, red_seeds, output_prefix="test"):
    """
    Visualize each step of the geodesic segmentation algorithm.
    
    Args:
        image_path (str): Path to input image
        blue_seeds (list): List of (x, y) coordinates for blue seeds
        red_seeds (list): List of (x, y) coordinates for red seeds
        output_prefix (str): Prefix for output files
    """
    print(f"\n{'='*70}")
    print("GEODESIC SEGMENTATION DIAGNOSTIC TEST")
    print(f"{'='*70}\n")
    
    # Load image
    image = Image.open(image_path)
    print(f"1. Original Image")
    print(f"   Size: {image.size} pixels")
    print(f"   Mode: {image.mode}")
    
    # Save original
    if image.mode != 'L':
        image_gray = image.convert('L')
    else:
        image_gray = image
    image_gray.save(f"{output_prefix}_00_original.png")
    print(f"   ✓ Saved: {output_prefix}_00_original.png")
    
    # Initialize segmentor
    segmentor = GeodesicSegmentation(
        image=image_gray,
        blue_seeds=blue_seeds,
        red_seeds=red_seeds,
        gradient_sigma=1.0
    )
    
    # Step 2: Show smoothed image
    print(f"\n2. Smoothed Image (for gradient computation)")
    print(f"   Gaussian sigma: 1.0")
    smoothed_image = Image.fromarray((segmentor.image * 255).astype(np.uint8))
    smoothed_image.save(f"{output_prefix}_01_smoothed_image.png")
    print(f"   ✓ Saved: {output_prefix}_01_smoothed_image.png")
    
    # Step 3: Show gradient magnitude
    print(f"\n3. Gradient Magnitude Map")
    print(f"   Min gradient: {segmentor.gradient.min():.4f}")
    print(f"   Max gradient: {segmentor.gradient.max():.4f}")
    print(f"   Mean gradient: {segmentor.gradient.mean():.4f}")
    gradient_display = Image.fromarray((segmentor.gradient * 255).astype(np.uint8))
    gradient_display.save(f"{output_prefix}_02_gradient_magnitude.png")
    print(f"   ✓ Saved: {output_prefix}_02_gradient_magnitude.png")
    print(f"   (Bright = strong edges, Dark = smooth regions)")
    
    # Step 4: Run segmentation
    print(f"\n4. Running Dijkstra's Algorithm...")
    print(f"   Blue seeds: {len(blue_seeds)} points")
    print(f"   Red seeds: {len(red_seeds)} points")
    
    results = segmentor.segment(connectivity=8)
    
    # Step 5: Analyze distance maps
    print(f"\n5. Distance Maps from Seeds")
    
    distance_blue = results['distance_blue']
    distance_red = results['distance_red']
    
    valid_blue = ~np.isinf(distance_blue)
    valid_red = ~np.isinf(distance_red)
    
    print(f"   Blue Distance Map (foreground):")
    print(f"     Valid pixels: {valid_blue.sum()} / {distance_blue.size}")
    if valid_blue.sum() > 0:
        print(f"     Min: {distance_blue[valid_blue].min():.2f}")
        print(f"     Max: {distance_blue[valid_blue].max():.2f}")
        print(f"     Mean: {distance_blue[valid_blue].mean():.2f}")
    
    print(f"   Red Distance Map (background):")
    print(f"     Valid pixels: {valid_red.sum()} / {distance_red.size}")
    if valid_red.sum() > 0:
        print(f"     Min: {distance_red[valid_red].min():.2f}")
        print(f"     Max: {distance_red[valid_red].max():.2f}")
        print(f"     Mean: {distance_red[valid_red].mean():.2f}")
    
    # Save distance maps (normalized for visualization)
    if valid_blue.sum() > 0:
        dist_blue_norm = np.copy(distance_blue)
        dist_blue_norm[~valid_blue] = 0
        dist_blue_max = dist_blue_norm.max()
        if dist_blue_max > 0:
            dist_blue_norm = (dist_blue_norm / dist_blue_max * 255).astype(np.uint8)
        else:
            dist_blue_norm = dist_blue_norm.astype(np.uint8)
        dist_blue_img = Image.fromarray(dist_blue_norm, mode='L')
        dist_blue_img.save(f"{output_prefix}_03_distance_blue_seeds.png")
        print(f"   ✓ Saved: {output_prefix}_03_distance_blue_seeds.png")
    
    if valid_red.sum() > 0:
        dist_red_norm = np.copy(distance_red)
        dist_red_norm[~valid_red] = 0
        dist_red_max = dist_red_norm.max()
        if dist_red_max > 0:
            dist_red_norm = (dist_red_norm / dist_red_max * 255).astype(np.uint8)
        else:
            dist_red_norm = dist_red_norm.astype(np.uint8)
        dist_red_img = Image.fromarray(dist_red_norm, mode='L')
        dist_red_img.save(f"{output_prefix}_04_distance_red_seeds.png")
        print(f"   ✓ Saved: {output_prefix}_04_distance_red_seeds.png")
    
    # Step 6: Segmentation result
    print(f"\n6. Segmentation Result (Label Map)")
    label_map = results['label_map']
    
    foreground_pixels = (label_map == 1).sum()
    background_pixels = (label_map == 0).sum()
    total_pixels = label_map.size
    
    print(f"   Foreground (label=1): {foreground_pixels} pixels ({100*foreground_pixels/total_pixels:.1f}%)")
    print(f"   Background (label=0): {background_pixels} pixels ({100*background_pixels/total_pixels:.1f}%)")
    
    # Visualize label map
    label_viz = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    label_viz[label_map == 1] = [0, 0, 255]      # Blue for foreground
    label_viz[label_map == 0] = [255, 0, 0]      # Red for background
    label_img = Image.fromarray(label_viz, 'RGB')
    label_img.save(f"{output_prefix}_05_segmentation_labels.png")
    print(f"   ✓ Saved: {output_prefix}_05_segmentation_labels.png")
    print(f"   (Blue = foreground area of interest, Red = background)")
    
    # Step 7: Confidence map
    print(f"\n7. Confidence Map")
    confidence = results['confidence']
    print(f"   Min confidence: {confidence.min():.4f}")
    print(f"   Max confidence: {confidence.max():.4f}")
    print(f"   Mean confidence: {confidence.mean():.4f}")
    
    conf_norm = (confidence * 255).astype(np.uint8)
    conf_img = Image.fromarray(conf_norm, mode='L')
    conf_img.save(f"{output_prefix}_06_confidence.png")
    print(f"   ✓ Saved: {output_prefix}_06_confidence.png")
    print(f"   (Bright = high confidence, Dark = uncertain/boundary)")
    
    # Step 8: Final foreground extraction (background removed)
    print(f"\n8. Final Output (Foreground Only)")
    
    # Convert to RGBA
    img_array = np.array(image_gray.convert('RGBA'))
    
    # Apply segmentation mask
    alpha = np.zeros_like(label_map, dtype=np.uint8)
    alpha[label_map == 1] = 255  # Keep foreground
    alpha[label_map == 0] = 0    # Transparent background
    
    img_array[:, :, 3] = alpha
    output_image = Image.fromarray(img_array, 'RGBA')
    output_image.save(f"{output_prefix}_07_foreground_only.png")
    print(f"   ✓ Saved: {output_prefix}_07_foreground_only.png")
    print(f"   (Area of interest with transparent background)")
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll test images saved with prefix: '{output_prefix}_'")
    print(f"\nTo analyze blur, compare:")
    print(f"  - {output_prefix}_00_original.png (original)")
    print(f"  - {output_prefix}_01_smoothed_image.png (after Gaussian smoothing)")
    print(f"  - {output_prefix}_02_gradient_magnitude.png (gradient)")
    print(f"  - {output_prefix}_05_segmentation_labels.png (segmentation)")
    print(f"\nIf blur occurs at step 01, increase gradient_sigma.")
    print(f"If sharp at 05 but blurry at 07, post-processing is blurring.\n")
    
    return results


def test_with_sample_image():
    """Run test with a sample image and synthetic seeds."""
    
    # Create a simple test image with objects
    print("\nGenerating synthetic test image...")
    test_image = Image.new('L', (400, 300), color=128)
    pixels = test_image.load()
    
    # Add a bright rectangle (object of interest)
    for y in range(80, 220):
        for x in range(100, 300):
            pixels[x, y] = 200
    
    # Add a dark rectangle (background reference)
    for y in range(250, 295):
        for x in range(10, 100):
            pixels[x, y] = 50
    
    # Add some noise
    np.random.seed(42)
    img_array = np.array(test_image).astype(float)
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255)
    test_image = Image.fromarray(img_array.astype(np.uint8))
    
    test_image.save("test_synthetic_image.png")
    print("✓ Saved: test_synthetic_image.png")
    
    # Define seeds
    blue_seeds = [(150, 120), (200, 150), (250, 100)]  # On bright object
    red_seeds = [(50, 270), (70, 280)]                  # On dark background
    
    print(f"\nBlue seeds (foreground): {blue_seeds}")
    print(f"Red seeds (background): {red_seeds}")
    
    # Run diagnostic
    results = visualize_algorithm_steps(
        "test_synthetic_image.png",
        blue_seeds,
        red_seeds,
        output_prefix="diagnostic_synthetic"
    )
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GEODESIC SEGMENTATION DIAGNOSTIC TOOL")
    print("="*70)
    
    if len(sys.argv) > 1:
        # Use provided image and seeds
        image_path = sys.argv[1]
        # Parse seeds (format: "x1,y1 x2,y2" for blue, "rx1,ry1 rx2,ry2" for red)
        # Example: test_segmentation.py image.png "100,100 150,150" "200,200 250,250"
        
        if len(sys.argv) > 3:
            blue_coords = sys.argv[2].split()
            red_coords = sys.argv[3].split()
            
            blue_seeds = [tuple(map(int, c.split(','))) for c in blue_coords]
            red_seeds = [tuple(map(int, c.split(','))) for c in red_coords]
            
            visualize_algorithm_steps(image_path, blue_seeds, red_seeds)
        else:
            print("\nUsage: python test_segmentation.py <image_path> '<blue_seeds>' '<red_seeds>'")
            print("Example: python test_segmentation.py image.png '100,100 150,150' '200,200'")
            print("\nOr run with no arguments to generate a synthetic test image:")
            test_with_sample_image()
    else:
        # Run with synthetic test image
        test_with_sample_image()
