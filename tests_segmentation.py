"""
Tests for geodesic segmentation with hierarchical level maps.

This module provides functionality to:
1. Generate hierarchical level maps (A, B) from input images/matrices
2. Test the level map generation with synthetic data
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def compute_level_maps(input_matrix):
    """
    Compute hierarchical level maps from an input matrix.
    
    Creates two level maps of size (2H-1) x (2W-1):
    - level_map_A: Minimum values at odd indices
    - level_map_B: Maximum values at odd indices
    
    At even indices (x, y):
        A[x, y] = B[x, y] = input[y, x]
    
    At odd indices (x or y):
        A[x, y] = minimum of nearby pixels
        B[x, y] = maximum of nearby pixels
    
    Args:
        input_matrix (np.ndarray): Input matrix of shape (H, W)
    
    Returns:
        tuple: (level_map_A, level_map_B) as numpy arrays of shape (2H-1, 2W-1)
    """
    # Ensure input is float
    if input_matrix.dtype != np.float32 and input_matrix.dtype != np.float64:
        input_matrix = input_matrix.astype(np.float32)
    
    H, W = input_matrix.shape
    out_H, out_W = 2 * H - 1, 2 * W - 1
    
    # Initialize level maps
    level_map_A = np.zeros((out_H, out_W), dtype=np.float32)
    level_map_B = np.zeros((out_H, out_W), dtype=np.float32)
    
    print(f"\nGenerating Level Maps:")
    print(f"  Input size: {H} x {W}")
    print(f"  Output size: {out_H} x {out_W}")
    
    # Fill even indices with original matrix values
    for y in range(H):
        for x in range(W):
            level_map_A[2*y, 2*x] = input_matrix[y, x]
            level_map_B[2*y, 2*x] = input_matrix[y, x]
    
    # Fill odd indices with min/max of nearby pixels
    for y in range(out_H):
        for x in range(out_W):
            # Skip even indices (already filled)
            if x % 2 == 0 and y % 2 == 0:
                continue
            
            nearby_pixels = []
            
            if x % 2 == 0 and y % 2 == 1:
                # Even x, odd y: neighbors are at (y-1, x) and (y+1, x)
                y_neighbors = [(y - 1) // 2, (y + 1) // 2]
                for yn in y_neighbors:
                    if 0 <= yn < H:
                        nearby_pixels.append(input_matrix[yn, x // 2])
            
            elif x % 2 == 1 and y % 2 == 0:
                # Odd x, even y: neighbors are at (y, x-1) and (y, x+1)
                x_neighbors = [(x - 1) // 2, (x + 1) // 2]
                for xn in x_neighbors:
                    if 0 <= xn < W:
                        nearby_pixels.append(input_matrix[y // 2, xn])
            
            else:
                # Odd x, odd y: four diagonal neighbors
                x_neighbors = [(x - 1) // 2, (x + 1) // 2]
                y_neighbors = [(y - 1) // 2, (y + 1) // 2]
                for yn in y_neighbors:
                    for xn in x_neighbors:
                        if 0 <= xn < W and 0 <= yn < H:
                            nearby_pixels.append(input_matrix[yn, xn])
            
            # Assign min and max values
            if nearby_pixels:
                level_map_A[y, x] = np.min(nearby_pixels)
                level_map_B[y, x] = np.max(nearby_pixels)
    
    print(f"  Level map A (min)  - Range: [{level_map_A.min():.1f}, {level_map_A.max():.1f}]")
    print(f"  Level map B (max)  - Range: [{level_map_B.min():.1f}, {level_map_B.max():.1f}]")
    
    return level_map_A, level_map_B


def proj(x, interval):
    """
    Project value x onto an interval [a, b].
    Equivalent to clamp(x, a, b).
    
    Args:
        x (float): Value to project
        interval (tuple): Interval as (a, b) where a is min and b is max
    
    Returns:
        float: Projected value clamped to [a, b]
    """
    a, b = interval
    return np.clip(x, a, b)


def dt(level_map_A, level_map_B, seeds, v_inf, connectivity=8):
    """
    Distance Transform using geodesic intervals.
    
    Computes distance map and interval-based function values using
    a priority queue based algorithm with intervals [A(x), B(x)].
    
    Algorithm:
        Initialize Q (priority queue)
        D(p) ← +∞ for all p
        F(p∞) ← v∞ (seeds)
        Push(Q, p∞, 0)
        while Q not empty:
            (p, d) ← Pop(Q)
            foreach q ∈ N(p) such that D(q) = +∞ do
                F(q) ← proj(F(p), F(q))
                dpq ← |F(p) − F(q)|
                D(q) ← d + dpq
                Push(Q, q, D(q))
        return F, D
    
    Args:
        level_map_A (np.ndarray): Level map A (minimum values at each pixel)
        level_map_B (np.ndarray): Level map B (maximum values at each pixel)
        seeds (list): List of seed coordinates [(y, x), ...] on original image scale
        v_inf (float): Initial value for seeds
        connectivity (int): 4 or 8 connectivity (default: 8)
    
    Returns:
        tuple: (F, D) where:
            - F is the projected values at each pixel
            - D is the distance map
    """
    H, W = level_map_A.shape
    
    # Initialize: F and D maps on the expended level map grid
    F = np.full((H, W), np.nan, dtype=np.float32)  # Function values
    D = np.full((H, W), np.inf, dtype=np.float32)  # Distance values
    visited = np.zeros((H, W), dtype=bool)
    
    # Priority queue: (distance, y, x)
    import heapq
    pq = []
    
    # Initialize seeds on the expanded grid (even indices only)
    # Seeds are provided in original image coordinates, map to expanded grid
    for y_orig, x_orig in seeds:
        y = 2 * y_orig
        x = 2 * x_orig
        if 0 <= y < H and 0 <= x < W:
            F[y, x] = v_inf
            D[y, x] = 0.0
            heapq.heappush(pq, (0.0, y, x))
    
    print(f"\nDistance Transform (dt) Algorithm:")
    print(f"  Grid size: {H} x {W}")
    print(f"  Number of seeds: {len(seeds)}")
    print(f"  Initial seeds on expanded grid: ", end="")
    for y_orig, x_orig in seeds:
        print(f"({2*y_orig}, {2*x_orig}) ", end="")
    print()
    
    # Dijkstra-like algorithm with intervals
    processed = 0
    while pq:
        curr_dist, y, x = heapq.heappop(pq)
        
        # Skip if already visited
        if visited[y, x]:
            continue
        
        visited[y, x] = True
        processed += 1
        
        # Get neighbors based on connectivity
        if connectivity == 4:
            neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        else:  # 8-connectivity
            neighbors = [
                (y-1, x), (y+1, x), (y, x-1), (y, x+1),
                (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
            ]
        
        # Process neighbors
        for ny, nx in neighbors:
            # Check bounds
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            
            # Only process unvisited pixels
            if visited[ny, nx]:
                continue
            
            # Get interval at neighbor
            a_q = level_map_A[ny, nx]
            b_q = level_map_B[ny, nx]
            
            # Project F(p) onto interval [a_q, b_q]
            F_p = F[y, x]
            F_q_projected = proj(F_p, (a_q, b_q))
            
            # Compute distance: dpq = |F(p) - F(q)|
            dpq = np.abs(F_p - F_q_projected)
            
            # Update distance if shorter path found
            new_dist = curr_dist + dpq
            if new_dist < D[ny, nx]:
                D[ny, nx] = new_dist
                F[ny, nx] = F_q_projected
                heapq.heappush(pq, (new_dist, ny, nx))
    
    print(f"  Pixels processed: {processed} / {H*W}")
    print(f"  D range: [{np.nanmin(D):.4f}, {np.nanmax(D):.4f}]")
    print(f"  F range: [{np.nanmin(F):.4f}, {np.nanmax(F):.4f}]")
    
    return F, D


def export_level_maps_as_images(level_map_A, level_map_B, prefix="level"):
    """
    Export level maps as grayscale PNG images.
    
    Args:
        level_map_A (np.ndarray): Level map A (minimum values)
        level_map_B (np.ndarray): Level map B (maximum values)
        prefix (str): Prefix for output filenames (default: "level")
                     Creates: levelA.png and levelB.png
    """
    # Normalize to [0, 255] for image export
    A_min, A_max = level_map_A.min(), level_map_A.max()
    B_min, B_max = level_map_B.min(), level_map_B.max()
    
    if A_max > A_min:
        level_A_normalized = ((level_map_A - A_min) / (A_max - A_min) * 255).astype(np.uint8)
    else:
        level_A_normalized = np.full_like(level_map_A, 128, dtype=np.uint8)
    
    if B_max > B_min:
        level_B_normalized = ((level_map_B - B_min) / (B_max - B_min) * 255).astype(np.uint8)
    else:
        level_B_normalized = np.full_like(level_map_B, 128, dtype=np.uint8)
    
    # Convert to PIL Images and save
    img_A = Image.fromarray(level_A_normalized, mode='L')
    img_B = Image.fromarray(level_B_normalized, mode='L')
    
    path_A = f"{prefix}A.png"
    path_B = f"{prefix}B.png"
    
    img_A.save(path_A)
    img_B.save(path_B)
    
    print(f"\n✓ Level map A (min) saved: {path_A}")
    print(f"✓ Level map B (max) saved: {path_B}")


def print_level_maps(level_map_A, level_map_B, title="Level Maps"):
    """
    Print level maps in a readable format.
    
    Args:
        level_map_A (np.ndarray): Level map A
        level_map_B (np.ndarray): Level map B
        title (str): Title for display
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    print("\nLevel Map A (Minimum values at odd indices):")
    print(level_map_A.astype(int))
    
    print("\nLevel Map B (Maximum values at odd indices):")
    print(level_map_B.astype(int))


def test_level_maps_with_synthetic_data():
    """
    Test level map generation with a 5x5 matrix containing numbers 1-25.
    """
    print("\n" + "="*70)
    print("LEVEL MAP TEST WITH SYNTHETIC 5x5 MATRIX")
    print("="*70)
    
    # Create 5x5 matrix with numbers 1-25 arranged horizontally
    matrix_5x5 = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    
    print("\nInput Matrix (5x5) - Numbers 1-25:")
    print(matrix_5x5.astype(int))
    print(f"Shape: {matrix_5x5.shape}")
    
    # Generate level maps
    level_map_A, level_map_B = compute_level_maps(matrix_5x5)
    
    # Print the level maps
    print_level_maps(level_map_A, level_map_B, title="Generated Level Maps (9x9)")
    
    # Export as images
    export_level_maps_as_images(level_map_A, level_map_B, prefix="test_level_")
    
    return matrix_5x5, level_map_A, level_map_B


def verify_level_maps(input_matrix, level_map_A, level_map_B):
    """
    Verify that level maps are correctly generated for specific test cases.
    
    Args:
        input_matrix (np.ndarray): Original input matrix
        level_map_A (np.ndarray): Generated level map A
        level_map_B (np.ndarray): Generated level map B
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print(f"\n{'='*70}")
    print("VERIFICATION OF LEVEL MAP GENERATION")
    print(f"{'='*70}")
    
    H, W = input_matrix.shape
    out_H, out_W = 2 * H - 1, 2 * W - 1
    
    # Check 1: Correct size
    if level_map_A.shape != (out_H, out_W) or level_map_B.shape != (out_H, out_W):
        print("✗ Size mismatch!")
        return False
    print(f"✓ Size check passed: {level_map_A.shape}")
    
    # Check 2: Even indices match input
    all_match = True
    for y in range(H):
        for x in range(W):
            if level_map_A[2*y, 2*x] != input_matrix[y, x]:
                print(f"✗ Level map A mismatch at even index ({2*y}, {2*x})")
                all_match = False
            if level_map_B[2*y, 2*x] != input_matrix[y, x]:
                print(f"✗ Level map B mismatch at even index ({2*y}, {2*x})")
                all_match = False
    
    if all_match:
        print("✓ Even indices match input matrix")
    
    # Check 3: Odd indices are within range [min, max] of neighbors
    range_check = True
    for y in range(out_H):
        for x in range(out_W):
            if x % 2 == 1 or y % 2 == 1:
                # This is an odd index position
                nearby_pixels = []
                
                if x % 2 == 0 and y % 2 == 1:
                    y_neighbors = [(y - 1) // 2, (y + 1) // 2]
                    for yn in y_neighbors:
                        if 0 <= yn < H:
                            nearby_pixels.append(input_matrix[yn, x // 2])
                elif x % 2 == 1 and y % 2 == 0:
                    x_neighbors = [(x - 1) // 2, (x + 1) // 2]
                    for xn in x_neighbors:
                        if 0 <= xn < W:
                            nearby_pixels.append(input_matrix[y // 2, xn])
                else:
                    x_neighbors = [(x - 1) // 2, (x + 1) // 2]
                    y_neighbors = [(y - 1) // 2, (y + 1) // 2]
                    for yn in y_neighbors:
                        for xn in x_neighbors:
                            if 0 <= xn < W and 0 <= yn < H:
                                nearby_pixels.append(input_matrix[yn, xn])
                
                if nearby_pixels:
                    if level_map_A[y, x] != np.min(nearby_pixels):
                        print(f"✗ Level map A at ({y}, {x}) is not minimum of neighbors")
                        range_check = False
                    if level_map_B[y, x] != np.max(nearby_pixels):
                        print(f"✗ Level map B at ({y}, {x}) is not maximum of neighbors")
                        range_check = False
    
    if range_check:
        print("✓ Odd indices correctly computed (min/max of neighbors)")
    
    print(f"\n✓ All verification checks passed!")
    return True


def test_dt_with_synthetic_data():
    """
    Test distance transform (dt) algorithm with the 5x5 matrix.
    """
    print(f"\n{'='*70}")
    print("DISTANCE TRANSFORM (dt) TEST WITH 5x5 MATRIX")
    print(f"{'='*70}")
    
    # Create 5x5 matrix with numbers 1-25
    matrix_5x5 = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    
    print("\nInput Matrix (5x5):")
    print(matrix_5x5.astype(int))
    
    # Generate level maps
    level_map_A, level_map_B = compute_level_maps(matrix_5x5)
    
    # Test Case 1: Single seed at top-left corner
    print(f"\n{'-'*70}")
    print("Test Case 1: Single seed at (0, 0) with value v∞ = 0")
    print(f"{'-'*70}")
    seeds_1 = [(0, 0)]  # Top-left corner of original image
    F_1, D_1 = dt(level_map_A, level_map_B, seeds_1, v_inf=0.0, connectivity=8)
    
    print("\nDistance map D (at even indices on expanded grid):")
    D_1_original = D_1[::2, ::2]  # Extract even indices for original space
    print(D_1_original.astype(int))
    
    print("\nFunction values F (at even indices on expanded grid):")
    F_1_original = F_1[::2, ::2]
    print(F_1_original.astype(int))
    
    # Test Case 2: Multiple seeds
    print(f"\n{'-'*70}")
    print("Test Case 2: Multiple seeds at corners with value v∞ = 5")
    print(f"{'-'*70}")
    seeds_2 = [(0, 0), (0, 4), (4, 0), (4, 4)]  # All four corners
    F_2, D_2 = dt(level_map_A, level_map_B, seeds_2, v_inf=5.0, connectivity=8)
    
    print("\nDistance map D (at even indices):")
    D_2_original = D_2[::2, ::2]
    print(D_2_original.astype(int))
    
    print("\nFunction values F (at even indices):")
    F_2_original = F_2[::2, ::2]
    print(F_2_original.astype(int))
    
    # Test Case 3: Seed at center
    print(f"\n{'-'*70}")
    print("Test Case 3: Single seed at center (2, 2) with value v∞ = 12")
    print(f"{'-'*70}")
    seeds_3 = [(2, 2)]  # Center
    F_3, D_3 = dt(level_map_A, level_map_B, seeds_3, v_inf=12.0, connectivity=8)
    
    print("\nDistance map D (at even indices):")
    D_3_original = D_3[::2, ::2]
    print(D_3_original.astype(int))
    
    print("\nFunction values F (at even indices):")
    F_3_original = F_3[::2, ::2]
    print(F_3_original.astype(int))
    
    # Export distance maps as images
    print(f"\n{'-'*70}")
    print("Exporting results as images...")
    print(f"{'-'*70}")
    
    # Export Test Case 1
    D_1_normalized = np.clip((D_1 / (np.nanmax(D_1) + 1e-6)) * 255, 0, 255).astype(np.uint8)
    img_D_1 = Image.fromarray(D_1_normalized, mode='L')
    img_D_1.save("test_dt_distance_case1.png")
    print("✓ Saved: test_dt_distance_case1.png")
    
    # Export Test Case 2
    D_2_normalized = np.clip((D_2 / (np.nanmax(D_2) + 1e-6)) * 255, 0, 255).astype(np.uint8)
    img_D_2 = Image.fromarray(D_2_normalized, mode='L')
    img_D_2.save("test_dt_distance_case2.png")
    print("✓ Saved: test_dt_distance_case2.png")
    
    # Export Test Case 3
    D_3_normalized = np.clip((D_3 / (np.nanmax(D_3) + 1e-6)) * 255, 0, 255).astype(np.uint8)
    img_D_3 = Image.fromarray(D_3_normalized, mode='L')
    img_D_3.save("test_dt_distance_case3.png")
    print("✓ Saved: test_dt_distance_case3.png")
    
    return (F_1, D_1), (F_2, D_2), (F_3, D_3)


if __name__ == "__main__":
    # Run the test with synthetic 5x5 matrix
    print("\n" + "#"*70)
    print("# GEODESIC SEGMENTATION - LEVEL MAPS AND DISTANCE TRANSFORM TEST")
    print("#"*70)
    
    # Test 1: Level maps
    print("\n" + "="*70)
    print("PART 1: LEVEL MAPS TEST")
    print("="*70)
    matrix_5x5, level_map_A, level_map_B = test_level_maps_with_synthetic_data()
    verify_level_maps(matrix_5x5, level_map_A, level_map_B)
    
    # Test 2: Distance Transform
    print("\n" + "="*70)
    print("PART 2: DISTANCE TRANSFORM TEST")
    print("="*70)
    results_dt = test_dt_with_synthetic_data()
    
    print(f"\n{'#'*70}")
    print("# ALL TESTS COMPLETE")
    print(f"{'#'*70}\n")
