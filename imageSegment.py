import cv2
import numpy as np

def segmentImage(inputImg):
    """
    Segments leaf disease regions from leaf images using traditional computer vision techniques.
    
    Args:
        inputImg: Input image, a 3D numpy array of row*col*3 in BGR format
        
    Returns:
        outputImg: A 2D numpy array segmentation mask where:
                  - Background = 0
                  - Disease Region = 1
    """
    
    # Step 1: Preprocessing - Noise reduction and smoothing
    # Apply bilateral filter to reduce noise while preserving edges
    img_filtered = cv2.bilateralFilter(inputImg, 9, 75, 75)
    
    # Step 2: Color space conversions for better feature extraction
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
    
    # Convert to LAB for perceptual color differences
    lab = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2LAB)
    
    # Convert to RGB for easier color analysis
    rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
    
    # Step 3: Extract individual channels for analysis
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    r, g, bl = cv2.split(rgb)
    
    # Step 4: Create initial leaf mask (separate leaf from background)
    # Leaves typically have green coloration, use multiple approaches
    
    # Method 1: HSV-based green detection
    # Green hue range in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Method 2: RGB-based approach - green channel dominance
    # Healthy leaves usually have higher green values
    green_dominant = (g > r) & (g > bl) & (g > 50)
    green_dominant = green_dominant.astype(np.uint8) * 255
    
    # Method 3: Saturation-based approach
    # Leaves typically have higher saturation than background
    sat_mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Combine leaf detection methods
    leaf_mask = cv2.bitwise_or(green_mask, green_dominant)
    leaf_mask = cv2.bitwise_or(leaf_mask, sat_mask)
    
    # Clean up leaf mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Disease detection within leaf regions
    # Diseased areas typically have different color characteristics
    
    # Method 1: Detect brown/yellow diseased areas using HSV
    # Brown/yellow hue range
    lower_disease1 = np.array([10, 50, 50])
    upper_disease1 = np.array([30, 255, 255])
    disease_mask1 = cv2.inRange(hsv, lower_disease1, upper_disease1)
    
    # Method 2: Detect darker diseased spots
    # Often diseased areas are darker
    lower_disease2 = np.array([0, 30, 20])
    upper_disease2 = np.array([180, 255, 100])
    disease_mask2 = cv2.inRange(hsv, lower_disease2, upper_disease2)
    
    # Method 3: LAB color space approach
    # Diseased areas often have different a* and b* values
    # Higher a* values (more red/magenta)
    a_thresh = cv2.threshold(a, 135, 255, cv2.THRESH_BINARY)[1]
    
    # Method 4: Edge-based approach for spot detection
    # Apply Gaussian blur and find edges
    gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to create regions
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
    
    # Method 5: Texture-based approach using local variance
    # Diseased areas often have different texture
    kernel_texture = np.ones((9, 9), np.float32) / 81
    mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel_texture)
    sqr_img = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel_texture)
    variance = sqr_img - mean_img**2
    
    # Normalize variance and threshold
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    texture_mask = cv2.threshold(variance_norm, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Step 6: Combine disease detection methods
    disease_combined = cv2.bitwise_or(disease_mask1, disease_mask2)
    disease_combined = cv2.bitwise_or(disease_combined, a_thresh)
    
    # Add texture information
    disease_combined = cv2.bitwise_or(disease_combined, texture_mask)
    
    # Step 7: Refine disease mask within leaf regions only
    # Only keep disease pixels that are within leaf areas
    disease_in_leaf = cv2.bitwise_and(disease_combined, leaf_mask)
    
    # Step 8: Morphological operations to clean up disease regions
    # Remove small noise and fill gaps
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_cleaned = cv2.morphologyEx(disease_in_leaf, cv2.MORPH_OPEN, kernel_clean)
    disease_cleaned = cv2.morphologyEx(disease_cleaned, cv2.MORPH_CLOSE, kernel_clean)
    
    # Step 9: Additional refinement using connected components
    # Remove very small connected components (noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(disease_cleaned, connectivity=8)
    
    # Create final mask
    final_disease_mask = np.zeros_like(disease_cleaned)
    
    # Keep only reasonably sized components
    min_area = 50  # Minimum area for disease regions
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_disease_mask[labels == i] = 255
    
    # Step 10: Final post-processing
    # Apply final morphological closing to smooth boundaries
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final_disease_mask = cv2.morphologyEx(final_disease_mask, cv2.MORPH_CLOSE, kernel_final)
    
    # Step 11: Convert to required output format
    # Background = 0, Disease = 1
    outputImg = (final_disease_mask > 0).astype(np.uint8)
    
    # Step 12: Apply bitwise NOT if needed for better results
    # This can help if the initial segmentation is inverted
    # Uncomment the line below if you need to invert the results
    # outputImg = cv2.bitwise_not(outputImg)
    
    # Alternative approach: Use bitwise_not to improve disease detection
    # by inverting intermediate masks for better combination
    inverted_leaf_mask = cv2.bitwise_not(leaf_mask)
    
    # Create alternative disease detection using inverted leaf mask
    # This can help detect diseases that appear as "holes" in healthy tissue
    alt_disease_mask = cv2.bitwise_and(inverted_leaf_mask, disease_combined)
    
    # Combine original and alternative detection
    combined_final = cv2.bitwise_or(final_disease_mask, alt_disease_mask)
    
    # Clean the combined result
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined_final = cv2.morphologyEx(combined_final, cv2.MORPH_CLOSE, kernel_final)
    combined_final = cv2.morphologyEx(combined_final, cv2.MORPH_OPEN, kernel_final)
    
    # Apply connected components analysis again
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(combined_final, connectivity=8)
    
    # Filter small components
    refined_mask = np.zeros_like(combined_final)
    min_area = 30  # Reduced minimum area after bitwise operations
    
    for i in range(1, num_labels_final):
        if stats_final[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels_final == i] = 255
    
    # Final output with bitwise_not consideration
    outputImg = (refined_mask > 0).astype(np.uint8)
    
    # Optional: Apply final bitwise_not if the results are consistently inverted
    # You can enable this based on evaluation results
    APPLY_FINAL_INVERSION = False  # Set to True if needed
    
    if APPLY_FINAL_INVERSION:
        outputImg = cv2.bitwise_not(outputImg)
    
    return outputImg


def enhance_disease_detection(inputImg, leaf_mask):
    """
    Additional helper function for enhanced disease detection
    """
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(inputImg, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(inputImg, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Create multiple disease masks
    masks = []
    
    # Mask 1: Yellow/brown spots (common in many leaf diseases)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masks.append(yellow_mask)
    
    # Mask 2: Dark spots (necrotic areas)
    dark_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)[1]
    masks.append(dark_mask)
    
    # Mask 3: High 'a' channel values (reddish areas)
    a_mask = cv2.threshold(a, 130, 255, cv2.THRESH_BINARY)[1]
    masks.append(a_mask)
    
    # Combine all masks
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply only within leaf regions
    disease_mask = cv2.bitwise_and(combined_mask, leaf_mask)
    
    return disease_mask

def segmentImage_with_inversion_analysis(inputImg):
    """
    Enhanced segmentation with bitwise_not analysis to determine optimal output
    """
    # Get both normal and inverted versions
    normal_result = segmentImage(inputImg)
    inverted_result = cv2.bitwise_not(normal_result)
    
    # Analyze which version makes more sense
    normal_disease_ratio = np.sum(normal_result == 1) / normal_result.size
    inverted_disease_ratio = np.sum(inverted_result == 1) / inverted_result.size
    
    # Heuristic: disease should typically be 5-30% of the image
    # If normal result has disease ratio in this range, use normal
    # If inverted result is closer to this range, use inverted
    
    target_ratio = 0.15  # 15% target disease ratio
    normal_diff = abs(normal_disease_ratio - target_ratio)
    inverted_diff = abs(inverted_disease_ratio - target_ratio)
    
    if inverted_diff < normal_diff and 0.05 < inverted_disease_ratio < 0.4:
        return inverted_result
    else:
        return normal_result

def apply_bitwise_operations_for_enhancement(inputImg):
    """
    Use bitwise_not operations strategically to enhance disease detection
    """
    # Step 1: Get initial segmentation
    img_filtered = cv2.bilateralFilter(inputImg, 9, 75, 75)
    
    # Step 2: Create multiple analysis approaches
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Step 3: Create base masks
    # Healthy leaf detection
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Step 4: Use bitwise_not to find non-healthy areas
    non_healthy_mask = cv2.bitwise_not(healthy_mask)
    
    # Step 5: Refine non-healthy areas to find diseases
    # Remove background from non-healthy areas
    # Background typically has low saturation
    low_sat_mask = cv2.threshold(s, 20, 255, cv2.THRESH_BINARY_INV)[1]
    background_mask = cv2.bitwise_and(non_healthy_mask, low_sat_mask)
    
    # Step 6: Use bitwise_not again to remove background from disease candidates
    disease_candidates = cv2.bitwise_and(non_healthy_mask, cv2.bitwise_not(background_mask))
    
    # Step 7: Further refine disease areas
    # Disease areas often have specific color characteristics
    lower_disease = np.array([10, 30, 30])
    upper_disease = np.array([30, 255, 200])
    disease_color_mask = cv2.inRange(hsv, lower_disease, upper_disease)
    
    # Dark disease spots
    dark_disease_mask = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Step 8: Combine disease indicators
    disease_combined = cv2.bitwise_or(disease_color_mask, dark_disease_mask)
    
    # Step 9: Final disease mask using bitwise operations
    final_disease = cv2.bitwise_and(disease_candidates, disease_combined)
    
    # Step 10: Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_disease = cv2.morphologyEx(final_disease, cv2.MORPH_OPEN, kernel)
    final_disease = cv2.morphologyEx(final_disease, cv2.MORPH_CLOSE, kernel)
    
    # Step 11: Connected components filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_disease, connectivity=8)
    
    refined_mask = np.zeros_like(final_disease)
    min_area = 25
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 255
    
    # Convert to binary output
    outputImg = (refined_mask > 0).astype(np.uint8)
    
    return outputImg