import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from imageSegment import segmentImage
import time

def calculate_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return precision, recall, f1, iou

def load_ground_truth(gt_path):
    gt_img = cv2.imread(gt_path)
    if gt_img is None:
        return None
    
    if len(gt_img.shape) == 3:
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        red_mask = (gt_rgb[:,:,0] > 100) & (gt_rgb[:,:,1] < 50) & (gt_rgb[:,:,2] < 50)
        return red_mask.astype(np.uint8)
    else:
        return (gt_img > 128).astype(np.uint8)

def test_single_image(img_path, gt_path=None):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    start_time = time.time()
    result = segmentImage(img)
    processing_time = time.time() - start_time
    
    disease_pixels = np.sum(result == 1)
    total_pixels = result.size
    disease_percentage = (disease_pixels / total_pixels) * 100
    
    metrics = None
    gt_mask = None
    
    if gt_path and os.path.exists(gt_path):
        gt_mask = load_ground_truth(gt_path)
        if gt_mask is not None:
            precision, recall, f1, iou = calculate_metrics(result, gt_mask)
            metrics = (precision, recall, f1, iou)
    
    return {
        'image': img,
        'result': result,
        'gt_mask': gt_mask,
        'metrics': metrics,
        'processing_time': processing_time,
        'disease_percentage': disease_percentage
    }

def display_single_result(filename, data, image_num, total_images):
    print(f"\n{'='*60}")
    print(f"IMAGE {image_num}/{total_images}: {filename}")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Result
    axes[1].imshow(data['result'], cmap='gray')
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    # Ground Truth
    if data['gt_mask'] is not None:
        axes[2].imshow(data['gt_mask'], cmap='Reds')
        axes[2].set_title('Ground Truth')
    else:
        axes[2].text(0.5, 0.5, 'NO GT', ha='center', va='center', transform=axes[2].transAxes, fontsize=16)
        axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Overlay
    overlay = data['image'].copy()
    overlay[data['result'] == 1] = [0, 0, 255]
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Disease Overlay (Red)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"Processing Time: {data['processing_time']:.4f} seconds")
    print(f"Disease Percentage: {data['disease_percentage']:.2f}%")
    print(f"Disease Pixels: {np.sum(data['result'] == 1)}")
    print(f"Total Pixels: {data['result'].size}")
    
    if data['metrics']:
        precision, recall, f1, iou = data['metrics']
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"IoU: {iou:.4f}")
        
        # Quality assessment
        if f1 > 0.7:
            print("🟢 EXCELLENT RESULT")
        elif f1 > 0.5:
            print("🟡 GOOD RESULT")
        elif f1 > 0.3:
            print("🟠 MODERATE RESULT")
        else:
            print("🔴 POOR RESULT")
    else:
        print("No ground truth available for evaluation")

def run_batch_testing():
    image_dir = "Leaf"
    gt_dir = "Groundtruth"
    
    if not os.path.exists(image_dir):
        print(f"ERROR: {image_dir} not found!")
        return
    
    # Get all images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(file)
    
    image_files.sort()
    total_images = len(image_files)
    
    if total_images == 0:
        print("No images found!")
        return
    
    print(f"Found {total_images} images")
    print("Processing 10 images per batch...")
    
    batch_size = 10
    current_batch = 0
    total_batches = (total_images + batch_size - 1) // batch_size
    
    all_results = []
    
    while current_batch < total_batches:
        batch_start = current_batch * batch_size
        batch_end = min(batch_start + batch_size, total_images)
        batch_files = image_files[batch_start:batch_end]
        
        print(f"\n🔥 PROCESSING BATCH {current_batch + 1}/{total_batches} 🔥")
        print(f"Images {batch_start + 1} to {batch_end}")
        
        batch_results = []
        
        for i, img_file in enumerate(batch_files):
            img_path = os.path.join(image_dir, img_file)
            
            # Find GT
            base_name = os.path.splitext(img_file)[0]
            gt_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_gt = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(potential_gt):
                    gt_path = potential_gt
                    break
            
            # Process image
            result_data = test_single_image(img_path, gt_path)
            if result_data:
                batch_results.append((img_file, result_data))
                all_results.append((img_file, result_data))
                
                # Display result immediately
                display_single_result(img_file, result_data, batch_start + i + 1, total_images)
        
        # Batch summary
        print(f"\n{'='*60}")
        print(f"BATCH {current_batch + 1} SUMMARY")
        print(f"{'='*60}")
        
        valid_metrics = [data[1]['metrics'] for data in batch_results if data[1]['metrics']]
        
        if valid_metrics:
            avg_precision = np.mean([m[0] for m in valid_metrics])
            avg_recall = np.mean([m[1] for m in valid_metrics])
            avg_f1 = np.mean([m[2] for m in valid_metrics])
            avg_iou = np.mean([m[3] for m in valid_metrics])
            
            print(f"Batch Average Precision: {avg_precision:.4f}")
            print(f"Batch Average Recall: {avg_recall:.4f}")
            print(f"Batch Average F1: {avg_f1:.4f}")
            print(f"Batch Average IoU: {avg_iou:.4f}")
        
        avg_disease_pct = np.mean([data[1]['disease_percentage'] for data in batch_results])
        avg_time = np.mean([data[1]['processing_time'] for data in batch_results])
        
        print(f"Batch Average Disease %: {avg_disease_pct:.2f}%")
        print(f"Batch Average Processing Time: {avg_time:.4f}s")
        
        # Navigation
        if current_batch < total_batches - 1:
            print(f"\nBatch {current_batch + 1} completed.")
            choice = input("Press ENTER for next batch, 'q' to quit: ").strip().lower()
            if choice == 'q':
                break
            current_batch += 1
        else:
            print("\n🎉 ALL BATCHES COMPLETED! 🎉")
            break
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    all_valid_metrics = [data[1]['metrics'] for _, data in all_results if data['metrics']]
    
    if all_valid_metrics:
        final_precision = np.mean([m[0] for m in all_valid_metrics])
        final_recall = np.mean([m[1] for m in all_valid_metrics])
        final_f1 = np.mean([m[2] for m in all_valid_metrics])
        final_iou = np.mean([m[3] for m in all_valid_metrics])
        
        print(f"OVERALL AVERAGE PRECISION: {final_precision:.4f}")
        print(f"OVERALL AVERAGE RECALL: {final_recall:.4f}")
        print(f"OVERALL AVERAGE F1: {final_f1:.4f}")
        print(f"OVERALL AVERAGE IoU: {final_iou:.4f}")
        
        if final_f1 > 0.6:
            print("🏆 ALGORITHM PERFORMANCE: EXCELLENT")
        elif final_f1 > 0.4:
            print("🥈 ALGORITHM PERFORMANCE: GOOD")
        elif final_f1 > 0.2:
            print("🥉 ALGORITHM PERFORMANCE: NEEDS IMPROVEMENT")
        else:
            print("💀 ALGORITHM PERFORMANCE: TERRIBLE - NEEDS MAJOR FIXES")
    
    final_disease_pct = np.mean([data[1]['disease_percentage'] for _, data in all_results])
    final_time = np.mean([data[1]['processing_time'] for _, data in all_results])
    
    print(f"OVERALL AVERAGE DISEASE %: {final_disease_pct:.2f}%")
    print(f"OVERALL AVERAGE PROCESSING TIME: {final_time:.4f}s")
    print(f"TOTAL IMAGES PROCESSED: {len(all_results)}")

if __name__ == "__main__":
    print("🔥 LEAF DISEASE SEGMENTATION TESTING 🔥")
    print("Using imageSegment.py with APPLY_FINAL_INVERSION = False")
    print("="*60)
    
    run_batch_testing()