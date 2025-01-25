import os
import numpy as np
import cv2
import json
from PIL import Image
from transformers import pipeline
from megadetector.detection.run_detector_batch import load_and_run_detector_batch

def main():
    # ---------------------------------------------------
    # 1. Determine script directory and set up paths
    # ---------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the test image (TestImage/TestImage.jpg)
    test_image_path = os.path.join(script_dir, "TestImage", "TestImage.jpg")
    
    # Create a results folder in the script's root directory
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Model paths (Update if needed)
    md_model_path = "MDV5A"  # MegaDetector model directory or filename
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"  # Depth model checkpoint
    
    # Known distance setup
    known_x, known_y = 579, 213     # Known pixel coordinates in the image
    known_distance = 3.91            # Known real-world distance in meters

    # ---------------------------------------------------
    # 2. Load and prepare the image
    # ---------------------------------------------------
    # For depth estimation pipeline (requires PIL Image)
    image_pil = Image.open(test_image_path).convert("RGB")
    
    # Also load via cv2 for bounding box drawing, etc.
    image_cv = cv2.imread(test_image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Could not find or open the image at {test_image_path}.")
    orig_height, orig_width = image_cv.shape[:2]
    
    # ---------------------------------------------------
    # 3. Run depth estimation
    # ---------------------------------------------------
    depth_pipe = pipeline("depth-estimation", model=checkpoint)
    prediction = depth_pipe(image_pil)
    
    # Convert the predicted depth to a NumPy array
    depth_map = np.array(prediction["depth"], dtype=np.float32)
    
    # ---------------------------------------------------
    # 4. Verify and Correct Depth Map Orientation
    # ---------------------------------------------------
    # Depth value at the known pixel (NOTE: must ensure the coordinates are within image bounds)
    if not (0 <= known_x < depth_map.shape[1] and 0 <= known_y < depth_map.shape[0]):
        raise ValueError("Known distance pixel coordinates are out of image bounds.")
    
    known_depth_value = depth_map[known_y, known_x]
    
    # Debug: Print known depth value
    print(f"[DEBUG] Known Depth Value (raw): {known_depth_value}")
    
    # Sample points to determine depth orientation
    sample_far_x, sample_far_y = 0, 0  # Top-left corner (assumed far)
    sample_near_x, sample_near_y = orig_width - 1, orig_height - 1  # Bottom-right corner (assumed near)
    sample_depth_far = depth_map[sample_far_y, sample_far_x]
    sample_depth_near = depth_map[sample_near_y, sample_near_x]
    
    print(f"[DEBUG] Sample Depth Far (top-left): {sample_depth_far}")
    print(f"[DEBUG] Sample Depth Near (bottom-right): {sample_depth_near}")
    
    # Determine if inversion is needed
    # Assuming that in the scene, top-left is far and bottom-right is near
    # If depth at near sample > depth at far sample, inversion is needed
    if sample_depth_near > sample_depth_far:
        print("[INFO] Inverting depth map as closer objects have larger depth values.")
        depth_map = 1.0 / (depth_map + 1e-8)  # Invert depth map
        known_depth_value = depth_map[known_y, known_x]
        print(f"[DEBUG] Known Depth Value after inversion: {known_depth_value}")
    else:
        print("[INFO] Depth map orientation is correct; no inversion needed.")
    
    # ---------------------------------------------------
    # 5. Convert depth map to real-world distances
    # ---------------------------------------------------
    scaling_factor = known_distance / known_depth_value  # distance / raw_depth
    real_world_depth_map = depth_map * scaling_factor  # Convert entire depth map
    
    # ---------------------------------------------------
    # 6. Run MegaDetector to find objects
    # ---------------------------------------------------
    # MegaDetector function expects a list of image paths
    results = load_and_run_detector_batch(md_model_path, [test_image_path])
    
    # Save results to a JSON file
    md_results_path = os.path.join(results_dir, "megadetector_results.json")
    with open(md_results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] MegaDetector results saved to {md_results_path}")
    
    # Extract bounding boxes from results
    bounding_boxes = results[0]["detections"]
    
    # ---------------------------------------------------
    # 7. Save "Original_image.jpg"
    # ---------------------------------------------------
    original_image_path = os.path.join(results_dir, "Original_image.jpg")
    cv2.imwrite(original_image_path, image_cv)
    print(f"[INFO] Saved Original Image to {original_image_path}")
    
    # ---------------------------------------------------
    # 8. Create "Image_with_BoundingBox.jpg"
    #    (draw the first valid animal/human bounding box)
    # ---------------------------------------------------
    image_with_box = image_cv.copy()
    chosen_box = None
    
    for bbox in bounding_boxes:
        # "1" in MegaDetector typically indicates "animal"
        # "2" indicates "person", etc. Adjust if needed.
        if bbox["category"] not in ["1", "2"]: 
            continue
        
        # Bbox format is [x, y, w, h] in normalized coords
        x, y, w, h = bbox["bbox"]
        x1 = int(x * orig_width)
        y1 = int(y * orig_height)
        x2 = int((x + w) * orig_width)
        y2 = int((y + h) * orig_height)
        
        # Draw bounding box
        color = (0, 0, 255)  # Red box
        thickness = 2
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), color, thickness)
        
        chosen_box = (x1, y1, x2, y2)
        break  # Only process the first bounding box with correct category
    
    image_with_box_path = os.path.join(results_dir, "Image_with_BoundingBox.jpg")
    cv2.imwrite(image_with_box_path, image_with_box)
    print(f"[INFO] Saved Image with Bounding Box to {image_with_box_path}")
    
    # ---------------------------------------------------
    # 9. Create "Depth_Map_image.jpg"
    #    (visualize depth map using a color map)
    # ---------------------------------------------------
    # Define fixed depth range for color mapping (in meters)
    min_depth = 0.0    # Adjust based on your scene
    max_depth = 10.0   # Adjust based on your scene
    
    # Clip the real_world_depth_map to the defined range
    real_world_depth_map_clipped = np.clip(real_world_depth_map, min_depth, max_depth)
    
    # Normalize depth map based on fixed range
    depth_normalized = ((real_world_depth_map_clipped - min_depth) / (max_depth - min_depth))
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    
    # Apply color mapping
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
    
    # Save depth map with colormap
    depth_map_image_path = os.path.join(results_dir, "Depth_Map_image.jpg")
    cv2.imwrite(depth_map_image_path, depth_colormap)
    print(f"[INFO] Saved Depth Map image to {depth_map_image_path}")
    
    # ---------------------------------------------------
    # 10. Create "Depth_Map_Image_With_BoundingBox.jpg"
    # ---------------------------------------------------
    depth_map_with_box = depth_colormap.copy()
    if chosen_box is not None:
        (x1, y1, x2, y2) = chosen_box
        cv2.rectangle(depth_map_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    depth_map_with_box_path = os.path.join(results_dir, "Depth_Map_Image_With_BoundingBox.jpg")
    cv2.imwrite(depth_map_with_box_path, depth_map_with_box)
    print(f"[INFO] Saved Depth Map Image with Bounding Box to {depth_map_with_box_path}")
    
    # ---------------------------------------------------
    # 11. Create "ResultImage.jpg"
    #     - Mark known-distance pixel in green
    #     - Mark bounding box center in red
    #     - Annotate distances
    # ---------------------------------------------------
    result_image = image_cv.copy()
    
    # 11.a. Draw known distance location in GREEN
    known_color = (0, 255, 0)  # Green
    cv2.circle(result_image, (known_x, known_y), 8, known_color, -1)
    cv2.putText(
        result_image, 
        f"Known Dist: {known_distance:.2f}m", 
        (known_x + 10, known_y + 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        known_color, 
        2
    )
    
    # 11.b. If bounding box was found, mark center in RED
    if chosen_box is not None:
        (x1, y1, x2, y2) = chosen_box
        
        # Use the bounding box center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Depth value at bounding box center (in real-world depth map)
        # Ensure coordinates are within bounds
        if 0 <= center_x < real_world_depth_map.shape[1] and 0 <= center_y < real_world_depth_map.shape[0]:
            distance_to_object_center = real_world_depth_map[center_y, center_x]
        else:
            distance_to_object_center = np.nan
        
        # Draw red circle and distance
        object_color = (0, 0, 255)  # Red
        cv2.circle(result_image, (center_x, center_y), 8, object_color, -1)
        
        if not np.isnan(distance_to_object_center):
            distance_text = f"Obj Dist: {distance_to_object_center:.2f}m"
        else:
            distance_text = "Obj Dist: N/A"
        
        cv2.putText(
            result_image, 
            distance_text, 
            (center_x + 10, center_y + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            object_color, 
            2
        )
    
    # Save the final result image
    result_image_path = os.path.join(results_dir, "ResultImage.jpg")
    cv2.imwrite(result_image_path, result_image)
    print(f"[INFO] Saved Result Image to {result_image_path}")

if __name__ == "__main__":
    main()
