import os
import glob
import time
import json

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from megadetector.detection.run_detector_batch import load_and_run_detector_batch

def get_click_with_countdown(
    original_image,
    max_disp_width=1280,
    max_disp_height=720,
    countdown_secs=3
):
    """
    Displays (optionally scaled) `original_image` in a window ("SelectPixel").
    User clicks on a pixel -> a circle is drawn at that location.
    A 3-second countdown begins, displayed as text "Saving position in 3... 2... 1...".
    If the user clicks again during the countdown, it resets (back to 3 seconds) & moves the circle.
    After 3 seconds with no new clicks, the window closes automatically, and we return (orig_x, orig_y).

    If user presses ESC before a click, or if no click ever happens, we return None.
    """

    orig_h, orig_w = original_image.shape[:2]
    scale_w = max_disp_width / float(orig_w)
    scale_h = max_disp_height / float(orig_h)
    scale = min(scale_w, scale_h, 1.0)  # Only shrink if bigger than max
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    display_image = cv2.resize(original_image, (disp_w, disp_h))

    clicked_point = None
    last_click_time = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_point, last_click_time
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            last_click_time = time.time()

    window_name = "SelectPixel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        temp_img = display_image.copy()

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to cancel
            clicked_point = None
            break

        if clicked_point is not None:
            # Draw circle
            cv2.circle(temp_img, clicked_point, 8, (0, 255, 0), -1)
            elapsed = time.time() - last_click_time
            remaining = countdown_secs - elapsed
            if remaining <= 0:
                # Countdown finished
                break
            else:
                text = f"Saving position in {int(np.ceil(remaining))}..."
                cv2.putText(
                    temp_img,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )
        else:
            instr = "Left-click a pixel to select. ESC to cancel."
            cv2.putText(
                temp_img,
                instr,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

        cv2.imshow(window_name, temp_img)

    cv2.destroyWindow(window_name)

    if clicked_point is None:
        return None

    # Convert scaled to original coords
    scaled_x, scaled_y = clicked_point
    orig_x = int(scaled_x / scale)
    orig_y = int(scaled_y / scale)

    return (orig_x, orig_y)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_dir = os.path.join(script_dir, "TestImage")

    # Gather images by extension
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    images_to_process = []
    for ext in image_extensions:
        images_to_process.extend(glob.glob(os.path.join(test_image_dir, ext)))

    if not images_to_process:
        raise FileNotFoundError(f"No images found in {test_image_dir}")

    # Create results folder
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Model paths
    md_model_path = "MDV5A"  # Make sure this points to your MegaDetector model
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"

    # Load depth model pipeline once
    print("[INFO] Loading depth model pipeline (CPU)...")
    depth_pipe = pipeline("depth-estimation", model=checkpoint, device="cpu")

    for i, test_image_path in enumerate(images_to_process, start=1):
        print(f"\n[INFO] Processing image {i}/{len(images_to_process)}: {test_image_path}")

        # Load image via OpenCV
        image_cv = cv2.imread(test_image_path)
        if image_cv is None:
            print(f"[WARNING] Could not open {test_image_path}, skipping.")
            continue

        # 1) Let user pick pixel with countdown
        clicked_coords = get_click_with_countdown(
            image_cv,
            max_disp_width=1280,
            max_disp_height=720,
            countdown_secs=3
        )
        if clicked_coords is None:
            print("[WARNING] No pixel selected (or user canceled). Skipping image.")
            continue

        known_x, known_y = clicked_coords
        print(f"[INFO] User selected pixel: ({known_x}, {known_y})")

        # 2) Ask user for known distance in console
        known_distance_str = input(
            f"Enter the real-world distance (meters) for pixel at ({known_x}, {known_y}): "
        )
        try:
            known_distance = float(known_distance_str)
        except ValueError:
            print("[ERROR] Invalid distance. Skipping this image.")
            continue

        # 3) Depth Estimation
        image_pil = Image.open(test_image_path).convert("RGB")
        prediction = depth_pipe(image_pil)
        depth_map = np.array(prediction["depth"], dtype=np.float32)

        orig_height, orig_width = image_cv.shape[:2]
        if not (0 <= known_x < depth_map.shape[1] and 0 <= known_y < depth_map.shape[0]):
            print(f"[WARNING] Pixel ({known_x},{known_y}) out of depth_map bounds. Skipping.")
            continue

        known_depth_value = depth_map[known_y, known_x]
        print(f"[DEBUG] Known Depth Value (raw): {known_depth_value}")

        # Orientation check
        sample_far_x, sample_far_y = 0, 0
        sample_near_x = min(orig_width - 1, depth_map.shape[1] - 1)
        sample_near_y = min(orig_height - 1, depth_map.shape[0] - 1)
        sample_depth_far = depth_map[sample_far_y, sample_far_x]
        sample_depth_near = depth_map[sample_near_y, sample_near_x]
        print(f"[DEBUG] Sample Depth Far (top-left): {sample_depth_far}")
        print(f"[DEBUG] Sample Depth Near (bottom-right): {sample_depth_near}")

        # If near is larger => invert
        if sample_depth_near > sample_depth_far:
            print("[INFO] Inverting depth map (closer => smaller).")
            depth_map = 1.0 / (depth_map + 1e-8)
            known_depth_value = depth_map[known_y, known_x]

        # Convert entire depth map to real distances
        scaling_factor = known_distance / known_depth_value
        real_world_depth_map = depth_map * scaling_factor

        # 4) MegaDetector
        md_results = load_and_run_detector_batch(md_model_path, [test_image_path])
        bounding_boxes = md_results[0]["detections"]

        # Save results JSON
        md_results_path = os.path.join(results_dir, f"megadetector_results_{i}.json")
        with open(md_results_path, "w") as f:
            json.dump(md_results, f, indent=4)
        print(f"[INFO] MegaDetector results saved to {md_results_path}")

        # 5) Save Original image
        original_image_path = os.path.join(results_dir, f"Original_image_{i}.jpg")
        cv2.imwrite(original_image_path, image_cv)
        print(f"[INFO] Saved Original Image to {original_image_path}")

        # 6) Identify highest-confidence detection for categories "1" or "2"
        image_with_box = image_cv.copy()
        chosen_box = None
        valid_boxes = [bbox for bbox in bounding_boxes if bbox["category"] in ["1", "2"]]

        if valid_boxes:
            # Pick the detection with max "conf"
            best_detection = max(valid_boxes, key=lambda b: b["conf"])
            x, y, w, h = best_detection["bbox"]
            x1 = int(x * orig_width)
            y1 = int(y * orig_height)
            x2 = int((x + w) * orig_width)
            y2 = int((y + h) * orig_height)
            chosen_box = (x1, y1, x2, y2)
            cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"[INFO] Chosen box from category {best_detection['category']}, conf={best_detection['conf']:.3f}")
        else:
            print("[INFO] No valid detections (category '1' or '2') found.")

        image_with_box_path = os.path.join(results_dir, f"Image_with_BoundingBox_{i}.jpg")
        cv2.imwrite(image_with_box_path, image_with_box)
        print(f"[INFO] Saved Image with Bounding Box to {image_with_box_path}")

        # 7) Create a depth color map
        min_depth, max_depth = 0.0, 10.0
        real_world_depth_map_clipped = np.clip(real_world_depth_map, min_depth, max_depth)
        depth_normalized = (real_world_depth_map_clipped - min_depth) / (max_depth - min_depth)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

        depth_map_image_path = os.path.join(results_dir, f"Depth_Map_image_{i}.jpg")
        cv2.imwrite(depth_map_image_path, depth_colormap)
        print(f"[INFO] Saved Depth Map image to {depth_map_image_path}")

        # 8) Depth map with bounding box
        depth_map_with_box = depth_colormap.copy()
        if chosen_box is not None:
            (x1, y1, x2, y2) = chosen_box
            cv2.rectangle(depth_map_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2)

        depth_map_with_box_path = os.path.join(results_dir, f"Depth_Map_Image_With_BoundingBox_{i}.jpg")
        cv2.imwrite(depth_map_with_box_path, depth_map_with_box)
        print(f"[INFO] Saved Depth Map Image with Bounding Box to {depth_map_with_box_path}")

        # 9) Final "ResultImage" with known dist + box center
        result_image = image_cv.copy()

        # Mark known distance
        known_color = (0, 255, 0)
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

        # If we have a chosen box, mark center in red
        if chosen_box is not None:
            (x1, y1, x2, y2) = chosen_box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if 0 <= center_x < real_world_depth_map.shape[1] and 0 <= center_y < real_world_depth_map.shape[0]:
                obj_dist = real_world_depth_map[center_y, center_x]
            else:
                obj_dist = np.nan

            object_color = (0, 0, 255)
            cv2.circle(result_image, (center_x, center_y), 8, object_color, -1)
            if not np.isnan(obj_dist):
                dist_text = f"Obj Dist: {obj_dist:.2f}m"
            else:
                dist_text = "Obj Dist: N/A"

            cv2.putText(
                result_image,
                dist_text,
                (center_x + 10, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                object_color,
                2
            )

        result_image_path = os.path.join(results_dir, f"ResultImage_{i}.jpg")
        cv2.imwrite(result_image_path, result_image)
        print(f"[INFO] Saved Result Image to {result_image_path}")

    cv2.destroyAllWindows()
    print("\n[INFO] Done processing all images.")

if __name__ == "__main__":
    main()
