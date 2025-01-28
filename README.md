# MonocularWildlifeDepthEstimation

## Welcome! 👋
Hello everyone who ended up here!

This is a short test script to estimate the depth to an animal (or person) from monocular wildcamera images.

**Note:** This code is not maintained, and there are no plans for further development.

---

## Overview
This script takes one input image and performs the following steps:
1. Uses MegaDetectorV5 to detect an animal or human in the image.
2. Generates a depth map of the image using the model provided at [Hugging Face's Monocular Depth Estimation documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/monocular_depth_estimation.md).
3. Calculates the depth to the detected object based on a known distance in the image.

You are free to use this code, but please note:
- The Ultralytics module used for MegaDetectorV5 and potentially the depth estimation models may be under licenses that restrict commercial use or similar activities. Check their respective licenses before usage.

---

## Getting Started

### Step 1: Create a Virtual Environment
Navigate to the root folder of this repository in your terminal and create a virtual environment:

- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **Mac/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 2: Install Dependencies
Once in the virtual environment, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Example Code
You can now run the example script! Be aware that loading and running the models may take some time.

To test the script with your own images, you will need to:
1. **Put image(s) into TestImage folder** The code looks for valid images (.jpg, .png and so on..) under the TestImage folder.
2. **Provide a known distance possition:** There will be a window with the image(s) where you press where you know the distance. 
3. **Provide a the known distance to the selected pixel in terminal (meters) :** Enter the distance in the terminal, use "." as seperator, like 5.12 (for 5 meters and 12 centimeters) and press enter,
4. **Find results:** The image(s) should be saved into the results folder under the project, here you can see the original image, where the animal was found, depth image and results with estimated distances.


## What Does the Code Do?

1. **Object Detection:**
   - Runs MegaDetectorV5 to detect a single animal or human in the image with the highest confidence.

2. **Depth Map Generation:**
   - Uses the Depth Anything model to create a depth map for the image.

3. **Depth Conversion:**
   - Converts the depth map values to actual measurements based on the known distance.

4. **Distance Estimation:**
   - Calculates the distance to the detected object (animal/human) using the center pixels of the bounding box.

---

## Delimitations

### Limitations:
- If the detected object is in front of the "known" distance point, the estimation will be inaccurate.
  - **Solution:** Use multiple known distances in the image as references.
- If the bounding box center pixels do not include the detected object (e.g., due to a gap in the box), the estimation may be incorrect.
- The depth estimation relies on a CNN trained on specific datasets and may not always be accurate.

---

## Future Work
> While I only spent a Sunday afternoon putting this example together, here are some ideas for extending it:

1. **Testing on Wildlife Data:**
   - Set up test parameters and validate the approach with actual wildlife images.

2. **Multiple Reference Points:**
   - Implement calibration using three or more known distances in the image to improve accuracy and handle objects closer than the reference points.

3. **Segmentation-Based Depth:**
   - Segment the detected object from the depth map and calculate an average depth or use a path of pixels for a more robust distance estimation.

---

## Disclaimer
This code is provided as-is and is not intended for production use. Use at your own risk, and ensure compliance with any licensing restrictions of the libraries used.

Happy testing! 🎉

