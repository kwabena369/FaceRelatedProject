# Face-Related Computer Vision Tasks

This repository is dedicated to exploring various computer vision tasks related to faces. It will include multiple projects, each focusing on a specific face-related task, such as smile detection, emotion recognition, facial landmark identification, and more. The first implemented project in this series is `SmileNow` for real-time smile detection.

## Current Task: SmileNow

`SmileNow` is a computer vision project designed to detect smiles in real-time using a webcam. This project leverages the OpenCV library to process video feeds, identify faces, and highlight smiles.

### Features
- Real-time detection of faces and smiles.
- Uses Haar Cascade Classifiers for face and smile detection.
- Processes video frames efficiently by converting them to grayscale for faster computation.
- Highlights detected smiles with bounding rectangles.

### How It Works
The `SmileNow` function captures video from your webcam and processes each frame to:
1. Detect faces using the `haarcascade_frontalface_default.xml` classifier.
2. Detect smiles within the identified face regions using the `haarcascade_smile.xml` classifier.
3. Draw rectangles around detected faces and smiles for visual feedback.

#### Key Functional Highlights
- **Grayscale Conversion**: Converts each video frame to grayscale for efficient processing.
- **Image Slicing**: Extracts face regions from the grayscale image for targeted smile detection.
- **Multi-Scale Detection**: Uses Haar Cascade parameters like `scaleFactor` and `minNeighbors` to fine-tune detection.

#### Code Walkthrough
Here is a brief overview of the key steps in the `SmileNow` function:

```python
# Converting the frame to grayscale
Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Detecting faces in the frame
detections = frontline_Harcase.detectMultiScale(
    Gray,
    scaleFactor=1.3,
    minNeighbors=5,
    minSize=(30, 30)
)

# Processing each detected face
for (x, y, w, h) in detections:
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 255), 2)

    # Extracting face region for smile detection
    Gray_slice = Gray[y:y+h, x:x+w]

    # Detecting smiles in the face region
    smile_detected = Smile_Harcase.detectMultiScale(
        Gray_slice,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(25, 25)
    )

    for (sx, sy, sw, sh) in smile_detected:
        cv.rectangle(
            frame,
            (x + sx, y + sy),
            (x + sx + sw, y + sy + sh),
            (0, 250, 200),
            2
        )
```

### Running the Code
To run the `SmileNow` function, follow these steps:

1. Install OpenCV:
   ```bash
   pip install opencv-python
   ```

2. Ensure the Haar Cascade XML files are available in the specified path (`../XmlFile/`).

3. Run the script:
   ```bash
   python smile_detection.py
   ```

4. Press `q` to exit the video feed.

## Future Tasks
This repository will be expanded to include additional face-related tasks, such as:
- **Emotion Recognition**: Detecting and classifying emotions from facial expressions.
- **Facial Landmark Detection**: Identifying key facial landmarks such as eyes, nose, and mouth.
- **Face Recognition**: Identifying and verifying individuals based on facial features.

## Visual Demo
Here’s a conceptual illustration of the smile detection process:

1. **Face Detection**
   - Bounding rectangles highlight detected faces.

2. **Smile Detection**
   - Smaller rectangles highlight detected smiles within the face region.

### Example Visualization
Below is an example image generated to show the face and smile detection process:

