import cv2
import numpy as np
import mediapipe as mp
from collections import deque

#  setting up the almight mediapipe 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize drawing super powers for making some kind of mesh on the face 
mp_drawing = mp.solutions.drawing_utils
drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#  the getting of the webcam
cap = cv2.VideoCapture(0)

#  the deque for storing the previouse position of the nose
nose_positions = deque(maxlen=10)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert BGR image to RGB that becos for  processing  by the mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect faces
    results = face_mesh.process(image_rgb)

    # Convert back to BGR for display
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_specs,
                drawing_specs
            )

            # Get nose landmark (tip of nose is point 1)
            nose_y = face_landmarks.landmark[1].y
            nose_positions.append(nose_y)

            # Detect nodding if we have enough positions
            if len(nose_positions) == nose_positions.maxlen:
                # Calculate movement
                movement = np.diff(list(nose_positions))
                avg_movement = np.mean(movement[-3:])  # Look at last 3 movements

                # Determine direction and magnitude of movement
                threshold = 0.01  # Adjust this value based on your needs
                if abs(avg_movement) > threshold:
                    if avg_movement > 0:
                        direction = "DOWN"
                        color = (0, 0, 255)  # Red
                    else:
                        direction = "UP"
                        color = (0, 255, 0)  # Green
                    
                    # Display direction
                    cv2.putText(image, f"Moving: {direction}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the image
    cv2.imshow('Head Nod Detection', image)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
face_mesh.close()