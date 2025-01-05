import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Open the webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,  # Lower confidence threshold for smoother detection
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape  # Get frame dimensions

                # Get landmarks for inner corners of the eyes
                left_eye = face_landmarks.landmark[133]
                right_eye = face_landmarks.landmark[362]

                # Convert landmarks to pixel coordinates
                left_eye_coords = (int(left_eye.x * iw), int(left_eye.y * ih))
                right_eye_coords = (int(right_eye.x * iw), int(right_eye.y * ih))

                # Calculate the center and angle between the eyes
                eye_center = (
                    (left_eye_coords[0] + right_eye_coords[0]) // 2,
                    (left_eye_coords[1] + right_eye_coords[1]) // 2,
                )
                angle = np.degrees(np.arctan2(
                    right_eye_coords[1] - left_eye_coords[1],
                    right_eye_coords[0] - left_eye_coords[0]
                ))

                # Define bar dimensions (slightly wider and slimmer)
                bar_width = int(3.5 * np.linalg.norm(
                    np.array(right_eye_coords) - np.array(left_eye_coords)
                ))  # Slightly wider
                bar_height = 80  # Slightly slimmer

                # Define the rectangle points (rotated)
                box = cv2.boxPoints(((eye_center[0], eye_center[1]), (bar_width, bar_height), angle))
                box = np.int0(box)  # Convert to integer

                # Draw the filled rectangle
                cv2.drawContours(frame, [box], 0, (0, 0, 0), -1)

        # Display the result
        cv2.imshow("Eye Mask - Refined Bar", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

