import cv2
import dlib
import numpy as np
import os

# Load pre-trained facial recognition model from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Create a directory to save captured faces if it doesn't exist
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize an empty list to store captured face encodings and their corresponding names
captured_face_encodings = []
captured_face_names = []

while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR color (OpenCV) to RGB color (dlib)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector(rgb_frame)

    for face in faces:
        # Get facial landmarks
        landmarks = shape_predictor(rgb_frame, face)

        # Compute face encodings
        face_encoding = np.array(face_recognizer.compute_face_descriptor(rgb_frame, landmarks))

        # Compare face encoding with captured face encodings
        match_found = False
        for i, captured_encoding in enumerate(captured_face_encodings):
            distance = np.linalg.norm(captured_encoding - face_encoding)
            if distance < 0.6:  # Adjust this threshold as needed
                match_found = True
                break

        if not match_found:
            # If face doesn't match any captured face, add it to captured faces
            captured_face_encodings.append(face_encoding)

            # Save the captured face to a separate folder
            captured_face_path = os.path.join("captured_faces", f"captured_face_{len(captured_face_encodings)}.jpg")
            cv2.imwrite(captured_face_path, frame)

            print(f"New face captured. Image saved to {captured_face_path}")

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()

# Provide the path to the folder containing images to search for similar faces
search_folder = "Testing face_reg"

# Now you can perform face recognition with the captured faces and the images in the search folder
# You can use techniques like face embedding comparison or any other face recognition algorithm to search for similar faces.
# Please note that implementing face recognition algorithms goes beyond the scope of this code snippet.
