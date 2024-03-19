import cv2
import dlib
import numpy as np

# Load pre-trained facial recognition model from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load database of known faces (you need to have this prepared beforehand)
known_face_encodings = np.load("known_face_encodings.npy")
known_face_names = np.load("known_face_names.npy")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

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

        # Compare face encoding with known face encodings
        matches = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        min_distance_index = np.argmin(matches)
        min_distance = matches[min_distance_index]

        if min_distance < 0.6:  # Adjust this threshold as needed
            name = known_face_names[min_distance_index]
        else:
            name = "Unknown"

        # Draw rectangle around the face
        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw label with name
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
