import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

# Load pre-trained facial recognition model from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to compute face encodings from an image
def compute_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)
    if len(faces) == 1:
        landmarks = shape_predictor(rgb_image, faces[0])
        return np.array(face_recognizer.compute_face_descriptor(rgb_image, landmarks))
    else:
        return None

# Function to compare face encodings and check for similarity
def compare_faces(encodings1, encodings2, threshold=0.6):
    encodings1 = np.array(encodings1)
    encodings2 = np.array(encodings2)
    distances = np.linalg.norm(encodings1 - encodings2, axis=1)
    return distances < threshold

# Create a directory to save face encodings if it doesn't exist
if not os.path.exists("saved_encodings"):
    os.makedirs("saved_encodings")

# Create a directory to save similar faces if it doesn't exist
if not os.path.exists("similar_faces"):
    os.makedirs("similar_faces")

# Initialize an empty list to store similar face names
similar_face_names = []

# Load captured face encodings and names
captured_face_encodings = []
captured_face_names = []

# Provide the path to the folder containing images to search for similar faces
search_folder = "Testing face_reg"

# Iterate over images in the search folder
for filename in tqdm(os.listdir(search_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(search_folder, filename)
        image = cv2.imread(filepath)
        face_encoding = compute_face_encoding(image)
        if face_encoding is not None:
            # Save the face encoding to a file
            encoding_file = os.path.join("saved_encodings", f"{filename.split('.')[0]}_encoding.npy")
            np.save(encoding_file, face_encoding)

# Load saved face encodings and names
for encoding_file in os.listdir("saved_encodings"):
    if encoding_file.endswith(".npy"):
        encoding_path = os.path.join("saved_encodings", encoding_file)
        face_encoding = np.load(encoding_path)
        captured_face_encodings.append(face_encoding)
        captured_face_names.append(encoding_file.split("_")[0])  # Extract the name from the file name

# Iterate over images in the search folder again to find similar faces
for filename in tqdm(os.listdir(search_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(search_folder, filename)
        image = cv2.imread(filepath)
        face_encoding = compute_face_encoding(image)
        if face_encoding is not None:
            # Compare the face encoding with saved face encodings
            for i, captured_encoding in enumerate(captured_face_encodings):
                match = compare_faces([captured_encoding], [face_encoding])
                if match:
                    similar_face_names.append(filename)
                    # Save the similar face to the separate folder
                    similar_face_path = os.path.join("similar_faces", filename)
                    cv2.imwrite(similar_face_path, image)
                    break  # Once a match is found, break out of 5the loop

# Print the similar face names
print("Similar faces found:", similar_face_names)
