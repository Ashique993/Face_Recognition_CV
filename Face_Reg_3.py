import cv2
import dlib
import numpy as np
import os

# Load pre-trained facial recognition model from dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to compute face encoding from an image
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

# Create a directory to save similar faces if it doesn't exist
if not os.path.exists("similar_faces"):
    os.makedirs("similar_faces")

# Load captured face encodings and names
captured_face_encodings = []
captured_face_names = []

# Load saved face encodings and names
for encoding_file in os.listdir("saved_encodings"):
    if encoding_file.endswith(".npy"):
        encoding_path = os.path.join("saved_encodings", encoding_file)
        face_encoding = np.load(encoding_path)
        captured_face_encodings.append(face_encoding)
        captured_face_names.append(encoding_file.split("_")[0])  # Extract the name from the file name

# Capture or load image
image_source = input("Enter 'webcam' to capture an image or 'file' to load an image file: ")

if image_source.lower() == "webcam":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    image = frame
elif image_source.lower() == "file":
    file_path = input("Enter the path to the image file: ")
    image = cv2.imread(file_path)
else:
    print("Invalid input. Exiting...")
    exit()

# Compute face encoding for the captured/loaded image
face_encoding = compute_face_encoding(image)

# Provide the path to the folder containing images to search for similar faces
search_folder = input("Enter the path to the folder containing images to search for similar faces: ")

# Iterate over images in the search folder
for filename in os.listdir(search_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(search_folder, filename)
        image = cv2.imread(filepath)
        face_encoding = compute_face_encoding(image)
        if face_encoding is not None:
            similar_face_names = []
            # Compare the face encoding with saved face encodings
            for i, captured_encoding in enumerate(captured_face_encodings):
                match = compare_faces([captured_encoding], [face_encoding])
                if match:
                    similar_face_names.append(captured_face_names[i])
                    # Save the similar face to the separate folder
                    similar_face_path = os.path.join("similar_faces", f"{captured_face_names[i]}.jpg")
                    cv2.imwrite(similar_face_path, image)

            # Print the similar face names
            print(f"Similar faces found for {filename}:", similar_face_names)