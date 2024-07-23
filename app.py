from flask import Flask, request, jsonify, send_from_directory
import face_recognition
import cv2
import numpy as np
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Directory containing known face images
KNOWN_FACES_DIR = 'known_faces'

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    # Iterate over files in the known_faces directory
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            # Use all encodings found in the image
            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as the name

load_known_faces()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        image_url = request.form['image_url']
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Image could not be fetched.'}), 400

        # Convert the image data to a numpy array
        np_img = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if not face_encodings:
            return jsonify({'names': []})

        unique_faces = {}

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            if name not in unique_faces:
                unique_faces[name] = 1

        return jsonify({'names': list(unique_faces.keys())})

    except Exception as e:
        # Print the error to the console for debugging
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Get the image URL and the name for the face
        image_url = request.form['image_url']
        name = request.form['name']

        if image_url and name:
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({'error': 'Image could not be fetched.'}), 400

            # Convert the image data to a numpy array
            np_img = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'error': 'Invalid image data'}), 400

            # Save the image to the known_faces directory
            filename = f"{name}.jpg"
            file_path = os.path.join(KNOWN_FACES_DIR, filename)
            cv2.imwrite(file_path, img)

            # Reload known faces
            load_known_faces()

            return jsonify({'message': 'Image uploaded successfully.'}), 200
        else:
            return jsonify({'error': 'Invalid input'}), 400

    except Exception as e:
        # Print the error to the console for debugging
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
