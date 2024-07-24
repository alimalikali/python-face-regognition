from flask import Flask, request, jsonify, send_from_directory
import face_recognition
import cv2
import numpy as np
import requests
from io import BytesIO
from pymongo import MongoClient
import gridfs

app = Flask(__name__)

# MongoDB configuration
MONGO_URI = 'mongodb://localhost:27017' 
client = MongoClient(MONGO_URI)
db = client['FaceDB']
fs = gridfs.GridFS(db)

# Global variables for storing known faces
known_face_encodings = []
known_face_names = []
known_face_ids = []  # Add this line

def load_known_faces():
    global known_face_encodings, known_face_names, known_face_ids
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    # Retrieve all images from MongoDB
    files = fs.find()
    for file in files:
        img_data = file.read()
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            name = file.filename.split('.')[0]
            face_id = str(file._id)  # Use MongoDB ObjectId as the face ID

            for encoding in encodings:
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                known_face_ids.append(face_id)  # Append the ID here

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
            return jsonify({'faces': []})

        unique_faces = {}

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_id = None
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                face_id = known_face_ids[first_match_index]
            if name not in unique_faces:
                unique_faces[name] = face_id

        # Convert the dictionary to a list of dicts with name and id
        result = [{'name': name, 'id': unique_faces[name]} for name in unique_faces]

        return jsonify({'faces': result})

    except Exception as e:
        # Print the error to the console for debugging
        print(f"Error in /recognize endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

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

            # Save the image to MongoDB
            filename = f"{name}.jpg"
            file_id = fs.put(response.content, filename=filename)

            # Reload known faces
            load_known_faces()

            return jsonify({'message': 'Image uploaded successfully.', 'id': str(file_id)}), 200
        else:
            return jsonify({'error': 'Invalid input'}), 400

    except Exception as e:
        # Print the error to the console for debugging
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
