# Face Recognition App

This project is a Flask-based web application for face recognition. It allows you to upload images and recognize faces using the `face_recognition` library.

## Features

- Upload face images from URLs to the known faces directory.
- Recognize faces in images provided via URLs.
- Display results of recognized faces.

## Requirements

- Python 3.6+
- Flask
- face_recognition
- OpenCV
- NumPy
- Requests

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a directory named `known_faces` in the root of your project:

    ```bash
    mkdir known_faces
    ```

5. Save the code in `app.py` as `app.py` in the root of your project:

   
6. Create a `requirements.txt` file with the following content:

    ```txt
    Flask
    face_recognition
    opencv-python
    numpy
    requests
    ```

7. Create a directory named `static` in the root of your project and add an `index.html` file inside it for the frontend.

8. Run the application:

    ```bash
    python app.py
    ```

## API Endpoints

### Upload Image

- **URL:** `/upload`
- **Method:** `POST`
- **Description:** Upload an image URL and associate it with a name for face recognition.
- **Request Body:** `application/x-www-form-urlencoded`
    - `image_url` (string): The URL of the image.
    - `name` (string): The name to associate with the face.

- **Response:**
    - `200 OK`: `{'message': 'Image uploaded successfully.'}`
    - `400 Bad Request`: `{'error': 'Invalid input'}`
    - `500 Internal Server Error`: `{'error': 'Error message'}`

### Recognize Face

- **URL:** `/recognize`
- **Method:** `POST`
- **Description:** Recognize faces in an image provided via URL.
- **Request Body:** `application/x-www-form-urlencoded`
    - `image_url` (string): The URL of the image.

- **Response:**
    - `200 OK`: `{'names': ['name1', 'name2', ...]}`
    - `400 Bad Request`: `{'error': 'Image could not be fetched.'}`
    - `500 Internal Server Error`: `{'error': 'Error message'}`

## Frontend

Create an HTML file named `index.html` in the `static` directory to serve as the frontend for the application. The frontend should allow users to input an image URL and a name for uploading images, as well as to recognize faces from an image URL.

