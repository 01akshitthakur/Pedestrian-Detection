

#run this command in terminal to run the app:
#streamlit run app.py --server.enableXsrfProtection false


import streamlit as st
import cv2 
import numpy as np
import imutils
from tempfile import NamedTemporaryFile
import requests
import time

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)

    return results

# Fetching LABELS from GitHub
labels_url = 'https://raw.githubusercontent.com/01akshitthakur/Pedestrian-Detection/main/coco.names'
response_labels = requests.get(labels_url)

if response_labels.status_code == 200:
    # Successfully fetched the file contents
    LABELS = response_labels.text.strip().split("\n")
else:
    # Handle the case where fetching failed
    st.error(f"Failed to fetch '{labels_url}'. Status code: {response_labels.status_code}")
    st.stop()

# Fetching weights_path from GitHub
weights_url = 'https://github.com/01akshitthakur/Pedestrian-Detection/raw/main/yolov4-tiny.weights'
weights_response = requests.get(weights_url)

if weights_response.status_code == 200:
    # Save weights to a temporary file
    weights_temp_file = NamedTemporaryFile(delete=False)
    weights_temp_file.write(weights_response.content)
    weights_path = weights_temp_file.name
else:
    st.error(f"Failed to fetch '{weights_url}'. Status code: {weights_response.status_code}")
    st.stop()

# Fetching config_path from GitHub
config_url = 'https://raw.githubusercontent.com/01akshitthakur/Pedestrian-Detection/main/yolov4-tiny.cfg'
config_response = requests.get(config_url)

if config_response.status_code == 200:
    # Successfully fetched the file contents
    config_temp_file = NamedTemporaryFile(delete=False)
    config_temp_file.write(config_response.content)
    config_path = config_temp_file.name
else:
    # Handle the case where fetching failed
    st.error(f"Failed to fetch '{config_url}'. Status code: {config_response.status_code}")
    st.stop()

# Load YOLO model
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# Uncomment the following lines if you have CUDA installed
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

st.title("Pedestrian Detection Streamlit App")
st.subheader("Upload a video file to detect pedestrians")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=700)
        results = pedestrian_detection(frame, model, layer_name, personidz=LABELS.index("person"))

        for res in results:
            cv2.rectangle(frame, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

        stframe.image(frame, channels="BGR", use_column_width=True)

        time.sleep(0.03)  # To control the frame rate

    cap.release()



   


   
