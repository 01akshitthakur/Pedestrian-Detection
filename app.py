import streamlit as st
import cv2
import numpy as np
import imutils
from tempfile import NamedTemporaryFile
import time
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load YOLO model
labelsPath = "coco.names"
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

if not os.path.exists(labelsPath) or not os.path.exists(weights_path) or not os.path.exists(config_path):
    st.error("Model files are missing. Ensure that coco.names, yolov4-tiny.weights, and yolov4-tiny.cfg are in the current directory.")
    st.stop()

LABELS = open(labelsPath).read().strip().split("\n")

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
    tfile.close()

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

        # Convert the frame to RGB before displaying
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Sleep to control the frame rate
        time.sleep(0)

    cap.release()
    os.remove(tfile.name)  # Clean up temporary file
