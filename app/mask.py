"pip install opencv-python numpy tensorflow==2.13.1"
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging
import sys
from pathlib import Path
from email_feedback import EmailFeedback
face_model_dir="face_detector"
mask_model_path=str(Path(__file__).parent.parent / 'models' /"mask_detector.model")
confidence=0.5

class MaskDetector:
    def __init__(self):
        # Load face detection model
        prototxtPath = str(Path(__file__).parent.parent / 'models' /"deploy.prototxt")
        weightsPath = str(Path(__file__).parent.parent / 'models' /"res10_300x300_ssd_iter_140000.caffemodel")
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # Load mask detection model
        self.maskNet = load_model(mask_model_path)

        # Set confidence threshold
        self.confidence = confidence

    def detect_mask(self, image):
        try:
            # Grab image dimensions
            (h, w) = image.shape[:2]

            # Construct blob from image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Pass blob through face detection network
            self.faceNet.setInput(blob)
            detections = self.faceNet.forward()

            # Loop over detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence:
                    # Compute bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure bounding box falls within image dimensions
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # Extract face ROI
                    face = image[startY:endY, startX:endX]
                    if face.size == 0:
                        continue

                    # Preprocess face for mask detection
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # Predict mask presence
                    (mask, withoutMask) = self.maskNet.predict(face)[0]
                    if mask > withoutMask:
                        return True  # Mask detected

            return False  # No mask detected
        except Exception as e:
            logging.error(f"Mask detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Mask detection failed: {e}")
            return False