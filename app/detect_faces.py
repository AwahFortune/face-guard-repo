import logging

import numpy as np

from .email_feedback import EmailFeedback

confidence_threshold = 0.5
quality_threshold = 0.5

class FaceDetector():
    def __init__(self, image: np.ndarray, app):
        self.image = image
        self.app = app
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold

    def detect_faces(self) -> list:
        """
        Robust face detection with quality checks and null safety.
        Returns a list of confident face objects (empty if none or multiple faces).
        """
        try:
            faces = self.app.get(self.image)
            if not faces:
                logging.warning("No faces were detected in the image.")
                return []

            # Filter with null safety and quality checks
            confident_faces = []
            for f in faces:
                # Ensure detection score exists and meets threshold
                if not hasattr(f, 'det_score') or f.det_score is None:
                    continue
                if f.det_score < self.confidence_threshold:
                    continue
                # Handle potential missing face_quality attribute
                quality = getattr(f, 'face_quality', 1.0)
                if quality is None:
                    quality = 1.0  # Default to maximum quality if missing
                    
                if quality > self.quality_threshold:  # Quality threshold
                    confident_faces.append(f)
            if confident_faces:
                face = confident_faces[0]
            num_confident_faces = len(confident_faces)
            if num_confident_faces == 0:
                logging.warning(f"Faces were found, but none met the confidence ({self.confidence_threshold}) or quality ({self.quality_threshold}) threshold.")
                return []
            elif num_confident_faces > 1:
                logging.warning(f"Multiple faces ({num_confident_faces}) detected, but only one is required.")
                return []
            else:
                face = confident_faces[0]
                logging.info(f"Successfully detected one face with score: {face.det_score:.2f}")
                return [face]  # Return the actual face object

        except Exception as e:
            error_msg = f"Face detection failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return []