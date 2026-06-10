import logging

import cv2
import numpy as np
from deepface import DeepFace

from .email_feedback import EmailFeedback

class Liveness:
    def __init__(self): 
        pass

    
    def check_liveness(self, frame):
        try:
            # Convert frame format if needed
            if isinstance(frame, np.ndarray):
                # DeepFace expects image in proper format
                faces = DeepFace.extract_faces(
                frame,
                enforce_detection=False,
                detector_backend='opencv',
                anti_spoofing=True
            )
                
                if faces:
                    for face in faces:
                        if 'is_real' in face:
                            return True if face['is_real'] else False
                    return True  # Default to real if no anti-spoofing info
                else:
                    return "No face detected"
            else:
                return "Invalid frame format"
                
        except Exception as e:
            logging.error(f"Liveness check failed: {str(e)}")
            EmailFeedback.compose_email("Error", f"Liveness check failed: {str(e)}")
            return "Error"
    def detect_smile(self, frame):
        pass
    def detect_blink(self, frame):
        pass
    