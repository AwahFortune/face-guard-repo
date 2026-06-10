import logging
from glasses_detector import GlassesClassifier
from email_feedback import EmailFeedback
class EyeglassClassifier:
    """
    A classifier to detect if a person is wearing eyeglasses, sunglasses,
    or no glasses at all.
    """
    def __init__(self):
        try:
            self.eyeglasses_classifier = GlassesClassifier(kind='eyeglasses')
            self.sunglasses_classifier = GlassesClassifier(kind='sunglasses')
        except Exception as e:
            logging.error(f"Failed to initialize GlassesClassifier: {e}")
            raise
    
    def detect(self, image):
        """
        Takes an image and returns the classification status.
        Returns: "sunglasses", "transparent", or "no-glasses".
        """
        try:
            eyeglasses_pred = self.eyeglasses_classifier.predict(image, format='bool')
            sunglasses_pred = self.sunglasses_classifier.predict(image, format='bool')
            # Apply logic for three-class classification
            if sunglasses_pred:
                return "sunglasses"
            elif eyeglasses_pred:
                return "transparent"
            else:
                return "no-glasses"
        except Exception as e:
            logging.error(f"Eyeglass detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Eyeglass detection failed: {e}")
            return "error"
