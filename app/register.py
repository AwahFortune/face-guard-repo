import logging
import insightface
import numpy as np
from typing import Dict 
import time
import logging 
from pymilvus import (
    connections
)
from email_feedback import EmailFeedback
from detect_faces import FaceDetector
from encrypt import encrypt

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

class FaceRegistration:
    def __init__(self, app, security_manager: encrypt, mysql_conn, mysql_cursor, collection):
        self.app=app
        self.security_manager = security_manager
        self.mysql_conn=mysql_conn
        self.mysql_cursor=mysql_cursor
        self.collection=collection

    def register_face(self, user_id: str, image: np.ndarray) -> Dict:

        try:
            if not user_id or not isinstance(user_id, str):
                raise ValueError("Invalid user ID format")
            
            logging.info(f"Starting face registration for user: {user_id}")
            faces = FaceDetector(image, self.app).detect_faces()
            face = faces[0]
            embedding = face.normed_embedding 
            if np.linalg.norm(embedding) < 0.05:
                raise ValueError("Low quality embedding")
            # For surveillance (Milvus): Store raw embedding
            surveillance_data = {
                "user_id": user_id,
                "embedding": embedding.tolist(),  # Raw embedding as list of floats
                "det_score": float(face.det_score),
                "model_version": insightface.__version__,
                "registration_time": int(time.time())
            }
            self.collection.insert([surveillance_data])
            self.collection.flush()
                         
            # For authorization (MySQL): Store encrypted embedding
            nonce, encrypted, mac = self.security_manager.encrypt_embedding(embedding)
            self.mysql_cursor.execute("""
                INSERT INTO users (user_id, embedding, nonce, hmac, det_score, model_version, registration_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, encrypted, nonce, mac, float(face.det_score), insightface.__version__, int(time.time())))
            self.mysql_conn.commit()
            logging.info(
                f"Registration successful for {user_id}.\nDet score: {face.det_score:.4f}"
            )
            EmailFeedback.compose_email("Success", f"Registration successful for {user_id}.\nDet score: {face.det_score:.4f}")

            return {"status": "success", "user_id": user_id}
        
        except Exception as e:
            error_msg = f"Registration failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return {
                "status": "error",
                "message": "Internal system error",
                "code": "SYSTEM_ERROR"
            } 

