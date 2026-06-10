import insightface
import numpy as np
from typing import Dict 
import time
import logging 
from email_feedback import EmailFeedback
from detect_faces import FaceDetector
from db import Database
from encrypt import encrypt
# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])


threshold: float = 0.5

class Recognize:
    def __init__(self, image, app, security_manager: encrypt):
        self.image=image
        self.db = Database()
        self.security_manager = security_manager
        self.app=app
        try:
            self.mysql_conn, self.mysql_cursor = self.db.initialize_mysql()
        except Exception as e:
            logging.critical(f"Database initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize database connection")
    
    def recognize(self, emb1, emb2):
        try:
            if not isinstance(emb1, np.ndarray) or not isinstance(emb2, np.ndarray):
                raise ValueError("Both inputs must be numpy arrays")
            
            if emb1.shape != emb2.shape:
                raise ValueError("Embeddings must have the same dimensions")
            
            # Compute cosine similarity (dot product for normalized vectors)
            similarity = np.dot(emb1, emb2)
            
            logging.debug(f"Embedding comparison similarity: {similarity:.4f}")
            
            return similarity
        
        except Exception as e:
            logging.error(f"Embedding recognition failed: {str(e)}")
            return None
        

    def recognize_face_authorization(self, user_id: str ) -> Dict:
        """
        Recognize a single face for authorization by comparing it against a stored embedding.
        Uses MySQL for secure storage of encrypted embeddings.
        """
        try:
            if not user_id or not isinstance(user_id, str):
                raise ValueError("Invalid user ID format")
            logging.info(f"Starting authorization for user: {user_id}")
            
            # Query MySQL for stored user data
            self.mysql_cursor.execute(
                "SELECT embedding, nonce, hmac FROM users WHERE user_id = %s",
                (user_id,)
            )
            user_data = self.mysql_cursor.fetchone()

            
            if not user_data:
                error_msg=f"Unregistered user attempt: {user_id}"
                logging.error(error_msg)
                EmailFeedback.compose_email("Error", error_msg)
                return {
                    "status": "error", 
                    "code": "UNREGISTERED_USER",
                    "message": "User not registered"
                }
            stored_embedding, nonce, hmac = user_data
            
            try:
                faces = FaceDetector(self.image, self.app).detect_faces()  # Assumes detect_faces returns one best face
                face = faces[0]
                
                embedding = face.normed_embedding
                try:
                    decrypted_embedding = self.security_manager.decrypt_embedding(nonce, stored_embedding, hmac)
                except ValueError as ve:
                    error_msg=f"Embedding decryption failed: {str(ve)}"
                    logging.error(error_msg)
                    EmailFeedback.compose_email("Error", error_msg)
                    return {
                        "status": "error",
                        "code": "DECRYPTION_FAILED",
                        "message": "Embedding decryption error"
                    }
                
                similarity = np.dot(decrypted_embedding, embedding)
                status = "authorized" if similarity >= threshold else "unauthorized"
                
                # Log to MySQL authorization_logs
                self.mysql_cursor.execute("""
                    INSERT INTO authorization_logs (user_id, similarity, det_score, model_version, timestamp, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    float(similarity),
                    float(face.det_score),
                    insightface.__version__,
                    int(time.time()),
                    status
                ))
                self.mysql_conn.commit()

                
                logging.info(f"Authorization result for {user_id}: {status} (similarity: {similarity:.4f})")
                return {
                    "status": "success",
                    "result": status,
                    "similarity": round(similarity, 4),
                    "confidence": f"{similarity*100:.2f}%"
                }
            
            except Exception as e:
                error_msg=f"ERROR: Authorization processing failed: {str(e)}"
                logging.error(error_msg)
                EmailFeedback.compose_email("Error", error_msg)
                return {"status": "error", "code": "PROCESSING_ERROR", "message":"Authorization failure"}
        
        except Exception as e:
            error_msg=f"Authorization workflow failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return {
                "status": "error",
                "code": "SYSTEM_ERROR",
                "message": "Internal system error"
            }
