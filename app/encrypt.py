import hashlib
import hmac
import logging
import os
from typing import Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .email_feedback import EmailFeedback

# Security constants
HMAC_DIGEST_SIZE = 32  # Standard size for HMAC-SHA256
AES_NONCE_SIZE = 12     # Recommended nonce size for AES-GCM
ANONYMIZATION_DIGEST_SIZE = 16  # Size for pseudonymous IDs


class encrypt: 
    def __init__(self):
        self.aesgcm=""
        self.AES_KEY=bytes.fromhex(os.getenv("AES_KEY", ""))
        self.HMAC_KEY=bytes.fromhex(os.getenv("HMAC_KEY", ""))
        self.ANONYMIZATION_KEY=bytes.fromhex(os.getenv("ANONYMIZATION_KEY", ""))
    def initialize_security(self):
        try:

            # Validate key presence
            if not all([self.AES_KEY, self.HMAC_KEY, self.ANONYMIZATION_KEY]):
                raise ValueError("Missing required encryption keys in environment variables")
            
            # Validate key lengths
            if len(self.AES_KEY) not in [16, 24, 32]:
                raise ValueError("AES key must be 16, 24, or 32 bytes")
            if len(self.HMAC_KEY) < 32:
                raise ValueError("HMAC key must be at least 32 bytes")
                
            # Initialize AES-GCM cipher
            self.aesgcm = AESGCM(self.AES_KEY)
            
        except Exception as e:
            error_msg = f"Security initialization failed: {str(e)}"
            logging.critical(error_msg)
            raise RuntimeError(error_msg) from e
        
    def encrypt_embedding(self, embedding: np.ndarray) -> Tuple[bytes, bytes, bytes]:
        """Encrypt face embedding with authenticated encryption"""
        if not isinstance(embedding, np.ndarray) or embedding.dtype != np.float32:
            raise ValueError("Invalid embedding format")
        
        try:
            nonce = os.urandom(AES_NONCE_SIZE)
            if self.aesgcm is None:
                raise RuntimeError("Encryptor not initialised — call initialize_security() first")
            ciphertext = self.aesgcm.encrypt(nonce, embedding.tobytes(), None)
            mac = hmac.new(self.HMAC_KEY, ciphertext, hashlib.sha256).digest()
            return nonce, ciphertext, mac
        except Exception as e:
            error_msg = f"Embedding encryption failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise ValueError(error_msg) from e

    def decrypt_embedding(self, nonce: bytes, ciphertext: bytes, mac: bytes) -> np.ndarray:
        """Decrypt and verify face embedding"""
        try:
            expected_mac = hmac.new(self.HMAC_KEY, ciphertext, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected_mac):
                raise ValueError("HMAC verification failed")
            decrypted = self.aesgcm.decrypt(nonce, ciphertext, None)
            embedding = np.frombuffer(decrypted, dtype=np.float32)
            # Validate embedding dimensions
            if embedding.shape[0] != 512:
                raise ValueError("Invalid embedding dimensions")
            return embedding
        except Exception as e:
            error_msg = f"Embedding decryption failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise ValueError(error_msg) from e
        
    def anonymize_user(self, user_id: str) -> str:
        """Generate pseudonymous user identifier"""
        try:
            return hashlib.blake2b(
                user_id.encode(),
                key=self.ANONYMIZATION_KEY,
                digest_size=ANONYMIZATION_DIGEST_SIZE
            ).hexdigest()
        except Exception as e:
            error_msg = f"User anonymization failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise ValueError(error_msg) from e