import cv2
import numpy as np
from insightface.app import MaskRenderer
from pathlib import Path
import logging
import sys
import traceback
from typing import Optional, Union
import logging
from email_feedback import EmailFeedback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = 'antelopev2'
DET_SIZE = (384, 384)
DEFAULT_SCALE_FACTOR = 1.15

class FaceRenderer:
    def __init__(self):
        """Initialize FaceRenderer with default settings."""
        self.mask_renderer: Optional[MaskRenderer] = None
        self.glasses_img: Optional[np.ndarray] = None
        self.DEFAULT_SCALE_FACTOR = DEFAULT_SCALE_FACTOR
        self.SCALE_BY_FACE_WIDTH = True
        self.model_root = str(Path(__file__).parent.parent  / 'models' / '.insightface')


    def initialize_mask_renderer(self, ctx_id: int = -1) -> None:
        try:
            self.mask_renderer =MaskRenderer(name=MODEL_NAME, root=self.model_root)
            self.mask_renderer.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
            logger.info("MaskRenderer initialized successfully")
        except Exception as e:
            error_msg = f"MaskRenderer initialization failed: {str(e)}"
            logger.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e

    def set_glasses_image(self, glasses_path: Union[str, Path]) -> None:
        try:
            path = Path(glasses_path)
            if not path.exists():
                raise FileNotFoundError(f"Glasses image not found at {glasses_path}")
                
            self.glasses_img= cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if self.glasses_img is None:
                raise ValueError(f"Failed to load image from {glasses_path}")
                
            logger.info(f"Successfully loaded glasses image from {glasses_path}")
            
        except Exception as e:
            error_msg = f"Failed to set glasses image: {str(e)}"
            logger.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise

    def overlay_glasses(
        self,
        img: np.ndarray,
        landmarks: np.ndarray,
        scale_factor: Optional[float] = None,
        offset: int = 10
    ) -> np.ndarray:
        try:
            # Validate inputs
            if self.glasses_img is None:
                raise ValueError("Glasses image not loaded")
            if not isinstance(img, np.ndarray) or img.size == 0:
                raise ValueError("Invalid input image")
            if landmarks.shape != (68, 2):
                raise ValueError("Landmarks must be 68x2 array")
                
            scale_factor = scale_factor or self.DEFAULT_SCALE_FACTOR
            
            # Calculate eye positions
            left_eye = np.mean(landmarks[36:42], axis=0).astype(int)
            right_eye = np.mean(landmarks[42:48], axis=0).astype(int)
            
            # Calculate glasses size
            if self.SCALE_BY_FACE_WIDTH:
                face_w = landmarks[:, 0].max() - landmarks[:, 0].min()
                target_w = int(face_w * scale_factor)
            else:
                target_w = int(np.linalg.norm(left_eye - right_eye) * scale_factor)
                
            # Resize glasses
            scale = target_w / self.glasses_img.shape[1]
            glasses = cv2.resize(
                self.glasses_img, 
                None, 
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            
            # Calculate position
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            nose_y = landmarks[27][1]
            glasses_y = int((eye_center_y + nose_y) / 2 - glasses.shape[0] / 2) + offset
            glasses_x = int((left_eye[0] + right_eye[0]) / 2 - glasses.shape[1] / 2)
            
            # Calculate safe overlay region
            x1, y1 = max(glasses_x, 0), max(glasses_y, 0)
            x2 = min(glasses_x + glasses.shape[1], img.shape[1])
            y2 = min(glasses_y + glasses.shape[0], img.shape[0])
            
            # Check for valid region
            if x2 <= x1 or y2 <= y1:
                logger.warning("Glasses outside image bounds")
                return img
                
            # Extract regions for blending
            glasses_region = glasses[
                y1-glasses_y : y2-glasses_y,
                x1-glasses_x : x2-glasses_x
            ]
            img_region = img[y1:y2, x1:x2]
            
            # Alpha blending
            alpha = glasses_region[:, :, 3:] / 255.0
            img[y1:y2, x1:x2] = (
                alpha * glasses_region[:, :, :3] + 
                (1 - alpha) * img_region
            ).astype(np.uint8)
            
            return img
            
        except Exception as e:
            error_msg = f"Glasses overlay failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)

            raise RuntimeError(error_msg) from e

    def render_mask(self, image: np.ndarray, mask_image: str = "mask_blue") -> np.ndarray:
        try:
            if self.mask_renderer is None:
                raise RuntimeError("MaskRenderer not initialized")
                
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError("Invalid input image")
                
            params = self.mask_renderer.build_params(image)
            return self.mask_renderer.render_mask(image, mask_image, params)
            
        except Exception as e:
            error_msg = f"Mask rendering failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e

    def add_glasses(self, image: np.ndarray, face) -> np.ndarray:
        try:
            if self.glasses_img is None:
                raise ValueError("Glasses image not loaded")
                
            if not hasattr(face, 'landmark_3d_68'):
                raise ValueError("Face object missing required landmarks")
                
            landmarks = face.landmark_3d_68[:, :2].astype(int)
            return self.overlay_glasses(image, landmarks)
            
        except Exception as e:
            error_msg = f"Failed to add glasses: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e