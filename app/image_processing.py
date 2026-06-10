import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage import exposure

from .detect_faces import FaceDetector
from .email_feedback import EmailFeedback

class DCENet(nn.Module):
    """Original Zero-DCE network for low-light enhancement (matching enhance_net_nopool)."""
    def __init__(self):
        super(DCENet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        x_r = torch.clamp(x_r, min=-0.2, max=0.2)  # Clamp to prevent overexposure artifacts
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        enhanced = x + r1 * (torch.pow(x, 2) - x)
        enhanced = enhanced + r2 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r3 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r4 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r5 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r6 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r7 * (torch.pow(enhanced, 2) - enhanced)
        enhanced = enhanced + r8 * (torch.pow(enhanced, 2) - enhanced)

        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8],1)

        return enhanced, enhanced, r  # Return to match 3 outputs; use first or second for enhanced

class ImageProcessor:
    def __init__(self, app, blur_threshold: float = 40, 
                 bright_low: float = 0.2,
                 bright_high: float = 0.8,
                 min_contrast_range: int = 20,
                 min_height: int = 112,
                 min_width: int = 112,
                 gamma_correction: float = 0.8,
                 low_light_threshold: float = 0.3,
                 dce_model_path: str = str(Path(__file__).parent.parent  / 'models' /"Epoch99.pth"),
                 crop_face: bool = True):
        self.blur_threshold = blur_threshold
        self.bright_low = bright_low
        self.bright_high = bright_high
        self.min_contrast_range = min_contrast_range
        self.min_height = min_height
        self.min_width = min_width
        self.gamma_correction = gamma_correction
        self.low_light_threshold = low_light_threshold
        self.crop_face = crop_face
        self.face_app = app
        
        # Initialize Zero-DCE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dce_net = DCENet().to(self.device)
        self.dce_net.eval()
        
        if dce_model_path:
            try:
                self.dce_net.load_state_dict(torch.load(dce_model_path, map_location=self.device))
                logging.info("Loaded pre-trained Zero-DCE model")
            except Exception as e:
                logging.error(f"Failed to load Zero-DCE model: {str(e)}. Ensure weights match DCENet architecture.")
                EmailFeedback.compose_email("Error", f"Failed to load Zero-DCE model: {str(e)}")
                raise RuntimeError(f"Model loading failed: {str(e)}")

    def detect_and_crop_face(self, image: np.ndarray) -> np.ndarray:
        try:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image            
            face = FaceDetector(bgr_image, self.face_app).detect_faces()[0]
            bbox = face.bbox.astype(int)
            
            if not self.crop_face:
                return image
            
            height, width = image.shape[:2]
            pad_x = int((bbox[2] - bbox[0]) * 0.1)
            pad_y = int((bbox[3] - bbox[1]) * 0.1)
            x1, y1, x2, y2 = max(0, bbox[0] - pad_x), max(0, bbox[1] - pad_y), min(width, bbox[2] + pad_x), min(height, bbox[3] + pad_y)
            
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                raise ValueError("Cropped face is empty")
            
            logging.info(f"Face cropped: {cropped.shape}")
            return cropped, [face]
        
        except Exception as e:
            error_msg = f"Face detection/cropping failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return image, []

    def enhance_contrast(self, image: np.ndarray):
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            contrast_level = np.std(l)
            clip_limit = 3.0 if contrast_level < 30 else 1.5
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            # Add unsharp mask for sharpening details
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            return unsharp
        except Exception as e:
            error_msg = f"Contrast enhancement failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg)

    def check_blur(self, image: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            tenengrad_var = np.var(gradient_magnitude)
            logging.info(f"Tenengrad variance: {tenengrad_var}")
            return tenengrad_var > self.blur_threshold
        except Exception as e:
            logging.error(f"Blur detection failed: {str(e)}")
            EmailFeedback.compose_email("Error", f"Blur detection failed: {str(e)}")
            return False

    def check_contrast(self, image: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hist, _ = np.histogram(gray.ravel(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]
            low_value = np.searchsorted(cdf_normalized, self.bright_low)
            high_value = np.searchsorted(cdf_normalized, self.bright_high)
            contrast_range = high_value - low_value
            # Add Michelson contrast for better perceptual evaluation
            min_val, max_val = np.min(gray), np.max(gray)
            michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-5)  # Avoid division by zero
            return contrast_range >= self.min_contrast_range and michelson_contrast >= 0.1
        except Exception as e:
            logging.error(f"Contrast check failed: {str(e)}")
            EmailFeedback.compose_email("Error", f"Contrast check failed: {str(e)}")
            return False

    def check_low_light(self, image: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_bright = np.mean(gray) / 255.0
            return mean_bright < self.low_light_threshold
        except Exception as e:
            logging.error(f"Low-light check failed: {str(e)}")
            return False

    def low_light_enhance(self, image: np.ndarray) -> np.ndarray:
        try:
            # Preprocess to match test code: normalize to [0,1]
            img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, enhanced_image, _ = self.dce_net(img_tensor)  # Use second output to match test code
            enhanced_np = (enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # Add denoising to reduce artifacts in enhanced low-light faces
            denoised = cv2.fastNlMeansDenoisingColored(enhanced_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            logging.info("Applied Zero-DCE low-light enhancement")
            return denoised
        except Exception as e:
            error_msg = f"Low-light enhancement failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return image

    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            mean_bright = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) / 255.0
            adaptive_gamma = np.log(0.5) / np.log(mean_bright + 1e-5) if mean_bright > 0 else self.gamma_correction
            gamma_corrected = exposure.adjust_gamma(image, gamma=adaptive_gamma)
            yuv = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        except (ValueError, cv2.error) as e:
            logging.warning(f"Brightness normalization skipped: {str(e)}")
            return image
        except Exception as e:
            error_msg = f"Critical normalization failure: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return image

    def check_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        try:
            if len(image.shape) < 3:
                return False, "Image must be color (RGB)"
            height, width = image.shape[:2]
            if height < self.min_height or width < self.min_width:
                return False, f"Image too small: {width}x{height}, minimum: {self.min_width}x{self.min_height}"
            if not self.check_blur(image):
                return False, f"Image too blurry (Tenengrad variance below {self.blur_threshold})"
            if not self.check_contrast(image):
                return False, f"Insufficient contrast (range below {self.min_contrast_range})"
            return True, "Quality checks passed"
        except Exception as e:
            error_msg = f"Quality check failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            return False, error_msg

    def process_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image.copy()
            
            processed, face = self.detect_and_crop_face(image_rgb)
            
            is_good, reason = self.check_quality(processed)
            if is_good:
                logging.info("Image passed initial quality checks")
                return processed, face
            
            logging.info(f"Image needs processing: {reason}")
            
            if self.check_low_light(processed):
                processed = self.low_light_enhance(processed)
            
            processed = self.enhance_contrast(processed)
            
            # processed = self.normalize_brightness(processed)
            
            is_good_final, reason_final = self.check_quality(processed)
            if not is_good_final:
                error_msg = f"Image quality insufficient after processing: {reason_final}"
                logging.warning(error_msg)
                raise ValueError(error_msg)
            
            logging.info("Image processing completed successfully")
            return processed, face
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Image processing failed: {str(e)}"
            logging.error(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg) from e