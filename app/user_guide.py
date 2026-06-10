import numpy as np
import logging
from scipy.spatial import distance as dist
from collections import deque
from datetime import datetime

# Local project imports
from email_feedback import EmailFeedback
from detect_faces import FaceDetector

# Configurable defaults (tweak as needed)
EAR_THRESHOLD = 0.20
SLIDE_YAW_THRESHOLD = 8.0
SLIDE_PITCH_THRESHOLD = 6.0
GAZE_TOLERANCE_X = 0.08
GAZE_TOLERANCE_Y = 0.03
GAZE_SMOOTH_ALPHA = 0.3
MIN_FACE_RATIO = 0.25
MAX_FACE_RATIO = 0.50
OPTIMAL_FACE_RATIO = 0.35
MIN_DISTANCE_CM = 35.0
MAX_DISTANCE_CM = 75.0
OPTIMAL_DISTANCE_CM = 50.0
KNOWN_FACE_WIDTH_CM = 14.0
KNOWN_DISTANCE_CM = 60.0
PIX_WIDTH_AT_KNOWN_DIST = 210
FOCAL_LENGTH = (PIX_WIDTH_AT_KNOWN_DIST * KNOWN_DISTANCE_CM) / KNOWN_FACE_WIDTH_CM
POSE_HISTORY_LEN = 64
CENTER_HISTORY_LEN = 30
STEADY_THRESHOLD = 0.01
FACE_BOUNDARY_LANDMARKS = [10, 152, 356, 127, 1]
LEFT_IRIS_IDS = [474, 475, 476, 477]
RIGHT_IRIS_IDS = [469, 470, 471, 472]
LEFT_IRIS_CORNERS = [33, 133]
RIGHT_IRIS_CORNERS = [362, 263]
LEFT_EYE_EAR_IDS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDS = [362, 385, 387, 263, 373, 380]
CLOCKWISE_ORDER = ["Up", "Right", "Down", "Left"]
COUNTER_CLOCKWISE_ORDER = ["Up", "Left", "Down", "Right"]
ADJACENT_TRANSITIONS = {
    "Up": {"Right", "Left"},
    "Right": {"Up", "Down"},
    "Down": {"Right", "Left"},
    "Left": {"Up", "Down"}
}

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])



class Guide:
    """Core Guide class (no UI, no demo loop).

    Usage:
      - Create one Guide instance and call update_frame(frame) every loop.
      - Call facemesh_process() once per frame and pass result to gaze/eye checks.
    """

    def __init__(self, frame, facemesh, app):
        # frame: initial BGR frame (used for sizes), facemesh: MediaPipe FaceMesh instance
        # app: InsightFace FaceAnalysis instance (used for pose & bbox)
        self.frame = frame
        self.facemesh = facemesh
        self.app = app

        # initial face detection
        try:
            self.face = FaceDetector(frame, app).detect_faces()
        except Exception as e:
            logging.warning(f"Initial face detection failed: {e}")
            self.face = []

        # histories & smoothing
        self.pose_history = deque(maxlen=POSE_HISTORY_LEN)
        self.center_history = deque(maxlen=CENTER_HISTORY_LEN)
        self.direction_sequence = []
        self.completed_directions = set()
        self.gaze_ema = np.array([0.0, 0.0])

        # calibration
        self.neutral_yaw = None
        self.neutral_pitch = None

    # ------------------
    # frame update
    # ------------------
    def update_frame(self, frame):
        """Update internal frame and re-run face detection for the current frame.
        Keep a single Guide instance across the loop to preserve histories.
        """
        self.frame = frame
        try:
            self.face = FaceDetector(frame, self.app).detect_faces()
        except Exception as e:
            logging.warning(f"update_frame face detection warning: {e}")
            self.face = []

    # ------------------
    # MediaPipe FaceMesh
    # ------------------
    def facemesh_process(self):
        try:
            return self.facemesh.process(self.frame)
        except Exception as e:
            logging.error(f"FaceMesh processing failed: {e}")
            EmailFeedback.compose_email("Error", f"FaceMesh processing failed: {e}")
            return None

    # ------------------
    # face completeness
    # ------------------
    def is_complete_face_visible(self, res):
        try:
            if not res or not hasattr(res, 'multi_face_landmarks') or not res.multi_face_landmarks:
                return False, "No face detected"
            landmarks = res.multi_face_landmarks[0].landmark
            h, w = self.frame.shape[:2]
            margin = 0.05
            for idx in FACE_BOUNDARY_LANDMARKS:
                lm = landmarks[idx]
                if lm.x < margin or lm.x > (1 - margin):
                    return False, "Face too close to horizontal edge"
                if lm.y < margin or lm.y > (1 - margin):
                    return False, "Face too close to vertical edge"
            for idx in LEFT_EYE_EAR_IDS + RIGHT_EYE_EAR_IDS:
                lm = landmarks[idx]
                if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                    return False, "Eyes not fully visible"
            return True, "Face fully visible"
        except Exception as e:
            logging.error(f"Face visibility check failed: {e}")
            EmailFeedback.compose_email("Error", f"Face visibility check failed: {e}")
            return False, "Error checking face visibility"

    # ------------------
    # distance & size
    # ------------------
    def detect_distance_and_size(self):
        try:
            if not isinstance(self.face, list) or len(self.face) == 0 or self.face[0] is None:
                return {"distance_cm": None, "face_ratio": None, "status": "No Face", "guidance": "Position face in view"}
            x1, y1, x2, y2 = self.face[0].bbox.astype(int)
            face_w = max(1, x2 - x1)
            frame_w = max(1, self.frame.shape[1])
            dist_cm = (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / face_w
            ratio = face_w / frame_w
            if ratio < MIN_FACE_RATIO or dist_cm > MAX_DISTANCE_CM:
                status = "Too Far"; guidance = "Move closer"
            elif ratio > MAX_FACE_RATIO or dist_cm < MIN_DISTANCE_CM:
                status = "Too Close"; guidance = "Move away"
            elif abs(ratio - OPTIMAL_FACE_RATIO) < 0.05:
                status = "Perfect"; guidance = "Perfect distance"
            else:
                status = "OK"; guidance = "Good distance"
            return {"distance_cm": dist_cm, "face_ratio": ratio, "status": status, "guidance": guidance}
        except Exception as e:
            logging.error(f"Distance detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Distance detection failed: {e}")
            return {"distance_cm": None, "face_ratio": None, "status": "Error", "guidance": "Error detecting distance"}

    # Backwards-compatible alias (older tests may call this name)
    def detect_distance_and_spoof(self):
        info = self.detect_distance_and_size()
        return info.get("distance_cm"), info.get("face_ratio"), info.get("status")

    # ------------------
    # EAR / eyes
    # ------------------
    def calculate_ear(self, landmarks, eye_ids, img_w, img_h):
        try:
            pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_ids]
            A = dist.euclidean(pts[1], pts[5])
            B = dist.euclidean(pts[2], pts[4])
            C = dist.euclidean(pts[0], pts[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.0
        except Exception as e:
            logging.error(f"EAR calculation failed: {e}")
            EmailFeedback.compose_email("Error", f"EAR calculation failed: {e}")
            return 0.0

    def detect_eye_state(self, res):
        try:
            if not res or not hasattr(res, 'multi_face_landmarks') or not res.multi_face_landmarks:
                return "No Face", 0.0
            h, w = self.frame.shape[:2]
            lm = list(res.multi_face_landmarks[0].landmark)
            ear_l = self.calculate_ear(lm, LEFT_EYE_EAR_IDS, w, h)
            ear_r = self.calculate_ear(lm, RIGHT_EYE_EAR_IDS, w, h)
            ear = (ear_l + ear_r) / 2.0
            state = "Open" if ear > EAR_THRESHOLD else "Closed"
            return state, ear
        except Exception as e:
            logging.error(f"Eye state detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Eye state detection failed: {e}")
            return "Error", 0.0

    # ------------------
    # gaze
    # ------------------
    def compute_normalized_gaze_offset(self, landmarks, iris_ids, corner_ids):
        try:
            iris_pts = np.array([[landmarks[i].x, landmarks[i].y] for i in iris_ids])
            iris_center = iris_pts.mean(axis=0)
            c1 = np.array([landmarks[corner_ids[0]].x, landmarks[corner_ids[0]].y])
            c2 = np.array([landmarks[corner_ids[1]].x, landmarks[corner_ids[1]].y])
            eye_center = (c1 + c2) / 2.0
            eye_width = np.linalg.norm(c2 - c1)
            if eye_width <= 1e-6:
                return np.array([0.0, 0.0])
            return (iris_center - eye_center) / eye_width
        except Exception as e:
            logging.error(f"Gaze offset calculation failed: {e}")
            EmailFeedback.compose_email("Error", f"Gaze offset calculation failed: {e}")
            return np.array([0.0, 0.0])

    def detect_gaze_focus(self, res):
        try:
            if not res or not hasattr(res, 'multi_face_landmarks') or not res.multi_face_landmarks:
                return {"status": "No Face", "offset": (0.0, 0.0), "guidance": "Face not detected"}
            lm = list(res.multi_face_landmarks[0].landmark)
            off_r = self.compute_normalized_gaze_offset(lm, RIGHT_IRIS_IDS, RIGHT_IRIS_CORNERS)
            off_l = self.compute_normalized_gaze_offset(lm, LEFT_IRIS_IDS, LEFT_IRIS_CORNERS)
            avg_offset = (off_r + off_l) / 2.0
            self.gaze_ema = (1 - GAZE_SMOOTH_ALPHA) * self.gaze_ema + GAZE_SMOOTH_ALPHA * avg_offset
            gx, gy = float(self.gaze_ema[0]), float(self.gaze_ema[1])
            if abs(gx) <= GAZE_TOLERANCE_X and abs(gy) <= GAZE_TOLERANCE_Y:
                status = "Focused"; guidance = "Looking at camera"
            else:
                status = "Away"
                if abs(gx) > GAZE_TOLERANCE_X:
                    guidance = f"Look {'left' if gx > 0 else 'right'}"
                else:
                    guidance = f"Look {'down' if gy > 0 else 'up'}"
            return {"status": status, "offset": (gx, gy), "guidance": guidance}
        except Exception as e:
            logging.error(f"Gaze focus detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Gaze focus detection failed: {e}")
            return {"status": "Error", "offset": (0.0, 0.0), "guidance": "Error detecting gaze"}

    # ------------------
    # head pose
    # ------------------
    def detect_head_pose(self):
        try:
            if not isinstance(self.face, list) or len(self.face) == 0 or self.face[0] is None:
                return {"direction": "No Face", "yaw": 0.0, "pitch": 0.0, "guidance": "Face not detected"}
            yaw, pitch, _ = self.face[0].pose
            yaw_rel = float(yaw); pitch_rel = float(pitch)
            if abs(yaw_rel) < SLIDE_YAW_THRESHOLD and abs(pitch_rel) < SLIDE_PITCH_THRESHOLD:
                direction = "Forward"; guidance = "Looking forward"
            elif abs(pitch_rel) >= abs(yaw_rel):
                if pitch_rel > SLIDE_PITCH_THRESHOLD:
                    direction, guidance = "Left", "Head left"
                else:
                    direction, guidance = "Right", "Head right"
            else:
                if yaw_rel > SLIDE_YAW_THRESHOLD:
                    direction, guidance = "Up", "Head up"
                else:
                    direction, guidance = "Down", "Head down"
            return {"direction": direction, "yaw": yaw_rel, "pitch": pitch_rel, "guidance": guidance}
        except Exception as e:
            logging.error(f"Head pose detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Head pose detection failed: {e}")
            return {"direction": "Error", "yaw": 0.0, "pitch": 0.0, "guidance": "Error"}

    # ------------------
    # steadiness
    # ------------------
    # def is_face_steady(self):
    #     try:
    #         if not isinstance(self.face, list) or len(self.face) == 0 or self.face[0] is None:
    #             return False, 1.0
    #         x1, y1, x2, y2 = self.face[0].bbox.astype(int)
    #         cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    #         self.center_history.append((cx, cy))
    #         if len(self.center_history) < self.center_history.maxlen:
    #             return False, 1.0
    #         arr = np.array(self.center_history)
    #         std = arr.std(axis=0)
    #         frame_w, frame_h = self.frame.shape[1], self.frame.shape[0]
    #         normalized_movement = (std[0] / frame_w + std[1] / frame_h) / 2.0
    #         is_steady = normalized_movement < STEADY_THRESHOLD
    #         return is_steady, normalized_movement
    #     except Exception as e:
    #         logging.error(f"Steadiness check failed: {e}")
    #         EmailFeedback.compose_email("Error", f"Steadiness check failed: {e}")
    #         return False, 1.0

    # ------------------
    # circular motion
    # ------------------
    def detect_circular_motion(self):
        try:
            pose = self.detect_head_pose()
            current_dir = pose["direction"]
            if current_dir not in {"Up", "Down", "Left", "Right"}:
                return {"detected": False, "pattern": None, "progress": list(self.completed_directions), "guidance": "Turn head to start"}
            if not self.direction_sequence:
                self.direction_sequence = [current_dir]
                self.completed_directions = {current_dir}
            else:
                last_dir = self.direction_sequence[-1]
                if current_dir != last_dir and current_dir in ADJACENT_TRANSITIONS.get(last_dir, set()):
                    self.direction_sequence.append(current_dir)
                    self.completed_directions.add(current_dir)
                elif current_dir != last_dir:
                    self.direction_sequence = [current_dir]
                    self.completed_directions = {current_dir}
            if len(self.completed_directions) == 4:
                seq = self.direction_sequence
                if self._matches_pattern(seq, CLOCKWISE_ORDER):
                    return {"detected": True, "pattern": "Clockwise", "progress": list(self.completed_directions), "guidance": "Clockwise complete"}
                if self._matches_pattern(seq, COUNTER_CLOCKWISE_ORDER):
                    return {"detected": True, "pattern": "Counter-Clockwise", "progress": list(self.completed_directions), "guidance": "Counter-clockwise complete"}
            remaining = 4 - len(self.completed_directions)
            return {"detected": False, "pattern": None, "progress": list(self.completed_directions), "guidance": f"{remaining} remaining"}
        except Exception as e:
            logging.error(f"Circular motion detection failed: {e}")
            EmailFeedback.compose_email("Error", f"Circular motion detection failed: {e}")
            return {"detected": False, "pattern": None, "progress": [], "guidance": "Error"}

    def _matches_pattern(self, sequence, pattern):
        if len(sequence) < 4:
            return False
        try:
            start_idx = pattern.index(sequence[0])
        except ValueError:
            return False
        for i, direction in enumerate(sequence[:4]):
            expected = pattern[(start_idx + i) % 4]
            if direction != expected:
                return False
        return True

    def reset_circular_motion(self):
        self.direction_sequence = []
        self.completed_directions = set()

   
    # combined analysis convenience (returns serializable dict)
    # ------------------
    def analyze_frame(self, res=None):
        try:
            if res is None:
                res = self.facemesh_process()
            visible, vis_msg = self.is_complete_face_visible(res)
            dist_info = self.detect_distance_and_size()
            eye_state, ear = self.detect_eye_state(res)
            gaze_info = self.detect_gaze_focus(res)
            pose_info = self.detect_head_pose()
            # steady, move_score = self.is_face_steady()
            circ_info = self.detect_circular_motion()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "face_visible": visible,
                "visibility_message": vis_msg,
                "distance": dist_info,
                "eye_state": eye_state,
                "eye_aspect_ratio": ear,
                "gaze": gaze_info,
                "head_pose": pose_info,
                # "is_steady": steady,
                # "steadiness_score": move_score,
                "circular_motion": circ_info,
            }
        except Exception as e:
            logging.error(f"Frame analysis failed: {e}")
            EmailFeedback.compose_email("Error", f"Frame analysis failed: {e}")
            return {"error": str(e), "face_visible": False}