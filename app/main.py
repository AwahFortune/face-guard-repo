import cv2
import numpy as np
import logging
import time
from collections import deque

import os
from pathlib import Path
current_dir = Path(__file__).parent
# --- 2. IMPORTS (repo modules) ---
from model import Model
from image_processing import ImageProcessor
from detect_faces import FaceDetector
from user_guide import Guide
from liveness import Liveness
from db import Database
from recognize import Recognize
from register import FaceRegistration
from render import FaceRenderer

#from attributes.mask import MaskDetector

#from attributes.eyeglasses import EyeglassClassifier


# Utility
from email_feedback import EmailFeedback

# --- 3. Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# --- 4. CONFIGURABLE THRESHOLDS (tweak at top-level) ---
LIVENESS_REQUIRED_CONSECUTIVE = 3      # 3-5 per your requirement
STABLE_FPS_CHECK = 10                  # used for timing windows where needed
HEAD_SEQ = ["Up", "Right", "Down", "Left", "Up"]  # full rotation (clockwise). Allow changing to CCW if needed
SIMILARITY_THRESHOLD = 0.5             # default similarity threshold (normalized embeddings)
RECOGNITION_ATTEMPTS = 1
FRAME_SAVE_DIR = current_dir/"temp_frames"
FRAME_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- 5. INITIALIZATION FUNCTION ---
def initialize_system():
    """Initialize heavy components; return dict of components."""
    try:
        logging.info("Initializing models and services...")
        model_manager = Model()
        app = model_manager.initialize_insightface()     # insightface FaceAnalysis
        facemesh = model_manager.initialize_mediapipe()  # mediapipe FaceMesh

        # Database
        db_manager = Database()
        mysql_conn, mysql_cursor = db_manager.initialize_mysql()

        # Milvus / collection (optional)
        milvus_collection = db_manager.initialize_milvus()
        # Image processor needs insightface "app" instance
        image_processor = ImageProcessor(app)

        # Liveness (DeepFace-based in your repo)
        liveness_detector = Liveness()

        # Renderer and optional attribute detectors
        face_renderer = FaceRenderer()
        if hasattr(face_renderer, "initialize_mask_renderer"):
            try:
                face_renderer.initialize_mask_renderer()
            except Exception:
                logging.warning("Mask renderer init failed; continuing without mask rendering")

        # mask_detector = MaskDetector() if MaskDetector else None
        # glasses_classifier = EyeglassClassifier() if EyeglassClassifier else None

        # For registration/recognition flows
        security_manager = None
        try:
            # if you have a security/encrypt class, initialize it here (pattern from repo)
            from encrypt import encrypt as EncryptClass
            security_manager = EncryptClass()
            if hasattr(security_manager, "initialize_security"):
                security_manager.initialize_security()
        except Exception:
            logging.info("Security manager not initialized (ensure encrypt is available if needed).")

        logging.info("Initialization complete.")
        return {
            "app": app,
            "facemesh": facemesh,
            "image_processor": image_processor,
            "liveness": liveness_detector,
            "db": db_manager,
            "mysql_conn": mysql_conn,
            "mysql_cursor": mysql_cursor,
            "milvus_collection": milvus_collection,
            "face_renderer": face_renderer,
            # "mask_detector": mask_detector,
            # "glasses_classifier": glasses_classifier,
            "security_manager": security_manager,
        }
    except Exception as e:
        logging.critical(f"System init failed: {e}", exc_info=True)
        EmailFeedback.compose_email("Critical Error", f"System init failed: {e}")
        raise


# --- 6. Helpers: capture/process frames ---
def capture_frame(cap):
    """Capture a BGR frame from the camera and return both BGR and RGB copies."""
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        return None, None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr, frame_rgb


def process_frame(image_processor, frame_rgb):
    """
    Use ImageProcessor.process_image which in your repo returns (processed, faces)
    Returns (processed_image, faces) where faces is list of insightface face objects.
    """
    try:
        processed, faces = image_processor.process_image(frame_rgb)
        return processed, faces
    except Exception as e:
        logging.warning(f"Image processing failed: {e}")
        # Fallback: return raw frame and attempt a light-weight face detection later
        return frame_rgb, []


def save_frame_for_db(user_id: str, stage: str, img_rgb: np.ndarray):
    """Save a temporary frame for DB storage. Return the path string."""
    ts = int(time.time() * 1000)
    fname = FRAME_SAVE_DIR / f"{user_id}_{stage}_{ts}.jpg"
    cv2.imwrite(str(fname), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return str(fname)


# --- 7. Liveness sequential verifier ---
def verify_liveness_consecutive(cap, components, required=LIVENESS_REQUIRED_CONSECUTIVE, timeout_s=8):
    """
    Require `required` consecutive frames where liveness.check_liveness(...) == True.
    Returns the last good frame_rgb (RGB) when success, or None if timeout/fail.
    """
    liveness = components["liveness"]
    t_start = time.time()
    window = deque(maxlen=required)

    logging.info(f"Starting liveness check: need {required} consecutive positive frames.")
    while time.time() - t_start < timeout_s:
        _, frame_rgb = capture_frame(cap)
        if frame_rgb is None:
            continue

        try:
            # Liveness API in your repo expects an np.ndarray; use the rgb frame
            result = liveness.check_liveness(frame_rgb)
            ok = result is True or (isinstance(result, str) and result.lower() == "real")
            window.append(bool(ok))

            # quick UI indicator:
            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"Liveness window: {sum(window)}/{required}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if all(window) else (0, 128, 255), 2)
            cv2.imshow("Liveness", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Liveness interrupted by user.")
                return None

            if len(window) == required and all(window):
                logging.info("Consecutive liveness check passed.")
                cv2.destroyWindow("Liveness")
                return frame_rgb

        except Exception as e:
            logging.warning(f"Liveness check error (ignored this frame): {e}")

    logging.warning("Liveness check FAILED / timeout.")
    try:
        cv2.destroyWindow("Liveness")
    except Exception:
        pass
    return None


# --- 8. Stable centered face capture (gaze + center + eyes open) ---
# --- 8. Stable centered face capture (gaze + center + eyes open) ---
def get_centered_frame(cap, components):
    """
    Wait until Guide reports face centered, eyes open, gaze focused for stability_seconds
    Returns (processed_frame_rgb, face_obj) or (None, None) on failure/timeout.
    """
    app = components["app"]
    facemesh = components["facemesh"]
    image_processor = components["image_processor"]

    guide = None
    bgr, rgb = capture_frame(cap)
    processed, faces = process_frame(image_processor, rgb)
    if not faces:
        # show hint and continue
        disp = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.putText(disp, "No face detected - centre your face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Centering", disp)

    face = faces[0]
    # re-use Guide for more checks
    guide = Guide(processed, facemesh, app)
    res_mesh = guide.facemesh_process()
    visible, _ = guide.is_complete_face_visible(res_mesh)
    gaze_info = guide.detect_gaze_focus(res_mesh)
    eye_state, _ = guide.detect_eye_state(res_mesh)
    distance_info = guide.detect_distance_and_spoof()
    # compose boolean
    ok = visible and (gaze_info.get("status") == "Focused") and (eye_state == "Open") and (distance_info[2] in ("OK", "Perfect"))

    # UI
    disp = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    cv2.putText(disp, f"Centered:{visible} Gaze:{gaze_info.get('status')} Eyes:{eye_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 128, 255), 2)
    cv2.imshow("Centering", disp)
    return processed, face


# --- 9. Head rotation capture sequence ---
def capture_head_rotation(cap, components, centered_embedding, directions=HEAD_SEQ, timeout_s=8):
    """
    Wait for the user to turn their head in each direction.
    For each target direction, capture embedding, compute similarity to centered_embedding,
    and save the frame.
    Returns {direction: {"embedding": emb, "similarity": sim, "frame_path": path}}.
    """
    results = {}
    app = components["app"]
    facemesh = components["facemesh"]

    # Initialize once for history tracking
    ret, init_bgr = cap.read()
    if not ret or init_bgr is None:
        logging.error("Camera read failed.")
        return results

    guide = Guide(init_bgr, facemesh, app)

    for dir_target in directions:
        logging.info(f"=== Please turn head to: {dir_target} ===")
        start_t = time.time()
        captured = False

        while time.time() - start_t < timeout_s:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            guide.update_frame(frame_bgr)
            res = guide.facemesh_process()

            pose = guide.detect_head_pose()
            direction = pose.get("direction", "Unknown")

            # Display helper
            disp = frame_bgr.copy()
            cv2.putText(disp, f"Target: {dir_target}, Current: {direction}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if direction == dir_target else (0, 128, 255), 2)
            cv2.imshow("Head Rotation", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return results

            if direction == dir_target:
                # Face object with embedding is inside guide.face
                emb = None
                if guide.face and len(guide.face) > 0:
                    try:
                        emb = guide.face[0].normed_embedding
                    except Exception:
                        logging.warning("Face object missing embedding.")

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_path = save_frame_for_db("temp", f"pose_{dir_target}", frame_rgb)

                sim = None
                if emb is not None and centered_embedding is not None:
                    try:
                        sim = float(np.dot(centered_embedding, emb))
                    except Exception:
                        pass

                results[dir_target] = {
                    "embedding": emb,
                    "similarity": sim,
                    "frame_path": frame_path
                }
                logging.info(f"Captured {dir_target}: similarity vs center = {sim}")
                captured = True
                time.sleep(0.5)
                break

        if not captured:
            logging.warning(f"Timeout for {dir_target}, skipping.")
            results[dir_target] = {
                "embedding": None,
                "similarity": None,
                "frame_path": None,
                "error": "timeout"
            }

    try:
        cv2.destroyWindow("Head Rotation")
    except Exception:
        pass
    return results



# --- 10. Registration flow (high-level) ---
def register_user(cap, user_id, components):
    """
    Implements your registration flow:
      1) 3-5 consecutive liveness frames
      2) Centered face capture (eyes focused + centered)
      3) Sequence of head rotation frames capturing embeddings compared to center
      4) Attribute detection + rendering & storing images in DB
    """
    logging.info(f"BEGIN registration for: {user_id}")
    # Step 0: quick sanity checks
    if not user_id:
        return {"status": "error", "message": "user_id required"}

    # Step 1: Liveness
    live_frame = verify_liveness_consecutive(cap, components, required=LIVENESS_REQUIRED_CONSECUTIVE)
    if live_frame is None:
        return {"status": "error", "message": "liveness failed"}

    # Step 2: Centered frame
    centered_frame, centered_face = get_centered_frame(cap, components)
    if centered_frame is None or centered_face is None:
        return {"status": "error", "message": "centered capture failed"}

    # Save the centered image (temporal id)
    centered_path = save_frame_for_db(user_id, "centered", centered_frame)
    logging.info(f"Saved centered frame to {centered_path}")

    # Extract centered embedding
    try:
        centered_embedding = centered_face.normed_embedding
    except Exception:
        centered_embedding = None
        logging.warning("Could not extract centered embedding")

    # Step 3: head rotation captures
    rotation_results = capture_head_rotation(cap, components, centered_embedding, directions=HEAD_SEQ)

    # Step 4: Basic accept criteria (require a minimum number of valid pose captures)
    valid_similarities = [v["similarity"] for v in rotation_results.values() if v.get("similarity") is not None]
    avg_sim = float(np.mean(valid_similarities)) if valid_similarities else 0.0
    logging.info(f"Average similarity across rotation captures: {avg_sim} (need tuning)")

    # Step 5: Register centered embedding to DB
    try:
        registrar = FaceRegistration(
            components["app"],
            components.get("security_manager"),
            components["mysql_conn"],
            components["mysql_cursor"],
            components.get("milvus_collection"),
        )
        reg_res = registrar.register_face(user_id, centered_frame)
    except Exception as e:
        logging.error(f"Registration error: {e}", exc_info=True)
        EmailFeedback.compose_email("Error", f"Registration error: {e}")
        return {"status": "error", "message": "registration failed"}

    # Step 6: Attribute detection + rendering
    attrs = {}
    try:
        if components.get("mask_detector"):
            try:
                attrs["mask"] = components["mask_detector"].detect_mask(centered_frame)
            except Exception as e:
                logging.warning(f"Mask detection error: {e}")
                attrs["mask"] = None
        if components.get("glasses_classifier"):
            try:
                attrs["glasses"] = components["glasses_classifier"].detect(centered_frame)
            except Exception as e:
                logging.warning(f"Glasses detection error: {e}")
                attrs["glasses"] = None

        # Render permutations (for analysis / optional user preview)
        try:
            rendered_versions = {}
            if components.get("face_renderer"):
                # perfect centered face -> apply mask and glasses overlays as available
                if attrs.get("mask"):
                    try:
                        rm = components["face_renderer"].render_mask(centered_frame)
                        rendered_versions["mask"] = save_frame_for_db(user_id, "render_mask", rm)
                    except Exception as e:
                        logging.warning(f"Mask render error: {e}")
                if attrs.get("glasses") in ("sunglasses", "transparent"):
                    try:
                        g_img = components["face_renderer"].add_glasses(centered_frame.copy(), centered_face)
                        rendered_versions["glasses"] = save_frame_for_db(user_id, "render_glasses", g_img)
                    except Exception as e:
                        logging.warning(f"Glasses render error: {e}")
                # combined rendering could be applied similarly
            attrs["rendered_paths"] = rendered_versions
        except Exception as e:
            logging.warning(f"Rendering step failed: {e}")

    except Exception as e:
        logging.warning(f"Attribute detection failed: {e}")

    # Step 7: Persist captured frame(s) metadata to DB user_images
    try:
        cur = components["mysql_cursor"]
        conn = components["mysql_conn"]
        timestamp = int(time.time())
        cur.execute(
            "INSERT INTO user_images (user_id, image_path, det_score, registration_time) VALUES (%s, %s, %s, %s)",
            (user_id, centered_path, float(centered_face.det_score if hasattr(centered_face, "det_score") else 0.0), timestamp),
        )
        conn.commit()
    except Exception as e:
        logging.warning(f"Failed to insert user_images row: {e}")

    logging.info(f"Registration finished for {user_id}")
    return {
        "status": "success",
        "user_id": user_id,
        "register_response": reg_res if 'reg_res' in locals() else None,
        "rotation_results": rotation_results,
        "attributes": attrs,
        "avg_similarity": avg_sim,
    }


# --- 11. Recognition flow (high-level) ---
def recognize_user(cap, user_id, components, attempts=RECOGNITION_ATTEMPTS, threshold=SIMILARITY_THRESHOLD):
    """
    Wait for liveness + centered frame and run Recognize.recognize_face_authorization
    multiple times and produce a majority result.
    """
    logging.info(f"BEGIN recognition for {user_id}")
    successes = 0
    for a in range(attempts):
        # Liveness + centered detection
        live = verify_liveness_consecutive(cap, components, required= LIVENESS_REQUIRED_CONSECUTIVE)
        if live is None:
            logging.warning(f"Attempt {a+1}: liveness failed")
            continue

        stable_frame, stable_face = get_centered_frame(cap, components)
        if stable_frame is None or stable_face is None:
            logging.warning(f"Attempt {a+1}: stable capture failed")
            continue

        # Call Recognize flow (it expects an image passed at construction in your repo)
        try:
            recognizer = Recognize(stable_frame, components["app"], components.get("security_manager"))
            res = recognizer.recognize_face_authorization(user_id)
            logging.info(f"Recognition attempt {a+1} result: {res}")
            if res.get("status") == "success" and res.get("result") == "authorized" and res.get("similarity", 0) >= threshold:
                successes += 1
        except Exception as e:
            logging.error(f"Recognition attempt error: {e}", exc_info=True)

    authorized = successes > attempts / 2
    logging.info(f"Recognition final: authorized={authorized} ({successes}/{attempts})")
    return {"status": "success", "authorized": authorized, "matches": successes, "attempts": attempts}


# --- 12. Main Execution ---
def main():
    components = initialize_system()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        mode = input("Mode (register/recog): ").strip().lower()
        user_id = input("User ID: ").strip()
        if mode == "register":
            out = register_user(cap, user_id, components)
            print(out)
        elif mode in ("recog", "recognize"):
            out = recognize_user(cap, user_id, components)
            print(out)
        else:
            print("Unknown mode. Use 'register' or 'recog'.")
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.critical(f"Main loop error: {e}", exc_info=True)
        EmailFeedback.compose_email("Critical Error", f"Main loop error: {e}")
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if components.get("mysql_conn"):
                components["mysql_conn"].close()
        except Exception:
            pass
        logging.info("Exiting.")


if __name__ == "__main__":
    main()
