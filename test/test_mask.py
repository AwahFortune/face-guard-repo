import time
import logging
import cv2
from pathlib import Path
import sys 
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "app")) 
from app.mask import MaskDetector  

# Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    logging.info("Starting MaskDetector live test...")
    detector = MaskDetector()  # will load faceNet and mask model

    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        raise IOError("Cannot open camera / file")

    print("Press 'q' to quit.")
    last_time = time.time()
    fps = 0.0
    process_every_n_frames = 1  

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            logging.warning("Frame read failed; exiting.")
            break

        frame_idx += 1
        frame_proc = frame 

        mask_flag = None
        if (frame_idx % process_every_n_frames) == 0:
            try:
                mask_flag = detector.detect_mask(frame_proc)
            except Exception as e:
                logging.exception("Unexpected error calling detect_mask: %s", e)
                mask_flag = False

        # compute fps (smoothed)
        now = time.time()
        dt = now - last_time if now != last_time else 1e-6
        last_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

        # Decide label and color
        if mask_flag:
            label = "MASK"
            color = (0, 255, 0)  # green
        else:
            label = "NO MASK"
            color = (0, 0, 255)  # red

        # Overlay text (top-left)
        text = f"{label}  FPS:{fps:.1f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Mask detector (local)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Exiting.")

if __name__ == "__main__":
    main()
