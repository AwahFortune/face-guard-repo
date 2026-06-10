#!/usr/bin/env python3
import cv2
import numpy as np
import time
import sys
from pathlib import Path
import logging
current_dir = Path(__file__).parent
sys.path.append(str(email_dir = current_dir.parent / "app"))
from app.eyeglasses import EyeglassClassifier
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])
def label_to_color(label: str):
    if label == "transparent":
        return "GLASSES", (0, 255, 0)
    if label == "sunglasses":
        return "SUNGLASSES", (0, 200, 255)
    if label == "no-glasses":
        return "NO GLASSES", (0, 0, 255)
    return "ERROR", (0, 255, 255)

def main():
    logging.info("Starting simple eyeglass live test...")
    clf = EyeglassClassifier()  # will raise if it can't initialize

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera (index 0)")
        return

    print("Press 'q' to quit, 's' to save a frame.")
    last_time = time.time()
    fps = 0.0
    saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logging.warning("Frame read failed; exiting.")
                break

            # Pass entire frame to classifier (no preprocessing)
            try:
                label = clf.detect(frame)
            except Exception as e:
                logging.exception("Classifier error: %s", e)
                label = "error"

            text, color = label_to_color(label)

            # Overlay label and FPS
            cv2.putText(frame, f"{text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # FPS update (simple)
            now = time.time()
            dt = now - last_time if now != last_time else 1e-6
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Eyeglass Live (Simple)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Exiting.")

if __name__ == "__main__":
    main()