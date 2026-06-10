import cv2
import mediapipe as mp
from collections import deque
from pathlib import Path
import sys
import json


current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "app"))

from app.user_guide import Guide

try:
    from app.model import Model
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# feature toggles
toggles = {
    'eye':      True,
    'gaze':     False,
    'head':     False,
    'distance': False,
    'circle':   False,
    'visible':  False,
    'steady':   False,
    'analysis': False,
}

# Calibration state
calibrating = False
yaw_samples = []
pitch_samples = []
CALIBRATION_SAMPLES = 10  # Number of samples to collect for calibration

help_text = [
    "[E] Eye     ",
    "[G] Gaze    ",
    "[H] Head    ",
    "[D] Dist.   ",
    "[C] Circle  ",
    "[V] Visible ",
    "[A] Analysis",
    "[R] Reset C.",
    "[Q] Quit    "
]


def draw_toggles(frame):
    y_offset = 0
    key_map = {
        'e': 'eye', 'g': 'gaze', 'h': 'head', 'd': 'distance', 'c': 'circle',
        'v': 'visible', 'a': 'analysis'
    }
    for txt in help_text:
        if txt.startswith("[K]") or txt.startswith("[R]") or txt.startswith("[Q]"):
            color = (255, 255, 255)
        else:
            k = txt[1].lower()
            toggle_key = key_map.get(k)
            color = (0, 255, 0) if toggles.get(toggle_key) else (0, 0, 255)
        cv2.putText(frame, txt, (10, 30 + y_offset * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 1


def main():
    # initialize MediaPipe & InsightFace (via Model if present)
    if MODEL_AVAILABLE:
        m = Model()
        app = m.initialize_insightface()
        mp_face = m.initialize_mediapipe()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # read first frame to create Guide instance
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    guide = Guide(frame, mp_face, app)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update and process
        guide.update_frame(frame)
        res = guide.facemesh_process()

        y0 = 160  # Starting y-position for text overlays

        # Eye
        if toggles['eye']:
            state, ear = guide.detect_eye_state(res)
            text = f"Eyes: {state} ({ear:.2f})"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y0 += 30

        # Gaze
        if toggles['gaze']:
            gaze = guide.detect_gaze_focus(res)
            status = gaze.get('status')
            gx, gy = gaze.get('offset', (0.0, 0.0))
            text = f"Gaze: {status} ({gx:.3f}, {gy:.3f})"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            y0 += 30

        # Head
        if toggles['head']:
            pose = guide.detect_head_pose()
            dir_ = pose.get('direction')
            yaw = pose.get('yaw', 0.0)
            pitch = pose.get('pitch', 0.0)
            text = f"Head: {dir_} (Y{yaw:.1f}, P{pitch:.1f})"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)
            y0 += 30

        # Distance
        if toggles['distance']:
            dist_info = guide.detect_distance_and_size()
            dc = dist_info.get('distance_cm') or 0.0
            ratio = dist_info.get('face_ratio') or 0.0
            status = dist_info.get('status')
            text = f"Dist: {dc:.1f}cm R{ratio:.2f} [{status}]"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            y0 += 30

        # Circular motion
        if toggles['circle']:
            circ = guide.detect_circular_motion()
            detected = circ.get('detected', False)
            pattern = circ.get('pattern')
            text = f"Circular: {pattern if detected else 'No'} ({circ.get('guidance', '')})"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
            y0 += 30

        # Visibility
        if toggles['visible']:
            visible, msg = guide.is_complete_face_visible(res)
            text = f"Visible: {visible} ({msg})"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y0 += 30

        # Full Analysis
        if toggles['analysis']:
            analysis = guide.analyze_frame(res)
            # Display a summary or print to console for full dict
            print(json.dumps(analysis, indent=2))  # Print to console for detailed view
            text = f"Analysis: Calib={analysis.get('calibrated')} Visible={analysis.get('face_visible')}"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 255), 2)
            y0 += 30

        draw_toggles(frame)

        cv2.imshow("Guide Tester", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            toggles['eye'] = not toggles['eye']
        elif key == ord('g'):
            toggles['gaze'] = not toggles['gaze']
        elif key == ord('h'):
            toggles['head'] = not toggles['head']
        elif key == ord('d'):
            toggles['distance'] = not toggles['distance']
        elif key == ord('c'):
            toggles['circle'] = not toggles['circle']
        elif key == ord('v'):
            toggles['visible'] = not toggles['visible']
        elif key == ord('a'):
            toggles['analysis'] = not toggles['analysis']        
        elif key == ord('r'):
            guide.reset_circular_motion()
            print("Circular motion reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()