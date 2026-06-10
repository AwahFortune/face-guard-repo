# test_glass.py
import os, sys, tempfile, subprocess, cv2
from pathlib import Path

PYTHON_CMD = r'C:\Users\user\anaconda3\envs\eyeglass\python.exe'
# project root
ROOT_DIR = Path(__file__).parent.parent
# make sure the package is importable
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT_DIR)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, frame)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [PYTHON_CMD, '-m', 'attributes.eyeglasses', '--image', tmp_path],
            capture_output=True, text=True, timeout=60,
            env=env
        )
        label = proc.stdout.strip() if proc.returncode == 0 else 'error'
    finally:
        os.remove(tmp_path)

    cv2.putText(frame, f"Glasses: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Eyeglass Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
