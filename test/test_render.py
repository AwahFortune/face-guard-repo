import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'app'))

from app.render import FaceRenderer
from app.model import Model

# Paths to assets
glasses_paths = {
    'sunglass': str(Path(__file__).parent.parent / 'assets' / 'sunglass.png'),
    'transparent': str(Path(__file__).parent.parent / 'assets' / 'trans_glass.png')
}
mask_type = 'mask_blue'

# Initialize InsightFace and renderer
model = Model()
app = model.initialize_insightface()
renderer = FaceRenderer()
renderer.initialize_mask_render()
# load default glasses images
current_type = None

toggles = {
    'mask': False,
    'sunglass': False,
    'transparent': False
}

help_text = [
    '[M] Mask:         OFF',
    '[S] Sunglass:     OFF',
    '[T] Transparent:  OFF',
    '[Q] Quit'
]

def draw_help(frame):
    for idx, text in enumerate(help_text):
        key = text[1].lower()
        color = (0, 255, 0) if 'ON' in text else (0, 0, 255)
        cv2.putText(frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def update_help():
    help_text[0] = f"[M] Mask:         {'ON' if toggles['mask'] else 'OFF'}"
    help_text[1] = f"[S] Sunglass:     {'ON' if toggles['sunglass'] else 'OFF'}"
    help_text[2] = f"[T] Transparent:  {'ON' if toggles['transparent'] else 'OFF'}"


def main():
    global current_type
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    current_type = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = frame.copy()
        faces = app.get(frame)

        # Render mask if toggled
        if toggles['mask']:
            try:
                out = renderer.render_mask(out, mask_image=mask_type)
            except Exception as e:
                print(f"Mask error: {e}")

        # Determine which glasses to render
        if toggles['sunglass'] or toggles['transparent']:
            # ensure only one type is on at a time
            if toggles['sunglass']:
                selected = 'sunglass'
            else:
                selected = 'transparent'

            # load if changed
            if current_type != selected:
                renderer.set_glasses_image(glasses_paths[selected])
                current_type = selected

            # render
            if faces:
                for face in faces:
                    try:
                        out = renderer.add_glasses(out, face)
                    except Exception as e:
                        print(f"Glasses error: {e}")

        # Draw toggle status
        update_help()
        draw_help(out)

        cv2.imshow('Render Test', out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # Q or ESC
            break
        elif key == ord('m'):
            toggles['mask'] = not toggles['mask']
        elif key == ord('s'):
            toggles['sunglass'] = not toggles['sunglass']
            if toggles['sunglass']:
                toggles['transparent'] = False
        elif key == ord('t'):
            toggles['transparent'] = not toggles['transparent']
            if toggles['transparent']:
                toggles['sunglass'] = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
