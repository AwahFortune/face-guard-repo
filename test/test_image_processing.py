import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Add the process directory to Python path
current_dir = Path(__file__).parent
process_dir = current_dir.parent / "processing"
sys.path.append(str(process_dir))
sys.path.append(str(Path(__file__).parent.parent / 'app'))
from app.model import Model
model = Model()
app = model.initialize_insightface()
# Import the image processor
from app.image_processing import ImageProcessor

def main():
    print("Starting real-time facial recognition preprocessing...")
    print("Press 'q' to quit, 's' to save frame")
    
    # Initialize the image processor
    processor = ImageProcessor(app)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Process frame for facial recognition
            try:
                processed_frame = processor.process_image(frame)
                
                if processed_frame is not None:
                    # Convert back to BGR for display
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add status text
                    cv2.putText(display_frame, 'Processed for Face Recognition', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    display_frame = frame
                    cv2.putText(display_frame, 'Processing Failed - Using Original', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('Real-time Processing for Facial Recognition', display_frame)
                
            except Exception as e:
                print(f"Processing error: {str(e)}")
                cv2.imshow('Real-time Processing for Facial Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current processed frame
                timestamp = int(time.time())
                filename = f"processed_frame_{timestamp}.jpg"
                if 'processed_frame' in locals() and processed_frame is not None:
                    save_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, save_frame)
                    print(f"Processed frame saved as {filename}")
                else:
                    cv2.imwrite(filename, frame)
                    print(f"Original frame saved as {filename}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\nSession Statistics:")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {frame_count / total_time:.1f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()