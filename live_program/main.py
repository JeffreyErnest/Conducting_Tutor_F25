import cv2
from mp_declaration import mediaPipeDeclaration
import time

def main():
    # Initialize MediaPipe Pose detection
    pose = mediaPipeDeclaration.initialize_pose_detection()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting live pose detection...")
    print("Press 'q' to quit")
    
    # Variables for timing
    prev_frame_time = 0
    new_frame_time = 0
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from camera")
                break
            
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose landmarks
            results = mediaPipeDeclaration.process_pose_detection(pose, rgb_frame)
            
            # Draw landmarks on the frame
            annotated_frame = mediaPipeDeclaration.draw_pose_landmarks(frame, results)
            
            # Calculate and display FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Add FPS text to frame
            cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add pose detection status
            status_text = "Pose Detected" if results.pose_landmarks else "No Pose Detected"
            status_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Display the frame
            cv2.imshow('Live Pose Detection', annotated_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping pose detection...")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        mediaPipeDeclaration.close_pose_detection(pose)
        print("Camera and pose detection closed")

if __name__ == "__main__":
    main()
