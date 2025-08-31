# main.py - This files carries the logic for the live version of the program
#  

import cv2
from mp_declaration import mediaPipeDeclaration
from camera_manager import CameraManager

def main():
    # Initialize MediaPipe Pose detection
    pose = mediaPipeDeclaration.initialize_pose_detection()# MediaPipe Pose detection
    camera_manager = CameraManager() # Camera manager
    
    # Initialize camera
    if not camera_manager.initialize_camera():
        print ("No camera detected")
        return
    
    print("Starting live pose detection...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame from camera
            success, frame = camera_manager.capture_frame()
            if not success:
                break

            # Media Pipe information
            rgb_frame = camera_manager.convert_to_rgb(frame)   # Convert BGR to RGB (MediaPipe expects RGB)
            results = mediaPipeDeclaration.process_pose_detection(pose, rgb_frame)  # Detect pose landmarks
            annotated_frame = mediaPipeDeclaration.draw_pose_landmarks(frame, results)   # Draw landmarks on the frame
            
            # Display Frame
            cv2.imshow('Live Pose Detection', annotated_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping pose detection...")
    finally:
        # Clean up
        camera_manager.cleanup()
        mediaPipeDeclaration.close_pose_detection(pose)
        print("Pose detection closed")

if __name__ == "__main__":
    main()
