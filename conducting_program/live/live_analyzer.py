# deals with what is happening when the program is live

import cv2
from live.system_state import State

def live_analyzer(camera_manager, media_pipe_declaration, pose, system_state, pose_landmarks):

    # Initialize camera
    if not camera_manager.initialize_camera():
        return # Error handling, handled in camera_manager
    
    # Debug Print Statments
    print("Starting live pose detection...")
    print("Press 'q' to quit")
    
    # Main loop
    try:
        while True:

            # Frame and skeleton setup information
            annotated_frame, detection_result = setup_frame(camera_manager, media_pipe_declaration, pose)
            if annotated_frame is None:
                  break
            
            pose_landmarks.update_landmarks(detection_result) # Update Landmarks

            # Get current state and call its main method
            current_state = system_state.get_current_state()
            next_state = current_state.main(pose_landmarks)
            
            # Check if we need to change states
            if next_state != current_state.get_state_name():
                system_state.change_state(next_state)

            # Display frame
            if show_frame(annotated_frame):
                break
                    
    except KeyboardInterrupt:
        print("\nStopping pose detection...")
    finally:
        # Clean up
        camera_manager.cleanup()
        media_pipe_declaration.close_pose_detection(pose)
        print("Pose detection closed")

# Display Frame
def show_frame(annotated_frame):
    
    cv2.imshow('Live Pose Detection', annotated_frame) # Display Frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # Break loop on 'q' key press, set flag to true
        return True # Signal to exit
    return False # Continue

# Frame and skeleton setup information
def setup_frame(camera_manager, media_pipe_declaration, pose):
    success, frame = camera_manager.capture_frame() # Capture frame from camera
    if not success:
       return None, None
    frame = cv2.flip(frame, 1) # Flip the frame on y axis. 
    rgb_frame = camera_manager.convert_to_rgb(frame) # Convert BGR to RGB (MediaPipe expects RGB)
    results = media_pipe_declaration.process_pose_detection(pose, rgb_frame) # Detect pose landmarks
    annotated_frame = media_pipe_declaration.draw_pose_landmarks(frame, results) # Draw landmarks on the frame
    return annotated_frame, results # 'annotated_image_bgr' is for display, 'results' contains the actualy data
