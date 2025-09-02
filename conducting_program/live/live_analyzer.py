# deals with what is happening when the program is live

import cv2

def live_analyzer(camera_manager, media_pipe_declaration, pose):

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
            success, frame = camera_manager.capture_frame() # Capture frame from camera
            if not success:
                break # If the frame wasn't captured we break out of the loop
            frame = cv2.flip(frame, 1) # Flip the frame on y axis. 
            rgb_frame = camera_manager.convert_to_rgb(frame) # Convert BGR to RGB (MediaPipe expects RGB)
            results = media_pipe_declaration.process_pose_detection(media_pipe_declaration.pose, rgb_frame) # Detect pose landmarks
            annotated_frame = media_pipe_declaration.draw_pose_landmarks(frame, results) # Draw landmarks on the frame
            
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
        media_pipe_declaration.close_pose_detection(media_pipe_declaration.pose)
        print("Pose detection closed")