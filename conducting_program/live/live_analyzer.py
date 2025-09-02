# deals with what is happening when the program is live

import cv2

def live_analyzer(camera_manager, media_pipe_declaration, pose, bpm_settings, live_start):

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
            
            # Get landmark coordinates
            x15, y15, x16, y16 = media_pipe_landmarks(detection_result)
             
            # TODO: 
            # Damn will you look at that I did what I wanted to do crazy

            if live_start.state == "setup":
                print("=== SETUP PHASE ===")

                live_start.wait_for_start_movement(y15, y16)

                if show_frame(annotated_frame): # Display Frame
                    break # Exit loop, if returned True

                continue # Continue to next frame

            elif live_start.state == "processing":
                print("=== PROCESSING PHASE ===")

                # TODO: 
                # Look to see if ending motion is triggered if it is, end the program
                
                if show_frame(annotated_frame):  # Display Frame
                    break # Exit loop, if returned True

                    
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

# Method to get landmarks
def media_pipe_landmarks(detection_result):
    if detection_result and detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks # MediaPipe pose landmarks are accessed directly, not as a list

        # Check if we have enough landmarks and they exist
        if len(landmarks.landmark) > 16:
            x15 = landmarks.landmark[15].x # Left wrist X axis
            y15 = landmarks.landmark[15].y # Left wrist Y axis
            x16 = landmarks.landmark[16].x # Right wrist X axis
            y16 = landmarks.landmark[16].y # Right wrist Y axis
            return x15, y15, x16, y16
   
    return None # Return None if no landmarks detected
