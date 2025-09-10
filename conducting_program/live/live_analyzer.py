# deals with what is happening when the program is live

import cv2

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

def draw_midpoint_line(pose_landmarks, annotated_frame):
    # Draw midpoint line if midpoint is available
    midpoint = pose_landmarks.get_midpoint()
    if midpoint is not None:
        # Get frame dimensions for full height line
        frame_height = annotated_frame.shape[0]
        frame_width = annotated_frame.shape[1]
        
        # Convert midpoint from normalized coordinates (0-1) to pixel coordinates
        midpoint_pixel = int(midpoint * frame_width)
        
        # Draw line from top to bottom of frame
        cv2.line(annotated_frame, (midpoint_pixel, 0), (midpoint_pixel, frame_height), (225, 255, 255), 2)
        
        # Debug: Show midpoint value
        cv2.putText(annotated_frame, f'Midpoint: {midpoint:.3f}', (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        # Debug: Show why midpoint is None
        left_shoulder = pose_landmarks.left_shoulder_12
        right_shoulder = pose_landmarks.right_shoulder_11
        cv2.putText(annotated_frame, f'L:{left_shoulder is not None} R:{right_shoulder is not None}', (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Has 3 seconds elapsed? if so update the midpoint
    # pose_landmarks.calculate_midpoint(current_time)  # Update Midpoint

def live_analyzer(camera_manager, media_pipe_declaration, pose, system_state, pose_landmarks, clock_manager):

    # Initialize camera
    if not camera_manager.initialize_camera():
        return # Error handling, handled in camera_manager
    
    # Start the program clock
    clock_manager.start_program_clock()
    
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
            
            # Calculate and display FPS and timing info
            fps = camera_manager.calculate_fps()
            program_time = clock_manager.get_program_elapsed_time()
            session_time = clock_manager.get_session_elapsed_time()
            
            # Display timing information on frame
            cv2.putText(annotated_frame, f'FPS: {fps}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Program: {clock_manager.format_time(program_time)}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f'Session: {clock_manager.format_time(session_time)}', (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Landmark Information
            pose_landmarks.update_landmarks(detection_result) # Update Landmarks
            draw_midpoint_line(pose_landmarks, annotated_frame)

            # Get current state and call its main method
            current_state = system_state.get_current_state()
            next_state = current_state.main(pose_landmarks, clock_manager)
            
            # Check if we need to change states
            if next_state != current_state.get_state_name():
                system_state.change_state(next_state, clock_manager)

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
