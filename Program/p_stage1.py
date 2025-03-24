from imports import *

# processes a single frame and returns the annotated image and detection results
def process_frame(cap, detector, image):
    if image is None:
        return None, None

    # Add debug print for frame position
    # print(f"Current frame position: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", end='\r')

    # process image through mediapipe
    frame_timestamp_ms = round(cap.get(cv2.CAP_PROP_POS_MSEC))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # convert back to bgr for display
    annotated_image = mediaPipeDeclaration.draw_landmarks_on_image(image_rgb, detection_result)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    return annotated_image_bgr, detection_result

# processes landmarks for each frame, tracking hand positions and movement
def process_landmarks(detection_result, frame_array, processed_frame_array, processing_active, swaying_detector, mirror_detector, elbow_detector, start_end_detector, frame_number, processing_intervals):
    pose_landmarks_list = detection_result.pose_landmarks
    if pose_landmarks_list:
        for landmarks in pose_landmarks_list:
            if len(landmarks) > 16:
                # get right hand coordinates
                x16, y16 = landmarks[16].x, landmarks[16].y 

                # get left hand coordinates
                x15, y15 = landmarks[15].x, landmarks[15].y

                # get right shoulder coordinates.
                x12, y12 = landmarks[12].x, landmarks[12].y

                # store coordinates
                frame_array.append((x16, y16)) 
                if processing_active:
                    processed_frame_array.append((x16, y16))
                else:
                    processed_frame_array.append((np.nan, np.nan))

                # update movement detectors
                mirror_detector.mirror_calculation(x15, y15, x16, y16)
                swaying_detector.midpoint_calculation(x12, landmarks[11].x)
                
                #14 is elbow, 16 is shoulder, 24 is hip saved to be used
                elbow_detector.elbow_calculation((landmarks[14].x, landmarks[14].y), (x12, y12), (landmarks[24].x, landmarks[24].y))

                # Check for start motion detection
                if not processing_active:
                    processing_active = start_end_detector.start_processing(y16, y15)
                    if processing_active:
                        start_end_detector.current_start_frame = frame_number  # Set start frame
                else:
                    processing_active = start_end_detector.end_processing(x16, x15, y16, y15)
                    if not processing_active and start_end_detector.current_start_frame is not None:
                        processing_intervals.append((start_end_detector.current_start_frame, frame_number))
                        start_end_detector.current_start_frame = None  # Reset start frame

                # Set the midpoint when processing starts
                if processing_active and not swaying_detector.midpointflag:
                    swaying_detector.set_midpoint()  # This will set the default midpoint only when processing starts

    return processing_active

# handles keyboard input for starting/stopping processing and exiting
def handle_user_input(key, frame_number, processing_active, current_start_frame, swaying_detector, processing_intervals, start_end_detector):
    # start processing on 's' key
    if key == ord('s') and not processing_active:
        processing_active = True
        current_start_frame = frame_number
        swaying_detector.set_midpoint_flag_true()
        swaying_detector.set_midpoint()  # Set the initial midpoint when processing starts
        # print(f"Started processing at frame: {frame_number}")

    # end processing on 'e' key
    elif key == ord('e') and processing_active:
        processing_active = False
        if current_start_frame is not None:  # Ensure start frame is set
            processing_intervals.append((current_start_frame, frame_number))
        swaying_detector.set_midpoint_flag_false()
        # print(f"Ended processing at frame: {frame_number}")

    # exit on ESC key
    elif key == 27:
        if processing_active:
            processing_intervals.append((current_start_frame, frame_number))
            # print(f"Ended processing at frame: {frame_number}")
        return None, None

    # Update start_end_detector state
    start_end_detector.processing_active = processing_active

    return processing_active, current_start_frame

# main video processing loop
def process_video(cap, out, detector, frame_array, processed_frame_array, processing_intervals, swaying_detector, mirror_detector, elbow_detector, start_end_detector):
    print("\n=== Video Processing Debug Information ===")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Starting video processing...")
    print(f"Expected total frames: {total_frames}")
    print(f"Video FPS: {fps}")
    
    # initialize processing variables
    frame_number = 0
    frames_read = 0
    processing_active = False
    current_start_frame = None

    while cap.isOpened():
        success, image = cap.read()
        frames_read += 1
        
        if not success:
            print(f"\nTotal frames read: {frames_read}")
            if processing_active and current_start_frame is not None:
                processing_intervals.append((current_start_frame, frame_number - 1))
                print(f"Ended processing at frame: {frame_number - 1}")
            break

        # verify frame is valid
        if image is None:
            continue

        # process current frame
        annotated_image_bgr, detection_result = process_frame(cap, detector, image)

        if annotated_image_bgr is not None:
            # display and process frame
            cv2.imshow('Video Feed - Selection Mode', annotated_image_bgr)
            processing_active = process_landmarks(detection_result, frame_array, processed_frame_array, 
                                                  processing_active, swaying_detector, mirror_detector, 
                                                  elbow_detector, start_end_detector, frame_number, processing_intervals)

            # Update processing status in the command line
            print(f"\rFrame: {frame_number} | Processing: {'Active' if processing_active else 'Inactive'}", end='')

            # add frame number and save frame
            cv2.putText(annotated_image_bgr, f'Frame: {frame_number}', (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            out.write(annotated_image_bgr)

        # handle user input
        key = cv2.waitKey(5) & 0xFF
        processing_active, current_start_frame = handle_user_input(
            key, frame_number, processing_active, current_start_frame, 
            swaying_detector, processing_intervals, start_end_detector)

        # exit if ESC pressed
        if processing_active is None:
            break

        frame_number += 1

    # cleanup resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nActual processed frames: {frame_number}")
    print(f"Number of processing intervals: {len(processing_intervals)}")
    print("=====================================\n")

    return frame_array, processed_frame_array, processing_intervals
    