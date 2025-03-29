from imports import *
from names import video_out_name
from interface import display_frame, get_screen, get_window_size

# processes a single frame and returns the annotated image
def process_frame(cap, detector, image):

    if image is None:
        return None

    image_bgr = image
    frame_timestamp_ms = round(cap.get(cv2.CAP_PROP_POS_MSEC))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    annotated_image = mediaPipeDeclaration.draw_landmarks_on_image(image_rgb, detection_result)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    return annotated_image_bgr

# checks if a frame index falls within any of the specified intervals
def is_within_intervals(frame_idx, intervals):
    return any(start <= frame_idx <= end for start, end in intervals)

# calculates bpm based on beats within the specified time window
def calculate_bpm(current_frame, beats, fps, window_duration):
    if len(beats) < 2:
        return 0
    
    # convert frames to seconds
    current_time = current_frame / fps
    beat_times = [beat / fps for beat in beats]
    
    # get recent beats within window
    recent_beats = [time for time in beat_times if current_time - time <= window_duration]
    
    if len(recent_beats) < 2:
        return 0
    
    # calculate time between first and last beat
    total_time = recent_beats[-1] - recent_beats[0]
    num_intervals = len(recent_beats) - 1
    
    # calculate and round bpm
    if total_time > 0:
        bpm = (num_intervals * 60) / total_time
    else:
        bpm = 0
    
    return round(bpm, 1)

# displays beat indicator and records bpm data
def print_beats(frame_index, output_frame, filtered_significant_beats, beats, fps, bpm_window, text_display_counter):
    window_size = get_window_size()
    screen = get_screen()

    # For pygame display (keep this for the UI)
    font = pygame.font.Font(None, 50)
    text = font.render("Beat!", True, (255, 255, 255))
    text_x = (window_size[0] - text.get_width()) // 2
    text_y = (window_size[1] - text.get_height()) // 2

    # For video output - make sure to position text within the visible area
    font_out = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (255, 255, 255)  # White text
    text_out = "Beat!"
    text_size_out = cv2.getTextSize(text_out, font_out, font_scale, font_thickness)[0]
    
    # Position the text properly in the center of the frame
    text_x_out = (output_frame.shape[1] - text_size_out[0]) // 2
    text_y_out = 100  # Fixed position near the top where it's visible

    if frame_index in filtered_significant_beats:
        # Show on pygame display
        screen.blit(text, (text_x, text_y))
        
        # Write to output frame for video
        cv2.putText(output_frame, text_out, (text_x_out, text_y_out), 
                   font_out, font_scale, font_color, font_thickness)

        beats.append(frame_index)
        bpm = calculate_bpm(frame_index, beats, fps, bpm_window)

        # Display BPM info below the Beat! text
        bpm_info = f'BPM: {bpm}'
        cv2.putText(output_frame, bpm_info, 
                   (text_x_out, text_y_out + 60), font_out, 1, font_color, font_thickness)
        
        # Save BPM data to file
        full_bpm_info = f'Beats per minute (BPM) at frame {frame_index}: {bpm}\n'
        print(full_bpm_info)
        output_file = video_out_name()
        with open(output_file + '_auto_BPM.txt', 'a') as file:
            file.write(full_bpm_info)

        # Duration of text display
        text_display_counter = 3
    elif text_display_counter > 0:
        # Show on pygame display
        screen.blit(text, (text_x, text_y))
        
        # Write to output frame for video
        cv2.putText(output_frame, text_out, (text_x_out, text_y_out), 
                   font_out, font_scale, font_color, font_thickness)
        text_display_counter -= 1

    return text_display_counter

# processes video for second pass, displaying beats and generating analysis
def output_process_video(cap, out, detector, filtered_significant_beats, processing_intervals, swaying_detector):
    print("\n=== Cycle Two Debug Information ===")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {total_frames/fps:.2f} seconds")
    print(f"Number of beats to display: {len(filtered_significant_beats)}")
    print(f"Processing intervals: {processing_intervals}")
    print("================================\n")
    
    # Initialize parameters
    font = pygame.font.Font(None, 50)
    screen = get_screen()
    bpm_window = 5
    beats = []
    text_display_counter = 0
    frame_index = 0
    
    # Check if we need to apply cropping
    crop_rect = None
    try:
        with open("interface_config.json", "r") as f:
            config = json.load(f)
            crop_data = config.get("crop_rect", None)
            if crop_data:
                crop_rect = tuple(crop_data)  # [x, y, width, height]
                print(f"Applying crop: {crop_rect}")
    except Exception as e:
        print(f"No cropping configuration found: {str(e)}")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Apply cropping if specified
        if crop_rect:
            x, y, w, h = crop_rect
            # Ensure crop dimensions are within image bounds
            height, width = image.shape[:2]
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width-x)
            h = min(h, height-y)
            
            if w > 0 and h > 0:
                # This is the same direct cropping technique used in process_video
                image = image[y:y+h, x:x+w]
            else:
                print("Warning: Invalid crop dimensions, using full frame")
                
        # Process the frame with MediaPipe
        annotated_image_bgr = process_frame(cap, detector, image)
        
        # Skip if the frame couldn't be processed
        if annotated_image_bgr is None:
            continue
            
        # Display the frame in PyGame window
        display_frame(annotated_image_bgr)
        
        # Create a copy of the frame for adding text and graphics
        output_frame = annotated_image_bgr.copy()
        
        # Process the frame if it's within the intervals
        if is_within_intervals(frame_index, processing_intervals):
            # Check if this is a beat frame and add beat text directly
            if frame_index in filtered_significant_beats:
                # Add "BEAT!" text directly to the output frame
                cv2.putText(output_frame, f'BEAT!', 
                           (output_frame.shape[1]//2 - 100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # Also show beat on pygame screen
                text = font.render("Beat!", True, (255, 255, 255))
                window_size = get_window_size()
                text_x = (window_size[0] - text.get_width()) // 2
                text_y = (window_size[1] - text.get_height()) // 2
                screen.blit(text, (text_x, text_y))
                
                # Track beat for BPM calculation
                beats.append(frame_index)
                bpm = calculate_bpm(frame_index, beats, fps, bpm_window)
                
                # Add BPM text to the frame
                bpm_text = f"BPM: {bpm}"
                cv2.putText(output_frame, bpm_text, 
                           (output_frame.shape[1]//2 - 80, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Save BPM data to file
                bpm_info = f'Beats per minute (BPM) at frame {frame_index}: {bpm}\n'
                print(bpm_info)
                output_file = video_out_name()
                with open(output_file + '_auto_BPM.txt', 'a') as file:
                    file.write(bpm_info)
                
                text_display_counter = 3
            elif text_display_counter > 0:
                # Keep showing beat text for a few more frames
                beat_text = "BEAT!"
                cv2.putText(output_frame, beat_text, 
                           (output_frame.shape[1]//2 - 100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                text_display_counter -= 1
            
            # Add swaying information directly
            swaying_detector.swaying_print(frame_index, output_frame)
        
        # Update PyGame display to show current processing
        frame_text = font.render(f"Frame: {frame_index}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 10))
        pygame.display.update()
        pygame.event.pump()
        pygame.time.delay(10)
        
        # Write the annotated frame to the output video
        if out is not None and out.isOpened():
            out.write(output_frame)
        else:
            print(f"Warning: Could not write frame {frame_index} to output video")
        
        # Check for escape key
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
            
        frame_index += 1
    
    # cleanup
    print(f"Processed {frame_index} frames")
    cap.release()
    if out is not None:
        out.release()
    pygame.quit()
    cv2.destroyAllWindows()
    return