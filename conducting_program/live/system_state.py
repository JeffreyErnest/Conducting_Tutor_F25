# This will house the logic
# for the live start of the program, 
# as well have infomation on the body ouline
# so people know how far to stand from the camera. 

class SystemState:
    def __init__(self):
        self.state = "setup" # "setup", "coundown", "processing"; System State

        # Hand Variables
        self.left_wrist_y15 = None 
        self.right_wrist_y16 = None

        # wait_for_start_movement variables
        self.current_start_frame = None
        self.previous_y_left = None
        self.previous_x_right = None
        self.previous_x_left = None
        self.previous_y_right = None
        self.frame_count_since_movement = 0
        self.slight_movement_threshold = .005 # TODO: some how make this dynamic
        self.movement_counter = 0

        # Count down variables
        self.countdown_value = 3 # 3 Seconds
        self.countdown_frames = 0 # Track frames for countdown


    # Called to see if user is ready, and set settings
    def pre_processing(self, bpm_settings):

        # Set BPM (This will be done via UI in the future)
        # bpm = input("Enter your target BPM: ")
        # bpm_settings.set_beats_per_minute(int(bpm))  # Convert input to int
        # print(f"BPM set to: {bpm_settings.get_beats_per_minute()}")

        # TODO: Implement remaining setup features:
        # - BPM features
        # - Time signature input
        # - Body outline overlay display
        # - Movement detection logic, if detected change state to processing
        # - 3-2-1 countdown
        # - Seamless transition to live analysis

    
        return  
    
    def show_body_outline(self, camera_manager):
        # Show body outline overlay
        pass
    
    # SETUP CODE PHASE
    def wait_for_start_movement(self, detection_result):

        self.update_landmarks(detection_result) # Update the landmarks

        if self.previous_y_left is None or self.previous_y_right is None:
            self.previous_y_left = self.left_wrist_y15
            self.previous_y_right = self.right_wrist_y16

        significant_movement_threshold = 0.1  

        # Check for significant upward movement for both left and right
        left_moved_up = self.left_wrist_y15 < self.previous_y_left - significant_movement_threshold
        right_moved_up = self.right_wrist_y16 < self.previous_y_right - significant_movement_threshold
        
        # Check for significant downward movement for both left and right
        left_dropped_down = self.left_wrist_y15 > self.previous_y_left + significant_movement_threshold
        right_dropped_down = self.right_wrist_y16 > self.previous_y_right + significant_movement_threshold

        # Determine if both hands are up
        both_hands_up = (left_moved_up and right_moved_up) and not (left_dropped_down or right_dropped_down)

        if both_hands_up and not self.processing_active:
            if self.frame_count_since_movement < 30: # Requires a full second (change to frame of video, later)
               self.frame_count_since_movement += 1  # Increment frame count
            else:
                print ("Started Processing")
                self.processing_active = True
                self.state = "processing" # Set state to Processing
        elif left_dropped_down or right_dropped_down:  # Check if hands have dropped
            self.frame_count_since_movement = 0  # Reset the frame count

    def update_landmarks(self, detection_result):
        landmark = detection_result.pose_landmarks.landmark if detection_result and detection_result.pose_landmarks else None
        if landmark and len(landmark) > 16:
            self.left_wrist_y15 = landmark[15].y
            self.right_wrist_y16 = landmark[16].y
        else:
            self.left_wrist_y15 = None
            self.right_wrist_y16 = None
        return

    # COUNT DOWN PHASE CODE
    def start_countdown(self):
        self.state = "countdown" # Transition to countdown state
        self.countdown_value = 3
        self.countdown_frames = 0
        print("Starting countdown...")

    def update_countdown(self):
        """Update countdown - call this every frame"""
        if self.state == "countdown":
            self.countdown_frames += 1
            
            if self.countdown_frames == 30: # Assuming 30 FPS, show each number for ~1 second (30 frames)
                print(f"{self.countdown_value}")
                self.countdown_value -= 1
                self.countdown_frames = 0
                
                if self.countdown_value <= 0:
                    print("GO!")
                    self.state = "processing"
    
   