# This will house the logic
# for the live start of the program, 
# as well have infomation on the body ouline
# so people know how far to stand from the camera. 

from enum import Enum

class State(Enum):
    SETUP = "setup"
    COUNTDOWN = "countdown"
    PROCESSING = "processing"
    ENDING = "ending"

class SystemState:
    def __init__(self):
        self.current_state = SetupState()  # Start with setup state
    
    def get_current_state(self):
        return self.current_state
    
    def change_state(self, new_state):
        if new_state == State.COUNTDOWN.value:
            self.current_state = CountdownState()
        elif new_state == State.PROCESSING.value:
            self.current_state = ProcessingState()
        elif new_state == State.ENDING.value:
            self.current_state = EndingState()
        print(f"State changed to: {new_state}")

class SetupState:
    def __init__(self):
        self.left_wrist_y15 = None
        self.right_wrist_y16 = None
        self.previous_y_left = None
        self.previous_y_right = None
        self.processing_active = False
        self.frame_count_since_movement = 0
        self.significant_movement_threshold = 0.1
        print("=== SETUP PHASE ===")
    
    def get_state_name(self):
        return State.SETUP.value
   
    def main(self, pose_landmarks): 
        return self.wait_for_start_movement(pose_landmarks)

    def wait_for_start_movement(self, pose_landmarks):
        (_, self.left_wrist_y15) = pose_landmarks.get_pose_landmark_15()
        (_, self.right_wrist_y16) = pose_landmarks.get_pose_landmark_16()

        if self.previous_y_left is None or self.previous_y_right is None:
            self.previous_y_left = self.left_wrist_y15
            self.previous_y_right = self.right_wrist_y16

        # Check for significant upward movement for both left and right
        left_moved_up = self.left_wrist_y15 < self.previous_y_left - self.significant_movement_threshold
        right_moved_up = self.right_wrist_y16 < self.previous_y_right - self.significant_movement_threshold
        
        # Check for significant downward movement for both left and right
        left_dropped_down = self.left_wrist_y15 > self.previous_y_left + self.significant_movement_threshold
        right_dropped_down = self.right_wrist_y16 > self.previous_y_right + self.significant_movement_threshold

        # Determine if both hands are up
        both_hands_up = (left_moved_up and right_moved_up) and not (left_dropped_down or right_dropped_down)

        if both_hands_up and not self.processing_active:
            if self.frame_count_since_movement < 30:
               self.frame_count_since_movement += 1
            else:
                print("Starting Countdown")
                self.processing_active = True
                return State.COUNTDOWN.value  # Use enum value
        elif left_dropped_down or right_dropped_down:
            self.frame_count_since_movement = 0
        
        return State.SETUP.value  # Use enum value

class CountdownState:
    def __init__(self):
        self.countdown_value = 3
        self.countdown_frames = 0
        print("=== COUNTDOWN PHASE ===")
    
    def get_state_name(self):
        return State.COUNTDOWN.value
    
    def main(self, pose_landmarks):
        return self.update_countdown()
    
    def update_countdown(self):
        self.countdown_frames += 1
        
        if self.countdown_frames == 30:  # Assuming 30 FPS
            print(f"{self.countdown_value}")
            self.countdown_value -= 1
            self.countdown_frames = 0
            
            if self.countdown_value <= 0:
                print("GO!")
                return State.PROCESSING.value  # Use enum value
        
        return State.COUNTDOWN.value  # Use enum value

class ProcessingState:
    def __init__(self):
        print("=== PROCESSING PHASE ===")
    
    def get_state_name(self):
        return State.PROCESSING.value
    
    def main(self, pose_landmarks):
        # TODO: Add processing logic 
        return State.PROCESSING.value  # Use enum value

class EndingState:
    def __init__(self):
        print("=== ENDING PHASE ===")
    
    def get_state_name(self):
        return State.ENDING.value
    
    def main(self, pose_landmarks):
        # TODO: Add ending logic 
        return State.ENDING.value  # Use enum value 
   