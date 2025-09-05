# This will house the logic
# for the live start of the program, 
# as well have infomation on the body ouline
# so people know how far to stand from the camera. 

from enum import Enum

class State(Enum): # Set Enum values
    SETUP = "setup"
    COUNTDOWN = "countdown"
    PROCESSING = "processing"
    ENDING = "ending"

class SystemState:
    def __init__(self):
        self.current_state = SetupState()  # Start with setup state
    
    def get_current_state(self):
        return self.current_state
    
    def change_state(self, new_state, clock_manager=None):
        if new_state == State.COUNTDOWN.value:
            self.current_state = CountdownState()
        elif new_state == State.PROCESSING.value:
            self.current_state = ProcessingState()
            clock_manager.start_session_clock()  # Start session timing for processing
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
        self.movement_start_time = None
        self.movement_hold_duration = 1.0  # Hold hands up for 1 second
        self.significant_movement_threshold = 0.1
        print("=== SETUP PHASE ===")
    
    def get_state_name(self):
        return State.SETUP.value
   
    def main(self, pose_landmarks, clock_manager): 
        return self.wait_for_start_movement(pose_landmarks, clock_manager)

    def wait_for_start_movement(self, pose_landmarks, clock_manager):
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
            if self.movement_start_time is None:
                self.movement_start_time = clock_manager.get_current_timestamp()
            elif clock_manager.get_current_timestamp() - self.movement_start_time >= self.movement_hold_duration:
                print("Starting Countdown")
                self.processing_active = True
                return State.COUNTDOWN.value  # Use enum value
        elif left_dropped_down or right_dropped_down:
            self.movement_start_time = None  # Reset movement timer
        
        return State.SETUP.value  # Use enum value

class CountdownState:
    def __init__(self):
        self.countdown_value = 3
        self.countdown_start_time = None
        self.countdown_interval = 1.0  # 1 second between countdown numbers
        print("=== COUNTDOWN PHASE ===")
    
    def get_state_name(self):
        return State.COUNTDOWN.value
    
    def main(self, pose_landmarks, clock_manager):
        return self.update_countdown(clock_manager)
    
    def update_countdown(self, clock_manager):
        if self.countdown_start_time is None:
            self.countdown_start_time = clock_manager.get_current_timestamp()
            print(f"{self.countdown_value}")
            return State.COUNTDOWN.value
        
        elapsed_time = clock_manager.get_current_timestamp() - self.countdown_start_time
        expected_countdown = 3 - int(elapsed_time)
        
        if expected_countdown != self.countdown_value and expected_countdown >= 0:
            self.countdown_value = expected_countdown
            print(f"{self.countdown_value}")
        
        if elapsed_time >= 3.0:  # After 3 seconds
            print("GO!")
            return State.PROCESSING.value  # Use enum value
        
        return State.COUNTDOWN.value  # Use enum value

class ProcessingState:
    def __init__(self):
        print("=== PROCESSING PHASE ===")
    
    def get_state_name(self):
        return State.PROCESSING.value
    
    def main(self, pose_landmarks, clock_manager):
        # TODO: Add processing logic 
        return State.PROCESSING.value  # Use enum value

class EndingState:
    def __init__(self):
        print("=== ENDING PHASE ===")
    
    def get_state_name(self):
        return State.ENDING.value
    
    def main(self, pose_landmarks, clock_manager):
        # TODO: Add ending logic 
        return State.ENDING.value  # Use enum value 
   