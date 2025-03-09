# This file includes the logic for the live start of the program. In which a user
# brings the band to attention; this will start the tacking in the program.

from imports import *

class startEndDetection:

    def __init__(self):
        self.processing_active = False
        self.current_start_frame = None
        self.previous_y_left = None
        self.previous_x_right = None
        self.previous_x_left = None
        self.previous_y_right = None
        self.frame_count_since_movement = 0
        self.slight_movement_threshold = .005
        self.movement_counter = 0

    def start_processing(self, left_y, right_y):

        if self.previous_y_left is None or self.previous_y_right is None:
            self.previous_y_left = left_y
            self.previous_y_right = right_y

        significant_movement_threshold = 0.1  

        # Check for significant upward movement for both left and right
        left_moved_up = left_y < self.previous_y_left - significant_movement_threshold
        right_moved_up = right_y < self.previous_y_right - significant_movement_threshold
        
        # Check for significant downward movement for both left and right
        left_dropped_down = left_y > self.previous_y_left + significant_movement_threshold
        right_dropped_down = right_y > self.previous_y_right + significant_movement_threshold

        # Determine if both hands are up
        both_hands_up = (left_moved_up and right_moved_up) and not (left_dropped_down or right_dropped_down)

        if both_hands_up and not self.processing_active:
            if self.frame_count_since_movement < 30: # Requires a full second (change to frame of video, later)
               self.frame_count_since_movement += 1  # Increment frame count
            else:
                print ("Started Processing")
                self.processing_active = True
        elif left_dropped_down or right_dropped_down:  # Check if hands have dropped
            self.frame_count_since_movement = 0  # Reset the frame count

        return self.processing_active  
    

    def end_processing(self, left_hand_x, right_hand_x, left_hand_y, right_hand_y ):

        # Invert the y-axis values
        left_hand_y = -left_hand_y
        right_hand_y = -right_hand_y

        left_significant_movement = (abs(left_hand_y - self.previous_y_left) > self.slight_movement_threshold or
                                     abs(left_hand_x - self.previous_x_left) > self.slight_movement_threshold)
        right_significant_movement = (abs(right_hand_y - self.previous_y_right) > self.slight_movement_threshold or
                                      abs(right_hand_x - self.previous_x_right) > self.slight_movement_threshold)
        
        no_movement_check = not (left_significant_movement or right_significant_movement)  # Check for no movement

        # print (f" Left: {left_significant_movement} right: {right_significant_movement}")

        if no_movement_check:
            self.movement_counter += 1  # Increment counter if no movement detected
        else:
            self.movement_counter = 0  # Reset counter if movement is detected
            
        # print (f"no movement check: {no_movement_check} counter is at {self.movement_counter}")

        # Check if there is no movement for 2 seconds (60 frames)
        no_movement_for_2_seconds = self.movement_counter >= 60

        # Check if hands have crossed
        hands_crossed = left_hand_x > right_hand_x
        
        # Use the flags in the condition
        if self.processing_active and (hands_crossed or no_movement_for_2_seconds):
            print("Ended processing.")
            self.processing_active = False

        self.previous_y_left = left_hand_y
        self.previous_x_left = left_hand_x
        self.previous_y_right = right_hand_y
        self.previous_x_right = right_hand_x
        
        return self.processing_active