from imports import *

class mirrorDetection:

    # Lists to store hand coordinates
    def __init__(self):
        self.left_hand_x = []
        self.left_hand_y = []
        self.right_hand_x = []
        self.right_hand_y = []
        # Add variables to track the last processed position
        self.last_left_x = None
        self.last_left_y = None
        self.last_right_x = None
        self.last_right_y = None

    # Used to pass coordinates to pattern detection
    def get_coordinates(self):
        return self.left_hand_x, self.left_hand_y, self.right_hand_x, self.right_hand_y

    def mirror_calculation(self, left_x, left_y, right_x, right_y):
        # Store all coordinates for graphing
        self.left_hand_x.append(left_x)
        self.left_hand_y.append(left_y)
        self.right_hand_x.append(right_x)
        self.right_hand_y.append(right_y)
        
        # Update last processed position
        self.last_left_x = left_x
        self.last_left_y = left_y
        self.last_right_x = right_x
        self.last_right_y = right_y

    def get_last_positions(self):
        # Return last positions for movement detection
        return (self.last_left_x, self.last_left_y, 
                self.last_right_x, self.last_right_y)