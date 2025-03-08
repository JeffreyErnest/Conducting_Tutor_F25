from imports import *

class mirrorDetection:

    # Lists to store hand coordinates
    def __init__(self):
        self.left_hand_x = []
        self.left_hand_y = []
        self.right_hand_x = []
        self.right_hand_y = []
        self.is_mirroring = False  # Flag to track mirroring state
        self.mirroring_history = []  # Track mirroring state over frames
        self.movement_history = []  # Track movement similarity over frames
        self.mirroring_count = 0  # Count of consecutive frames mirroring is detected

    # Used to pass coordinates to pattern detection
    def get_coordinates(self):
        return self.left_hand_x, self.left_hand_y, self.right_hand_x, self.right_hand_y

    def mirror_calculation(self, left_x, left_y, right_x, right_y):
        # Store all coordinates for graphing
        self.left_hand_x.append(left_x)
        self.left_hand_y.append(left_y)
        self.right_hand_x.append(right_x)
        self.right_hand_y.append(right_y)

    def detect_mirroring(self, left_x, right_x, left_y, right_y, midpoint_x):
        # Calculate distances from the midpoint
        left_distance = abs(left_x - midpoint_x)
        right_distance = abs(right_x - midpoint_x)

        # Check if distances are similar
        distance_threshold = 0.1
        similar_distance = abs(left_distance - right_distance) < distance_threshold

        # Check if y movements are similar
        y_movement_difference = left_y - right_y
        y_movement_similar = abs(y_movement_difference) < 0.075  

        # Determine if mirroring is happening
        current_mirroring = similar_distance and y_movement_similar

        # Update histories
        self.mirroring_history.append(current_mirroring)
        self.movement_history.append(similar_distance and y_movement_similar)

        # Keep only the last 5 and 10 frames
        if len(self.mirroring_history) > 5:
            self.mirroring_history.pop(0)
        if len(self.movement_history) > 10:
            self.movement_history.pop(0)

        # Check if mirroring has been detected for 15 frames
        if current_mirroring:
            self.mirroring_count += 1  # Increment count if currently mirroring
        else:
            self.mirroring_count = 0  # Reset count if not currently mirroring

        # Determine if mirroring should be active
        if self.mirroring_count >= 15:
            self.is_mirroring = True  # Set to true if mirroring for 15 frames
        elif self.is_mirroring and not any(self.mirroring_history[-5:]):
            # If currently mirroring, check the last 5 frames
            self.is_mirroring = False  # Turn off if no mirroring for 5 frames

        return self.is_mirroring

    def print_mirroring(self, frame_index, annotated_image_bgr, midpoint_x):
        if frame_index < len(self.left_hand_x) and frame_index < len(self.right_hand_x):
            left_x = self.left_hand_x[frame_index]
            left_y = self.left_hand_y[frame_index]
            right_x = self.right_hand_x[frame_index]
            right_y = self.right_hand_y[frame_index]

            # Check for mirroring using the current hand coordinates and the midpoint from swaying
            self.is_mirroring = self.detect_mirroring(left_x, right_x, left_y, right_y, midpoint_x)

            if self.is_mirroring:
                cv2.putText(annotated_image_bgr, "Mirroring", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        return