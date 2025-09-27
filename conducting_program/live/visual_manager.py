# This will house the logic for 
# all the visuals / animation for the
# live detection.

import cv2

class ConductingGuide:
    """
    Visual guide for conducting practice.
    Note: Beat timing is now handled centrally in system_state.py
    """
    
    def __init__(self):
        """Initialize the visual guide."""
        self.beat_duration = 0.1  # How long the circle shows (100ms)
    
    def draw_beat_circle(self, frame, show_beat=False):
        """Draw the red circle overlay on the frame when a beat occurs."""
        if show_beat:
            # Get frame dimensions
            frame_height = frame.shape[0] # again as in live analyzer might want to make this a function somwhere
            frame_width = frame.shape[1]
            
            # Center of the frame
            center_x = frame_width // 2
            center_y = frame_height // 2
            
            # Draw red circle
            cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), -1)  # Filled red circle
            cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)  # White border