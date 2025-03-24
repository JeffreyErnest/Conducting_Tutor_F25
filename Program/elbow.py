# This Files includes the logic for dectecting
# the movement of the elbow and printing "watch elbow"

from imports import *
import math

class elbowDetection:

    def __init__(self):
        self.elbow_coords = []
        self.shoulder_coords = []
        self.hip_coords = []

    def elbow_calculation(self, elbow, shoulder, hip):
        self.elbow_coords.append(elbow)
        self.shoulder_coords.append(shoulder)
        self.hip_coords.append(hip)
    
    def elbow_print(self, frame_index, annotated_image_bgr):
        angle = self.calculate_angle(frame_index) 

        angle_threshold = 32
        if angle is not None and angle > angle_threshold:
            cv2.putText(annotated_image_bgr, "Watch Elbow", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) #type: ignore
        return
    
    def calculate_angle(self, frame_index):

        # Check if frame_index is within the valid range
        if frame_index < 0 or frame_index >= len(self.shoulder_coords):
            print(f"Frame index {frame_index} is out of range. Returning None.")
            return None  # or return 0, or any default value you prefer

        A = self.shoulder_coords[frame_index]   # Shoulder coordinates
        B = self.elbow_coords[frame_index]      # Elbow coordinates
        C = self.hip_coords[frame_index]        # Hip coordinates

        # Invert the y-coordinates 
        A_inverted = (A[0], -A[1])
        B_inverted = (B[0], -B[1])
        C_inverted = (C[0], -C[1])

        # Calculate vectors using the inverted coordinates
        AB = (B_inverted[0] - A_inverted[0], B_inverted[1] - A_inverted[1])  # Vector AB (shoulder to elbow)
        AC = (C_inverted[0] - A_inverted[0], C_inverted[1] - A_inverted[1])  # Vector AC (shoulder to hip)
        
        # Calculate dot product of AB and AC
        dot_product = AB[0] * AC[0] + AB[1] * AC[1]
        
        # Calculate magnitudes of AB and AC
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_AC = math.sqrt(AC[0]**2 + AC[1]**2)
        
        # Cosine of the angle
        cos_theta = dot_product / (magnitude_AB * magnitude_AC)
        
        # Ensure the cosine value is within the valid range for acos due to floating point precision
        cos_theta = max(-1, min(1, cos_theta))
        
        # Calculate the angle in radians
        angle_radians = math.acos(cos_theta)
        
        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        # print(f"Calculated angle: {angle_degrees}")
        
        return angle_degrees