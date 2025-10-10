import math

# Landmarks are written in reverse order, so the right side is the left side
# and the left side is the right side (this is because the camera is flipped)
class PoseLandmarks:
    def __init__(self):
        self.right_wrist_15 = (None, None)
        self.left_wrist_16 = (None, None)

        self.right_shoulder_11 = (None, None)
        self.left_shoulder_12 = (None, None)

        self.right_elbow_13 = (None, None)
        self.left_elbow_14 = (None, None)

        self.right_hip_23 = (None, None)
        self.left_hip_24 = (None, None)

        self.midpoint_x_axis = 0
        self.left_angle = None
        self.right_angle = None

    def update_landmarks(self, detection_result):
        if detection_result and detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks.landmark
            if len(landmarks) > 16:
                # Hands
                self.right_wrist_15 = (landmarks[15].x, landmarks[15].y)
                self.left_wrist_16 = (landmarks[16].x, landmarks[16].y)
                # Shoulders
                self.right_shoulder_11 = (landmarks[11].x, landmarks[11].y)
                self.left_shoulder_12 = (landmarks[12].x, landmarks[12].y)
                # Elbows 
                self.right_elbow_13 = (landmarks[13].x, landmarks[13].y)
                self.left_elbow_14 = (landmarks[14].x, landmarks[14].y)
                # Hips
                self.right_hip_23 = (landmarks[23].x, landmarks[23].y)
                self.left_hip_24 = (landmarks[24].x, landmarks[24].y)
        else:
            # Reset landmarks when no detection
            self.right_wrist_15 = (None, None)
            self.left_wrist_16 = (None, None)
            self.right_shoulder_11 = (None, None)
            self.left_shoulder_12 = (None, None)
            self.right_elbow_13 = (None, None)
            self.left_elbow_14 = (None, None)
            self.right_hip_23 = (None, None)
            self.left_hip_24 = (None, None)

    def get_pose_landmark_15(self):
        return self.right_wrist_15
    
    def get_pose_landmark_16(self):
        return self.left_wrist_16
    
    def get_pose_landmark_14(self):
        return self.left_elbow_14
    
    def get_pose_landmark_13(self):
        return self.right_elbow_13
    
    def get_pose_landmark_24(self):
        return self.left_hip_24
    
    def get_pose_landmark_23(self):
        return self.right_hip_23

    def calculate_midpoint(self):
        if (self.left_shoulder_12[0] is not None and self.right_shoulder_11[0] is not None):
            self.midpoint_x_axis = abs(self.right_shoulder_11[0] - self.left_shoulder_12[0]) * 0.5 + self.left_shoulder_12[0]
    
    def get_midpoint(self):
        return self.midpoint_x_axis

    def calculate_left_angle(self):
        """
        Calculate the left armpit angle (arm to torso) using three points:
        - Shoulder (index 12) -> Elbow (index 14) -> Hip (index 24)
        This measures the angle between your left arm and your torso.
        Returns angle in degrees, or None if insufficient data.
        """
        # Check if all landmarks are available
        if (self.left_shoulder_12[0] is None or self.left_elbow_14[0] is None or self.left_hip_24[0] is None):
            self.left_angle = None
            return None
        
        # Get landmark coordinates
        shoulder_x, shoulder_y = self.left_shoulder_12
        elbow_x, elbow_y = self.left_elbow_14
        hip_x, hip_y = self.left_hip_24
        
        # Create vectors from shoulder point (the vertex of the angle)
        # Vector 1: shoulder -> elbow (arm direction)
        arm_x = elbow_x - shoulder_x
        arm_y = elbow_y - shoulder_y
        
        # Vector 2: shoulder -> hip (torso direction)
        torso_x = hip_x - shoulder_x
        torso_y = hip_y - shoulder_y
        
        # Calculate angle between arm and torso vectors
        angle_degrees = self._calculate_vector_angle(arm_x, arm_y, torso_x, torso_y)
        
        if angle_degrees is not None:
            self.left_angle = int(angle_degrees)
        else:
            self.left_angle = None

    def calculate_right_angle(self):
        """
        Calculate the right armpit angle (arm to torso) using three points:
        - Shoulder (index 11) -> Elbow (index 13) -> Hip (index 23)
        This measures the angle between your right arm and your torso.
        Returns angle in degrees, or None if insufficient data.
        """
        # Check if all landmarks are available
        if (self.right_shoulder_11[0] is None or self.right_elbow_13[0] is None or self.right_hip_23[0] is None):
            self.right_angle = None
            return None
        
        # Get landmark coordinates
        shoulder_x, shoulder_y = self.right_shoulder_11
        elbow_x, elbow_y = self.right_elbow_13
        hip_x, hip_y = self.right_hip_23
        
        # Create vectors from shoulder point (the vertex of the angle)
        # Vector 1: shoulder -> elbow (arm direction)
        arm_x = elbow_x - shoulder_x
        arm_y = elbow_y - shoulder_y
        
        # Vector 2: shoulder -> hip (torso direction)
        torso_x = hip_x - shoulder_x
        torso_y = hip_y - shoulder_y
        
        # Calculate angle between arm and torso vectors
        angle_degrees = self._calculate_vector_angle(arm_x, arm_y, torso_x, torso_y)
        
        if angle_degrees is not None:
            self.right_angle = int(angle_degrees)
        else:
            self.right_angle = None
    
 
    def _calculate_vector_angle(self, v1_x, v1_y, v2_x, v2_y):
        """Calculate angle between two vectors in degrees."""
        # Calculate vector magnitudes
        magnitude_1 = math.hypot(v1_x, v1_y)
        magnitude_2 = math.hypot(v2_x, v2_y)
        
        # Avoid division by zero
        if magnitude_1 == 0 or magnitude_2 == 0:
            return None
        
        # Calculate dot product
        dot_product = v1_x * v2_x + v1_y * v2_y
        
        # Calculate cosine of angle 
        cos_angle = dot_product / (magnitude_1 * magnitude_2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Convert to degrees
        angle_radians = math.acos(cos_angle)
        return math.degrees(angle_radians)

    def get_left_angle(self):
        return self.left_angle
    
    def get_right_angle(self):
        return self.right_angle