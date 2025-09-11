# pose_tracker.py
class PoseLandmarks:
    def __init__(self):
        self.left_wrist_15 = None
        self.right_wrist_16 = None

        self.left_shoulder_12 = None
        self.right_shoulder_11 = None

        self.midpoint_x_axis = 0

    def update_landmarks(self, detection_result):
        if detection_result and detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks.landmark
            if len(landmarks) > 16:
                self.left_wrist_15 = (landmarks[15].x, landmarks[15].y)
                self.right_wrist_16 = (landmarks[16].x, landmarks[16].y)
                self.left_shoulder_12 = (landmarks[12].x, landmarks[12].y)
                self.right_shoulder_11 = (landmarks[11].x, landmarks[11].y)
        else:
            # Reset landmarks when no detection
            self.left_wrist_15 = None
            self.right_wrist_16 = None
    
    def get_pose_landmark_15(self):
        if self.left_wrist_15 is None:
            return (None, None)  # Return tuple when no data
        return self.left_wrist_15
    
    def get_pose_landmark_16(self):
        if self.right_wrist_16 is None:
            return (None, None)  # Return tuple when no data
        return self.right_wrist_16
    
    def calculate_midpoint(self):
        if self.left_shoulder_12 and self.right_shoulder_11:
            self.midpoint_x_axis = abs(self.left_shoulder_12[0] - self.right_shoulder_11[0]) * 0.5 + self.left_shoulder_12[0]
    
    def get_midpoint(self):
        return self.midpoint_x_axis

    