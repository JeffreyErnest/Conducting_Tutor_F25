# pose_tracker.py
class PoseLandmarks:
    def __init__(self):
        self.left_wrist_15 = None
        self.right_wrist_16 = None
        # self.left_elbow_14 = None
        # self.right_elbow = None
        # self.left_shoulder = None
        # self.right_shoulder = None
        # # ... more landmark points
    
    def update_landmarks(self, detection_result):
        if detection_result and detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks.landmark
            if len(landmarks) > 16:
                self.left_wrist_15 = (landmarks[15].x, landmarks[15].y)
                self.right_wrist_16 = (landmarks[16].x, landmarks[16].y)
                # self.left_elbow_14 = (landmarks[14].x, landmarks[14].y)
                # self.right_elbow_13 = (landmarks[13].x, landmarks[13].y)
                # ... more landmarks
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
    
