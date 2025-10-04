
class ElbowDetection: 
    def __init__(self):
        self.angle_threshold = 32 # TODO: Maybe make this a setting, or somehow set from setup state
        self.watch_left_elbow = False
        self.watch_right_elbow = False

    def get_watch_left_elbow(self):
        return self.watch_left_elbow
    
    def get_watch_right_elbow(self):
        return self.watch_right_elbow

    def _check_left_elbow(self, pose_landmarks):
        left_angle = pose_landmarks.get_left_angle()
        if left_angle is not None and left_angle > self.angle_threshold:
            if not self.watch_left_elbow:
                self.watch_left_elbow = True
        else:
            if self.watch_left_elbow:
                self.watch_left_elbow = False
        
    def _check_right_elbow(self, pose_landmarks):
        right_angle = pose_landmarks.get_right_angle()
        if right_angle is not None and right_angle > self.angle_threshold:
            if not self.watch_right_elbow:
                self.watch_right_elbow = True
        else:
            if self.watch_right_elbow:
                self.watch_right_elbow = False

    def main(self, pose_landmarks):
        # Calculate both left and right arm angles
        pose_landmarks.calculate_left_angle()
        pose_landmarks.calculate_right_angle()

        self._check_left_elbow(pose_landmarks)  # Check left elbow
        self._check_right_elbow(pose_landmarks) # Check right elbow
