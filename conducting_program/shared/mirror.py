
class MirrorDetection(): 
    def __init__(self):
        self.x15 = None
        self.y15 = None 
        self.x16 = None
        self.y16 = None

        self.y_threshold = .075
        self.x_threshold = .05

        self.before_starting = None
        self.before_ending = None
        self.mirroring_flag = False

    def mirror_on_y(self):
        
        if abs(self.y15 - self.y16) < self.y_threshold:
            return True
        else:
            return False

    def mirror_on_x(self, current_midpoint):

        if abs(abs(self.x15 - current_midpoint) - abs(self.x16 - current_midpoint)) < self.x_threshold:
            return True
        else: 
            return False

    def buffer_start_time(self, current_time, interval_seconds=0.5):
        
        if self.before_starting is None:
            self.before_starting = current_time
            return False  # Need to wait the full interval

        return (current_time - self.before_starting) >= interval_seconds

    def buffer_end_time(self, current_time, interval_seconds=0.5):
        
        if self.before_ending is None:
            self.before_ending = current_time
            return False  # Need to wait the full interval

        return (current_time - self.before_ending) >= interval_seconds

    def main(self, pose_landmarks, clock_manager, current_midpoint):

        self.x15, self.y15 = pose_landmarks.get_pose_landmark_15()
        self.x16, self.y16 = pose_landmarks.get_pose_landmark_16()
        
        current_time = clock_manager.get_current_timestamp()

        # Check if we have valid wrist data
        if self.x15 is None or self.y15 is None or self.x16 is None or self.y16 is None:
            return

        # Check if currently mirroring
        if self.mirror_on_y() and self.mirror_on_x(current_midpoint): 
            is_mirroring = True
        else: 
            is_mirroring = False

        if is_mirroring:
            if not self.mirroring_flag:
                # Start buffering for mirroring
                if self.buffer_start_time(current_time, 0.5):
                    self.mirroring_flag = True
                    print("Mirroring")
            else:
                # Already confirmed mirroring - keep printing
                print("Mirroring")
                # Reset end buffer since we're still mirroring
                self.before_ending = None
        else:
            # Not mirroring
            if self.mirroring_flag:
                # Start buffering for end of mirroring
                if self.buffer_end_time(current_time, 0.5):
                    self.mirroring_flag = False
                    # Reset buffers for next cycle
                    self.before_starting = None
                    self.before_ending = None
                else:
                    # Still in end buffer - keep printing
                    print("Mirroring")
            else:
                # Not mirroring and not flagged - reset start buffer
                self.before_starting = None

    def get_mirroring_flag(self):
        return self.mirroring_flag