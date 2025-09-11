
class SwayDetection():
    def __init__(self):
        self.sway_threshold = 0.025  # Threshold for swaying detection
        self.sway_threshold_left = None
        self.sway_threshold_right = None
        self.sway_flag = False

    def get_threshold_left(self):
        return self.sway_threshold_left
        
    def get_threshold_right(self):
        return self.sway_threshold_right

    def get_sway_flag(self): 
        return self.sway_flag

    def set_sway_flag(self, new_midpoint):
        # Guard for uninitialized thresholds or new_midpoint
        if self.sway_threshold_left is None or self.sway_threshold_right is None or new_midpoint is None:
            self.sway_flag = False
            return
        self.sway_flag = (new_midpoint > self.sway_threshold_left or new_midpoint < self.sway_threshold_right)
        return

    def main(self, midpoint, new_midpoint):

        if midpoint is None:
            self.sway_threshold_left = None
            self.sway_threshold_right = None
            self.sway_flag = False
            return
        self.sway_threshold_left = midpoint + self.sway_threshold
        self.sway_threshold_right = midpoint - self.sway_threshold

        # Check if we are swaying
        self.set_sway_flag(new_midpoint)

