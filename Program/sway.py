from imports import *

class swayingDetection:
    def __init__(self):
        self.default_midpoint_x = 0
        self.sway_threshold = 0.025  # Threshold for swaying detection
        self.midpoints_x = []
        self.midpointflag = False

    def midpoint_calculation(self, x12, x11):
        self.midpoint_x = abs(x12 - x11) * 0.5 + x12
        self.midpoints_x.append(self.midpoint_x)

    def set_midpoint(self):
        if self.midpointflag:
            self.default_midpoint_x = self.midpoint_x
            self.set_midpoint_flag_false()

    def set_midpoint_flag_true(self):
        self.midpointflag = True

    def set_midpoint_flag_false(self):
        self.midpointflag = False

    def swaying_print(self, frame_index, annotated_image_bgr):
        if frame_index < len(self.midpoints_x):
            midpoint_x = self.midpoints_x[frame_index]
            if midpoint_x > self.default_midpoint_x + self.sway_threshold or midpoint_x < self.default_midpoint_x - self.sway_threshold:
                cv2.putText(annotated_image_bgr, "Swaying", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        return