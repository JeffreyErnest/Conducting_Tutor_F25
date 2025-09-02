# This file will house the settings for the live detection program, such as:
# Do we want visuals, to ender the feedback
# save the video, etc...
# As well as constants that are set by the user.

# Setting that controls BPM TODO maybe use a config?
class BPMSettings:
    def __init__(self):
        self.beats_per_minute = 0
    
    def set_beats_per_minute(self, passed_bpm): # Set BPM
        self.beats_per_minute = passed_bpm
    
    def get_beats_per_minute(self): # Get BPM
        return self.beats_per_minute
