# This file will house the settings for the live detection program, such as:
# Do we want visuals, to ender the feedback
# save the video, etc...
# As well as constants that are set by the user.

# Settings for live conducting program
class Settings:
    def __init__(self):
        self.beats_per_minute = 60  # Default BPM
        self.time_signature = "4/4"  # Default time signature
    
    def set_beats_per_minute(self, bpm):
        self.beats_per_minute = int(bpm)
    
    def get_beats_per_minute(self):
        return self.beats_per_minute
    
    def set_time_signature(self, time_sig):
        self.time_signature = time_sig
    
    def get_time_signature(self):
        return self.time_signature
