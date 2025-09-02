# main.py - This files carries the logic for the live version of the program
#  

from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_analyzer import live_analyzer
from live.settings import BPM_settings

def live_main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose instance
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe instance
    camera_manager = CameraManager() # Camera manager instance
    bpm_settings = BPM_settings()  # Create BPM settings instance

    # Setup (make this a method later on)
    bpm = input("Enter your target BPM: ")
    bpm_settings.set_beats_per_minute(int(bpm))  # Convert input to int
    print(f"BPM set to: {bpm_settings.get_beats_per_minute()}")

    
    # Call main process
    live_analyzer(camera_manager, media_pipe_declaration, pose)
