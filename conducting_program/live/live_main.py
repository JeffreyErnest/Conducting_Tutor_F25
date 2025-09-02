# main.py - This files carries the logic for the live version of the program
#  

from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_analyzer import live_analyzer
from live.settings import BPMSettings
from live.start import LiveStart

def live_main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose instance
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe instance
    camera_manager = CameraManager() # Camera manager instance
    bpm_settings = BPMSettings()  # Create BPM settings instance
    live_start = LiveStart()  # Create start instance

    # Call main process
    live_analyzer(camera_manager, media_pipe_declaration, pose, bpm_settings, live_start)
