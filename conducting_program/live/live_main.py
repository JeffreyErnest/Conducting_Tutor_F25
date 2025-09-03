# main.py - This files carries the logic for the live version of the program
#  

from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_analyzer import live_analyzer
# from conducting_program.live.settings import BPMSettings
from live.system_state import SystemState
from live.pose_landmarks import PoseLandmarks

def live_main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose instance
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe instance
    camera_manager = CameraManager() # Camera manager instance
    # bpm_settings = BPMSettings()  # Create BPM settings instance
    system_state = SystemState()  # Create start instance
    pose_landmarks = PoseLandmarks() 

    # Call main process
    live_analyzer(camera_manager, media_pipe_declaration, pose, system_state, pose_landmarks)
