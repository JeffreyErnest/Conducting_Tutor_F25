# main.py - This files carries the logic for the live version of the program
from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_display import processing_loop
from live.settings import BPMSettings
from live.system_state import SystemState
from live.pose_landmarks import PoseLandmarks
from live.clock_manager import ClockManager

def live_main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose instance
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe instance
    camera_manager = CameraManager() # Camera manager instance
    bpm_settings = BPMSettings()
    system_state = SystemState()  # Create start instance
    pose_landmarks = PoseLandmarks() 
    clock_manager = ClockManager()  # Create clock manager instance
    
    bpm_input = input("enter bpm as a whole number: ")
    bpm_settings.set_beats_per_minute(bpm_input)
    print(bpm_settings.get_beats_per_minute())
    
    # Call main process
    processing_loop(camera_manager, media_pipe_declaration, pose, system_state, pose_landmarks, clock_manager, bpm_settings)
