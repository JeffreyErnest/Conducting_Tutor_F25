# main.py - This files carries the logic for the live version of the program
from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_display import processing_loop
from live.settings import Settings
from live.system_state import SystemState
from live.pose_landmarks import PoseLandmarks
from live.clock_manager import ClockManager

def live_main():

    # Initialize settings
    settings = Settings()
    bpm_input = input("enter bpm as a whole number: ")
    settings.set_beats_per_minute(bpm_input)
    print(f"BPM set to: {settings.get_beats_per_minute()}")
    
    # Initialize components
    pose = mediaPipeDeclaration.initialize_pose_detection()
    media_pipe_declaration = mediaPipeDeclaration()
    camera_manager = CameraManager()
    system_state = SystemState(settings)  # Pass settings to SystemState
    pose_landmarks = PoseLandmarks() 
    clock_manager = ClockManager()
    
    # Call main process
    processing_loop(camera_manager, media_pipe_declaration, pose, system_state, pose_landmarks, clock_manager, settings)
