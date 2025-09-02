# main.py - This files carries the logic for the live version of the program
#  

from live.mp_declaration import mediaPipeDeclaration
from live.camera_manager import CameraManager
from live.live_analyzer import live_analyzer

def live_main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose initilizer
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe initilizer
    camera_manager = CameraManager() # Camera manager initilizer
    
    # Call main process
    live_analyzer(camera_manager, media_pipe_declaration, pose)
