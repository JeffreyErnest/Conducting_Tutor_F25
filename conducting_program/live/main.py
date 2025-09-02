# main.py - This files carries the logic for the live version of the program
#  

from mp_declaration import mediaPipeDeclaration
from camera_manager import CameraManager
from live_analyzer import live_analyzer

def main():
    # Initialize methods
    pose = mediaPipeDeclaration.initialize_pose_detection() # MediaPipe Pose initilizer
    media_pipe_declaration = mediaPipeDeclaration() # MediaPipe initilizer
    camera_manager = CameraManager() # Camera manager initilizer
    
    # Call main process
    live_analyzer(camera_manager, media_pipe_declaration, pose)


if __name__ == "__main__":
    main()
