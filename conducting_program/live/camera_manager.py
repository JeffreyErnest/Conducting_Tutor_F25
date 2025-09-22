import cv2
import time

class CameraManager:

    def __init__(self, camera_index="C:/Users/Jeffrey Ernest/Desktop/videos/Marchingband(2).mp4"):
        self.camera_index = camera_index
        self.cap = None
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
    def initialize_camera(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
            
        return True
    
    def capture_frame(self):
        # Capture a frame from the camera
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame from camera")
            return False, None
            
        return True, frame
    
    def convert_to_rgb(self, frame):
        # Convert BGR frame to RGB (MediaPipe expects RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def calculate_fps(self):
        # Calculate current FPS based on frame timing
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        return int(fps)
    
    def cleanup(self):
        # Clean up camera resources
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    