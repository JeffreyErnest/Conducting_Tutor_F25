import cv2

class CameraManager:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        
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
    
    def cleanup(self):
        # Clean up camera resources
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()