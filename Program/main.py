from imports import *

# handles the first pass through the video, detecting conducting movements and beats
class cycleOne: 

    # initializes the first cycle, setting up video capture and processing parameters
    def __init__(self):

        # get mediapipe detector
        self.detector = mediaPipeDeclaration.get_pose_landmarker()
        self.videoFileName = initialize_video()
        
        # initialize video capture
        self.cap = cv2.VideoCapture(self.videoFileName)
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        # initialize tracking arrays
        self.frame_array = []
        self.processed_frame_array = []
        self.processing_intervals = []

        # initialize movement detectors
        self.swaying_detector = swayingDetection()
        self.mirror_detector = mirrorDetection()

        # setup video writer
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(video_beat_plot_name() + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.frame_width, self.frame_height))

        # Add debugging info after video capture initialization
        print("\n=== Cycle One Debug Information ===")
        print(f"Video File: {self.videoFileName}")
        print(f"Frame Width: {self.frame_width}")
        print(f"Frame Height: {self.frame_height}")
        print(f"FPS: {self.fps}")
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total Frames: {total_frames}")
        print(f"Video Duration: {total_frames/self.fps:.2f} seconds")
        print("================================\n")

        # process video and detect beats
        process_video(self.cap, self.out, self.detector, self.frame_array, self.processed_frame_array, self.processing_intervals, self.swaying_detector, self.mirror_detector)
        
        # analyze detected movements for beats
        (self.filtered_significant_beats, self.beat_coordinates, self.x_peaks, self.x_valleys, self.y_peaks, self.y_valleys, self.x, self.y) = filter_beats(self.frame_array, self.processed_frame_array)

        # After beat detection, add more debug info
        print("\n=== Beat Detection Results ===")
        print(f"Total frames processed: {len(self.frame_array)}")
        print(f"Number of beats detected: {len(self.filtered_significant_beats)}")
        print(f"Processing intervals: {self.processing_intervals}")
        print("============================\n")


# handles the second pass through the video, visualizing detected beats and generating analysis
class cycleTwo: 

    # initializes the second cycle, using data from cycle one to create visualizations
    def __init__(self, cycle_one_instance):

        # get mediapipe detector
        self.detector = mediaPipeDeclaration.get_pose_landmarker()
        self.videoFileName = initialize_video()
        self.cap = cv2.VideoCapture(self.videoFileName)

        # reuse swaying detector from cycle one
        self.swaying_detector = cycle_one_instance.swaying_detector
        self.mirror_detector = cycle_one_instance.mirror_detector
        self.pattern_detector = patternDetection()

        # setup video writer
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))    
        self.out = cv2.VideoWriter(video_out_name() + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.frame_width, self.frame_height))
        
        # process video with detected beats
        output_process_video(self.cap, self.out, self.detector, cycle_one_instance.filtered_significant_beats, cycle_one_instance.processing_intervals, self.swaying_detector)
        
        # Detect patterns and write to file
        patterns = self.pattern_detector.pattern_detection(cycle_one_instance.beat_coordinates)
        with open(self.videoFileName + "_video_pattern.txt", "w") as f:
            for pattern in patterns:
                f.write(pattern + "\n")

        # generate analysis graphs
        generate_all_graphs(cycle_one_instance)
        
        cv2.destroyAllWindows()
    
# main execution point of the program
if __name__ == "__main__":
    cycle_one_instance = cycleOne()
    cycle_two_instance = cycleTwo(cycle_one_instance)