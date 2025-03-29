"""
Main entry point for the Movement Detection and Conducting Analysis application.
This file loads the interface and handles the processing workflow, combining
movement detection with conducting pattern analysis.
"""
import os
import sys
import json
import cv2
import numpy as np
import time
import traceback
from datetime import datetime
import subprocess

# Import interface module
from interface import run_interface

# Try to import conducting-specific modules, but don't fail if they're not available
try:
    from imports import *
    CONDUCTING_MODULES_AVAILABLE = True
except ImportError:
    CONDUCTING_MODULES_AVAILABLE = False
    print("Note: Conducting analysis modules not found, will run in movement detection mode only.")

# Add these helper functions at the top of main.py after imports
def print_checkbox_menu(options):
    """Display a checkbox menu for options"""
    print("\n=== Graph Selection Menu ===")
    print("Select which graphs to generate:")
    for i, (key, value) in enumerate(options.items(), 1):
        status = "[X]" if value else "[ ]"
        print(f"{i}. {status} {key.replace('generate_', '').replace('_', ' ').title()}")
    print("\nEnter the number to toggle selection, or 'c' to continue: ")

def get_checkbox_input(options):
    """Get user input for checkbox menu"""
    keys = list(options.keys())
    while True:
        choice = input("> ").strip().lower()
        if choice == 'c':
            return options
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                key = keys[idx]
                options[key] = not options[key]
                print_checkbox_menu(options)
            else:
                print(f"Please enter a number between 1 and {len(keys)}, or 'c' to continue")
        except ValueError:
            print("Please enter a number or 'c' to continue")

def select_graph_options():
    """Interactive menu for selecting graph options"""
    options = {
        "generate_beat_plot": True,
        "generate_hand_path": True,
        "generate_cluster_graph": True,
        "generate_overtime_graph": True,
        "generate_swaying_graph": True,
        "generate_mirror_x_graph": True,
        "generate_mirror_y_graph": True
    }
    
    print_checkbox_menu(options)
    return get_checkbox_input(options)

# Default configuration
DEFAULT_CONFIG = {
    "video_path": "",
    "export_path": "output",
    "crop_rect": [200, 150, 400, 300],
    "process_markers": [],
    "processing_options": {
        "conducting_analysis": CONDUCTING_MODULES_AVAILABLE,
        "graph_options": {
                    "generate_beat_plot": True,
                    "generate_hand_path": True,
                    "generate_cluster_graph": True,
                    "generate_overtime_graph": True,
                    "generate_swaying_graph": True,
                    "generate_mirror_x_graph": True,
                    "generate_mirror_y_graph": True
        }
    }
}

def load_config():
    """Load configuration from interface_config.json"""
    if os.path.exists("interface_config.json"):
        try:
            with open("interface_config.json", "r") as f:
                config = json.load(f)
                
                # Ensure processing_options contains conducting_analysis setting
                if "processing_options" in config:
                    if "conducting_analysis" not in config["processing_options"]:
                        config["processing_options"]["conducting_analysis"] = CONDUCTING_MODULES_AVAILABLE
                
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    return DEFAULT_CONFIG

def get_export_path():
    """Get export path from config file"""
    try:
        with open("interface_config.json", "r") as f:
            config = json.load(f)
            return config.get("export_path", "output")
    except:
        return "output"

def set_export_path(path):
    """Create export directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

# Conducting Analysis Classes

# handles the first pass through the video, detecting conducting movements and beats
class CycleOne: 
    # initializes the first cycle, setting up video capture and processing parameters
    def __init__(self, config):
        if not CONDUCTING_MODULES_AVAILABLE:
            print("Error: Conducting analysis modules not available")
            return
            
        # get mediapipe detector
        self.detector = mediaPipeDeclaration.get_pose_landmarker()
        self.videoFileName = config["video_path"]
        
        # initialize video capture
        self.cap = cv2.VideoCapture(self.videoFileName)
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        # initialize tracking arrays
        self.frame_array = []
        self.processed_frame_array = []
        
        # Get processing intervals from the configuration
        self.processing_intervals = config.get("process_markers", [])

        # initialize movement detectors
        self.swaying_detector = swayingDetection()
        self.mirror_detector = mirrorDetection()

        # setup video writer
        export_path = config["export_path"]
        os.makedirs(export_path, exist_ok=True)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Get crop settings if available
        crop_data = config.get("crop_rect", None)
        if crop_data:
            self.frame_width = crop_data[2]
            self.frame_height = crop_data[3]
        
        output_file = os.path.join(export_path, video_beat_plot_name() + '.mp4')
        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))

        # Add debugging info after video capture initialization
        print("\n=== Cycle One Debug Information ===")
        print(f"Video File: {self.videoFileName}")
        print(f"Frame Width: {self.frame_width}")
        print(f"Frame Height: {self.frame_height}")
        print(f"FPS: {self.fps}")
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total Frames: {total_frames}")
        print(f"Video Duration: {total_frames/self.fps:.2f} seconds")
        print(f"Processing Intervals: {self.processing_intervals}")
        print("================================\n")

        # process video and detect beats
        process_video(self.cap, self.out, self.detector, self.frame_array, self.processed_frame_array, 
                     self.processing_intervals, self.swaying_detector, self.mirror_detector)
        
        # analyze detected movements for beats
        (self.filtered_significant_beats, self.beat_coordinates, self.y_peaks, self.y_valleys, self.y, self.x) = filter_beats(self.frame_array, self.processed_frame_array)

        # After beat detection, add more debug info
        print("\n=== Beat Detection Results ===")
        print(f"Total frames processed: {len(self.frame_array)}")
        print(f"Number of beats detected: {len(self.filtered_significant_beats)}")
        print(f"Processing intervals: {self.processing_intervals}")
        print("============================\n")


# handles the second pass through the video, visualizing detected beats and generating analysis
class CycleTwo:
    # initializes the second cycle, using data from cycle one to create visualizations
    def __init__(self, cycle_one_instance, config):
        if not CONDUCTING_MODULES_AVAILABLE:
            print("Error: Conducting analysis modules not available")
            return
            
        # get mediapipe detector
        self.detector = mediaPipeDeclaration.get_pose_landmarker()
        self.videoFileName = config["video_path"]
        self.cap = cv2.VideoCapture(self.videoFileName)
        
        # reuse swaying detector from cycle one
        self.swaying_detector = cycle_one_instance.swaying_detector
        self.mirror_detector = cycle_one_instance.mirror_detector
        self.pattern_detector = patternDetection()
        
        # setup video writer
        export_path = config["export_path"]
        os.makedirs(export_path, exist_ok=True)
        
        # Get the frame dimensions from config
        self.frame_width = 268  # Default to cropped width from logs
        self.frame_height = 496  # Default to cropped height from logs
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Get crop settings if available to confirm dimensions
        crop_data = config.get("crop_rect", None)
        if crop_data:
            x, y, w, h = crop_data
            self.frame_width = w
            self.frame_height = h
            print(f"Using crop dimensions: {self.frame_width}x{self.frame_height}")
            
        output_file = os.path.join(export_path, video_out_name() + '.mp4')
        
        # Create VideoWriter with the cropped dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                                  (self.frame_width, self.frame_height))
        
        # Verify the writer is opened successfully
        if not self.out.isOpened():
            print("Error: Failed to open video writer")
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_file, fourcc, self.fps, 
                                     (self.frame_width, self.frame_height))
            if not self.out.isOpened():
                print("Error: Failed to open video writer with alternative codec")
                return
            
        # Add debugging info
        print("\n=== Cycle Two Initialization ===")
        print(f"Video File: {self.videoFileName}")
        print(f"Frame Width: {self.frame_width}")
        print(f"Frame Height: {self.frame_height}")
        print(f"FPS: {self.fps}")
        print(f"Output File: {output_file}")
        print(f"VideoWriter opened: {self.out.isOpened()}")
        print("================================\n")

        # Process video with detected beats
        output_process_video(self.cap, self.out, self.detector, 
                           cycle_one_instance.filtered_significant_beats,
                           cycle_one_instance.processing_intervals, 
                           self.swaying_detector)
        
        # Detect patterns and write to files
        patterns = self.pattern_detector.pattern_detection(cycle_one_instance.beat_coordinates)
        pattern_file = os.path.join(export_path, os.path.basename(self.videoFileName) + "_video_pattern.txt")
        with open(pattern_file, "w") as f:
            for pattern in patterns:
                f.write(pattern + "\n")
        
        graph_options = config.get("processing_options", {}).get("graph_options", None)
        # Generate analysis graphs
        generate_all_graphs(cycle_one_instance, graph_options)
        
        # Make sure to release resources
        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

def process_with_visualization(video_path, start_frame, end_frame, crop_rect, options, output_dir):
    """Process a segment of the video with progress visualization"""
    # Simple command-line progress display instead of using pygame
    # This avoids display surface issues
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Unpack crop rectangle
    crop_x, crop_y, crop_w, crop_h = crop_rect
    
    # Ensure crop rectangle is within video bounds
    crop_x = max(0, min(width - 10, crop_x))
    crop_y = max(0, min(height - 10, crop_y))
    crop_w = max(10, min(width - crop_x, crop_w))
    crop_h = max(10, min(height - crop_y, crop_h))
    
    # Initialize variables for motion detection
    motion_history = []
    position_tracking = []
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    
    # Process frames
    try:
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        total_frames = end_frame - start_frame
        
        prev_gray = None
        
        # Process all frames in segment
        while frame_count < total_frames:
            # Update progress bar
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                progress_bar = "▓" * int(progress // 2) + "░" * (50 - int(progress // 2))
                print(f"\rProcessing: [{progress_bar}] {progress:.1f}% | Frame {frame_count}/{total_frames}", end="")
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop the frame
            crop_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
        # Complete the progress bar
        progress_bar = "▓" * 50
        print(f"\rProcessing: [{progress_bar}] 100.0% | Frame {frame_count}/{total_frames}")
        print(f"Processed segment {start_frame}-{end_frame}: 100% complete")
        
        
        print(f"Segment processing complete. Output saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing video segment: {e}")
        traceback.print_exc()
        return False    
        
def run_conducting_analysis(config):
    """Run the conducting analysis workflow"""
    if not CONDUCTING_MODULES_AVAILABLE:
        print("Conducting analysis modules not available. Skipping conducting analysis.")
        return False
        
    try:
        print("\n=== Starting Conducting Movement Analysis ===")
        # Make sure output directory exists
        export_path = config["export_path"]
        os.makedirs(export_path, exist_ok=True)
        
        # Set the export path in imports.py to ensure all graphs are saved to the correct location
        from imports import set_export_path
        set_export_path(export_path)
        
        print(f"Set export path to: {export_path}")
        
        # Create the content for names.py with explicit line breaks
        names_py_content = "from imports import *\n"
        names_py_content += "import os\n\n"
        names_py_content += f'VIDEO_PATH = "{config["video_path"]}"\n'
        names_py_content += f'EXPORT_PATH = "{export_path}"\n\n'
        
        # Add all the functions
        names_py_content += "# returns the name of the video file being processed\n"
        names_py_content += "def initialize_video():\n"
        names_py_content += "    # Return just the basename without extension\n"
        names_py_content += "    return os.path.splitext(os.path.basename(VIDEO_PATH))[0]\n\n"
        
        names_py_content += "# returns name for the final output video\n"
        names_py_content += "def video_out_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    outNames = videoFileName + \"_analyzed\"\n"
        names_py_content += "    return outNames\n\n"
        
        names_py_content += "# returns name for the main coordinates plot\n"
        names_py_content += "def video_plot_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    plotName = videoFileName + '_coordinates_plot'\n"
        names_py_content += "    return plotName\n\n"
        
        names_py_content += "# returns name for the sway analysis plot\n"
        names_py_content += "def video_sway_plot_Name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    swayPlotName = videoFileName + '_sway_plot'\n"
        names_py_content += "    return swayPlotName\n\n"
        
        names_py_content += "# returns name for the x-axis hand movement plot\n"
        names_py_content += "def video_hands_plot_x_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    handsPlotName_X = videoFileName + '_hands_plot_x'\n"
        names_py_content += "    return handsPlotName_X\n\n"
        
        names_py_content += "# returns name for the y-axis hand movement plot\n"
        names_py_content += "def video_hands_plot_y_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    handsPlotName_Y = videoFileName + '_hands_plot_y'\n"
        names_py_content += "    return handsPlotName_Y\n\n"
        
        names_py_content += "# returns name for the beat detection plot\n"
        names_py_content += "def video_beat_plot_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    beatPlotName = videoFileName + '_beat_plot'\n"
        names_py_content += "    return beatPlotName\n\n"
        
        names_py_content += "# returns name for the conducting path visualization\n"
        names_py_content += "def video_conduct_path_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    conductPath = videoFileName + '_conduct_path'\n"
        names_py_content += "    return conductPath\n\n"
        
        names_py_content += "# returns name for the bpm text file\n"
        names_py_content += "def video_bpm_output_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    bpmOutputName = videoFileName + '_auto_BPM.txt'\n"
        names_py_content += "    return bpmOutputName\n\n"
        
        names_py_content += "# returns name for the cluster plot\n"
        names_py_content += "def video_cluster_plot_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    clusterPlotName = videoFileName + '_cluster_plot'\n"
        names_py_content += "    return clusterPlotName\n\n"
        
        names_py_content += "# returns name for the overtime plot\n"
        names_py_content += "def video_overtime_plot_name():\n"
        names_py_content += "    videoFileName = initialize_video()\n"
        names_py_content += "    overtimePlotName = videoFileName + '_overtime_plot'\n"
        names_py_content += "    return overtimePlotName\n\n"
        
        # Add function to ensure paths are combined correctly
        names_py_content += "# Helper function to join export path with filename\n"
        names_py_content += "def get_full_path(filename):\n"
        names_py_content += "    return os.path.join(EXPORT_PATH, filename)\n"
        
        # Write the content to names.py
        with open("names.py", "w") as f:
            f.write(names_py_content)
            
        # Reload the names module to ensure it picks up changes
        import importlib
        import names
        importlib.reload(names)
            
        # Run the conducting analysis
        print(f"Analyzing video: {config['video_path']}")
        print(f"Saving results to: {export_path}")
        
        cycle_one_instance = CycleOne(config)
        cycle_two_instance = CycleTwo(cycle_one_instance, config)
        
        print("=== Conducting Analysis Complete ===\n")
        return True
    except Exception as e:
        print(f"Error in conducting analysis: {e}")
        traceback.print_exc()
        return False

def main():
    """Main application entry point"""
    print("=== Movement Detection and Conducting Analysis Application ===")
    
    # Run interface to get user input
    if run_interface():
        # Load configuration from interface
        config = load_config()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(config["video_path"]))[0]
        output_dir = os.path.join(config["export_path"], f"{video_name}_analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration copy
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Check if conducting analysis should run
        if config["processing_options"].get("conducting_analysis", False) and CONDUCTING_MODULES_AVAILABLE:
            print("\nRunning conducting analysis...")
            # Make a copy of config for conducting analysis
            conducting_config = config.copy()
            conducting_config["export_path"] = os.path.join(output_dir, "conducting_analysis")
            os.makedirs(conducting_config["export_path"], exist_ok=True)
            
            # Explicitly update the export path in imports.py
            try:
                from imports import set_export_path
                set_export_path(conducting_config["export_path"])
                print(f"Updated export path to: {conducting_config['export_path']}")
            except Exception as e:
                print(f"Warning: Could not update export path in imports.py: {e}")
            
            run_conducting_analysis(conducting_config)
        
        # Reset the path for movement detection
        try:
            from imports import set_export_path
            set_export_path(output_dir)
        except Exception as e:
            print(f"Warning: Could not reset export path for movement detection: {e}")
        
        # Run standard movement detection
        if config["processing_options"].get("detect_motion", True):
            print("\nBeginning movement detection processing...")
            
            # Process each segment
            for i, marker in enumerate(config["process_markers"]):
                print(f"\nProcessing segment {i+1} of {len(config['process_markers'])}...")
                
                # Extract the start_frame, end_frame, and crop values based on marker type
                if isinstance(marker, dict):
                    # New format with dictionary
                    start_frame = marker["start_frame"]
                    end_frame = marker["end_frame"]
                    crop = marker.get("crop", config["crop_rect"])
                elif isinstance(marker, (list, tuple)) and len(marker) >= 2:
                    # Old format with tuple/list [start_frame, end_frame]
                    start_frame = marker[0]
                    end_frame = marker[1]
                    # Use crop from marker if available, otherwise use default
                    crop = marker[2] if len(marker) > 2 else config["crop_rect"]
                else:
                    print(f"Warning: Invalid marker format: {marker}. Skipping.")
                    continue
                
                # Use the visualization-enabled processing function
                process_with_visualization(
                    config["video_path"],
                    start_frame,
                    end_frame,
                    crop,
                    config["processing_options"],
                    output_dir
                )
        
        print("\nAll processing complete!")
        print(f"Results saved to: {os.path.abspath(output_dir)}")
        
        # Open output folder if on Windows
        if os.name == 'nt':
            os.startfile(output_dir)
    else:
        print("Interface closed without configuration. Exiting.")       

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Keep console open on error if running as executable
        if getattr(sys, 'frozen', False):
            input("\nPress Enter to exit...")