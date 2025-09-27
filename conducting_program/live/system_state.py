# This will house the logic
# for the live start of the program, 
# as well have infomation on the body ouline
# so people know how far to stand from the camera. 

from enum import Enum
import threading
import time
import cv2

class State(Enum): # Set Enum values
    SETUP = "setup"
    COUNTDOWN = "countdown"
    PROCESSING = "processing"
    ENDING = "ending"

class SystemState:
    def __init__(self):
        self.current_state = SetupState()  # Start with setup state
    
    def get_current_state(self):
        return self.current_state
    
    def is_swaying(self):
        # Defer to processing state's sway flag if applicable
        try:
            return isinstance(self.current_state, ProcessingState) and self.current_state.is_swaying()
        except NameError:
            return False

    def get_sway_thresholds(self):
        # Defer to processing state's thresholds if applicable
        try:
            if isinstance(self.current_state, ProcessingState):
                return self.current_state.get_sway_thresholds()
        except NameError:
            pass
        return None, None

    def get_reference_midpoint(self):
        # Expose stable/reference midpoint when in processing
        try:
            if isinstance(self.current_state, ProcessingState):
                return self.current_state.get_reference_midpoint()
        except NameError:
            pass
        return None

    def is_mirroring(self):
        # Defer to processing state's sway flag if applicable
        try:
            return isinstance(self.current_state, ProcessingState) and self.current_state.is_mirroring()
        except NameError:
            return False

    def change_state(self, new_state, clock_manager=None, bpm=0):
        if new_state == State.COUNTDOWN.value:
            self.current_state = CountdownState()
        elif new_state == State.PROCESSING.value:
            self.current_state = ProcessingState(bpm, clock_manager)
            clock_manager.start_session_clock()  # Start session timing for processing
        elif new_state == State.ENDING.value:
            self.current_state = EndingState()
        print(f"State changed to: {new_state}")

class SetupState:
    def __init__(self):
        self.left_wrist_y15 = 0
        self.right_wrist_y16 = 0
        self.previous_y_left = None
        self.previous_y_right = None
        self.processing_active = False
        self.movement_start_time = None
        self.movement_hold_duration = 1.0  # Hold hands up for 1 second
        self.significant_movement_threshold = 0.1
        print("=== SETUP PHASE ===")
    
    def get_state_name(self):
        return State.SETUP.value
    
    def has_visual_beats(self):
        return False  # SetupState doesn't have visual beats
   
    def main(self, pose_landmarks, clock_manager): 
        return self.wait_for_start_movement(pose_landmarks, clock_manager)

    def wait_for_start_movement(self, pose_landmarks, clock_manager):
        (_, self.left_wrist_y15) = pose_landmarks.get_pose_landmark_15()
        (_, self.right_wrist_y16) = pose_landmarks.get_pose_landmark_16()

        # Check to make sure we have data for both wrists
        if self.left_wrist_y15 is None or self.right_wrist_y16 is None: 
            return State.SETUP.value

        if self.previous_y_left is None or self.previous_y_right is None: # We only need to do this once for the frist frame, maybe change this later
            self.previous_y_left = self.left_wrist_y15
            self.previous_y_right = self.right_wrist_y16
            return State.SETUP.value 

        # Check for significant upward movement for both left and right
        left_moved_up = self.left_wrist_y15 < self.previous_y_left - self.significant_movement_threshold
        right_moved_up = self.right_wrist_y16 < self.previous_y_right - self.significant_movement_threshold
        
        # Check for significant downward movement for both left and right
        left_dropped_down = self.left_wrist_y15 > self.previous_y_left + self.significant_movement_threshold
        right_dropped_down = self.right_wrist_y16 > self.previous_y_right + self.significant_movement_threshold

        # Determine if both hands are up
        both_hands_up = (left_moved_up and right_moved_up) and not (left_dropped_down or right_dropped_down)

        if both_hands_up and not self.processing_active:
            if self.movement_start_time is None:
                self.movement_start_time = clock_manager.get_current_timestamp()
            elif clock_manager.get_current_timestamp() - self.movement_start_time >= self.movement_hold_duration:
                print("Starting Countdown")
                self.processing_active = True
                return State.COUNTDOWN.value  # Use enum value
        elif left_dropped_down or right_dropped_down:
            self.movement_start_time = None  # Reset movement timer
        
        return State.SETUP.value  # Use enum value

class CountdownState:
    def __init__(self):
        self.countdown_value = 3
        self.countdown_start_time = None
        self.countdown_interval = 1.0  # 1 second between countdown numbers
        print("=== COUNTDOWN PHASE ===")
    
    def get_state_name(self):
        return State.COUNTDOWN.value
    
    def has_visual_beats(self):
        return False  # CountdownState doesn't have visual beats
    
    def main(self, pose_landmarks, clock_manager):
        return self.update_countdown(clock_manager)
    
    def update_countdown(self, clock_manager):
        if self.countdown_start_time is None:
            self.countdown_start_time = clock_manager.get_current_timestamp()
            print(f"{self.countdown_value}")
            return State.COUNTDOWN.value
        
        elapsed_time = clock_manager.get_current_timestamp() - self.countdown_start_time
        expected_countdown = 3 - int(elapsed_time)
        
        if expected_countdown != self.countdown_value and expected_countdown >= 0:
            self.countdown_value = expected_countdown
            print(f"{self.countdown_value}")
        
        if elapsed_time >= 3.0:  # After 3 seconds
            print("GO!") # Debug
            return State.PROCESSING.value  # Use enum value
        
        return State.COUNTDOWN.value  # Use enum value

from shared.sway import SwayDetection
from shared.mirror import MirrorDetection
from shared.sound_manager import SoundManager
from live.visual_manager import ConductingGuide

class ProcessingState:
    def __init__(self, bpm, clock_manager):
        self.reference_midpoint = None  # Stable reference midpoint updated periodically
        self.last_midpoint_checked = None  # Track when midpoint was last checked
        self.midpoint_stable_count = 0  # Count how many times midpoint has been stable
        self.live_midpoint = None
        self.sway = SwayDetection() # Create an instance of sway
        self.mirror = MirrorDetection() # Create an instance of mirror
        self.sound_manager = SoundManager() # Initialize sound manager
        self.visual_guide = ConductingGuide() # Initialize visual guide

        # Centralized beat timing
        self.bpm = int(bpm)
        self.clock_manager = clock_manager
        self.beat_interval = 60 / self.bpm  # Delay in seconds between beats
        self.last_beat_session_time = 0.0
        self.visual_beat_flag = False
        self.visual_beat_lock = threading.Lock()
        self.beat_start_time = 0.0
        self.beat_timing_thread = None # Initialize thread tracking

        print("=== PROCESSING PHASE ===")

    # Midpoint Functions -------------------------------------------

    def get_reference_midpoint(self):
        return self.reference_midpoint
    
    # Check to see if 3 seconds have passed
    def should_update_midpoint(self, current_time, interval_seconds=3.0):
        # Check if enough time has passed to update the midpoint
        if self.last_midpoint_checked is None:
            self.last_midpoint_checked = current_time
            return True  # First time, always update

        return (current_time - self.last_midpoint_checked) >= interval_seconds
    
    def update_current_midpoint(self, pose_landmarks):
        if pose_landmarks.left_shoulder_12 and pose_landmarks.right_shoulder_11:
            pose_landmarks.calculate_midpoint() 
            self.live_midpoint = pose_landmarks.get_midpoint()
    
    # Check to see if we need to update the midpoint
    def update_midpoint_check(self, pose_landmarks, clock_manager):
        current_time = clock_manager.get_current_timestamp() # Get current time
        
        # Check if it's time to update (every 3 seconds)
        if not self.should_update_midpoint(current_time, 3.0):
            return False  # Not time to check yet

        # 3 seconds have passed update refrence midpoint; require a valid live midpoint
        if self.live_midpoint is None:
            self.last_midpoint_checked = current_time
            return False

        # If this is the first time, set the reference midpoint
        if self.reference_midpoint is None:
            self.reference_midpoint = self.live_midpoint
            self.last_midpoint_checked = current_time
            print("Reference midpoint set") # DEBUG
            return True

        # Delegate the evaluation logic to a dedicated method
        updated = self.evaluate_reference_update(current_time)
        self.last_midpoint_checked = current_time
        return updated

    def evaluate_reference_update(self, current_time):
        # Compare live directly to reference
        midpoint_difference = abs(self.live_midpoint - self.reference_midpoint)

        # Micro-adjust when close to reference (smooth small drift)
        if midpoint_difference <= 0.02:
            self.reference_midpoint = self.live_midpoint
            self.midpoint_stable_count = 0
            print("Reference midpoint micro-adjusted") # DEBUG
            return True

        # Large movement: require stability across 2 checks (6s total) before updating
        if midpoint_difference > 0.05:
            self.midpoint_stable_count += 1
            if self.midpoint_stable_count >= 2:
                self.reference_midpoint = self.live_midpoint
                self.midpoint_stable_count = 0
                print("Reference midpoint updated (stable large move)") # DEBUG
                return True
        else:
            # Small-to-medium movement: do not update reference; reset stability counter
            self.midpoint_stable_count = 0
        return False

    
    # Swaying Functions --------------------------------------------

    def is_swaying(self):
        return self.sway.get_sway_flag()

        # Maybe remove later
    def get_sway_thresholds(self):
        return self.sway.get_threshold_left(), self.sway.get_threshold_right()

    # Mirror Functions --------------------------------------------

    def is_mirroring(self):
        return self.mirror.get_mirroring_flag()

    # Beat Timing Functions ----------------------------------------
    
    def _start_beat_timing_thread(self):
        """Start the centralized beat timing thread."""
        if self.beat_timing_thread is None or not self.beat_timing_thread.is_alive():
            self.beat_timing_thread = threading.Thread(target=self._beat_timing_loop, daemon=True)
            self.beat_timing_thread.start()
    
    def _beat_timing_loop(self):
        """Centralized beat timing loop that triggers both sound and visual."""
        while True:
            session_elapsed_time = self.clock_manager.get_session_elapsed_time()
            expected_next_beat_session_time = self.last_beat_session_time + self.beat_interval
            
            if session_elapsed_time >= expected_next_beat_session_time:
                # Use the expected beat time for consistent timing
                beat_time = expected_next_beat_session_time
                
                # Trigger sound beat
                self.sound_manager.play_metronome_beat()
                
                # Trigger visual beat with consistent timing
                with self.visual_beat_lock:
                    self.visual_beat_flag = True
                    self.beat_start_time = beat_time
                
                self.last_beat_session_time = expected_next_beat_session_time
                
                # Catch up if significantly behind
                while session_elapsed_time >= self.last_beat_session_time + self.beat_interval:
                    self.last_beat_session_time += self.beat_interval
            
            time.sleep(0.001)  # Small sleep to prevent 100% CPU usage
    
    # Visual Functions --------------------------------------------
    
    def should_show_beat(self):
        """Check if we should show the beat circle and update flag."""
        with self.visual_beat_lock:
            if self.visual_beat_flag:
                session_elapsed_time = self.clock_manager.get_session_elapsed_time()
                # Hide the circle after beat_duration
                if session_elapsed_time - self.beat_start_time >= self.visual_guide.beat_duration:
                    self.visual_beat_flag = False
                return True
            return False
    
    def draw_beat_circle(self, frame):
        """Draw the red circle overlay on the frame."""
        if self.should_show_beat():
            # Get frame dimensions
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            
            # Center of the frame
            center_x = frame_width // 2
            center_y = frame_height // 2
            
            # Draw red circle
            cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), -1)  # Filled red circle
            cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)  # White border
    
    def has_visual_beats(self):
        """Check if this state supports visual beat indicators."""
        return True  # Only ProcessingState has visual beats

    # State Management Functions ----------------------------------

    def get_state_name(self):
        return State.PROCESSING.value

    def main(self, pose_landmarks, clock_manager):
        """Main processing loop for the conducting analysis."""
        self._start_beat_timing_thread() # Start centralized beat timing thread

        # Initialize Threads
        midpoint_thread = threading.Thread(target=self.update_current_midpoint, args=(pose_landmarks,))
        update_midpoint = threading.Thread(target=self.update_midpoint_check, args=(pose_landmarks, clock_manager))
        sway_thread = threading.Thread(target=self.sway.main, args=(self.reference_midpoint, self.live_midpoint))
        mirror_thread = threading.Thread(target=self.mirror.main, args=(pose_landmarks, clock_manager, self.live_midpoint))

        # Execute threads
        midpoint_thread.start() # Update midpoint
        
        # Midpoint Reliant Threads
        midpoint_thread.join() # Join midpoint thread before midpoint reliant methods
        update_midpoint.start()
        sway_thread.start()
        mirror_thread.start() 

        # Join threads before returning
        update_midpoint.join()
        sway_thread.join()
        mirror_thread.join()

        print("All threads done next cycle")
        return State.PROCESSING.value  # Use enum value

class EndingState:
    def __init__(self):
        print("=== ENDING PHASE ===")
    
    def get_state_name(self):
        return State.ENDING.value
    
    def has_visual_beats(self):
        """Check if this state supports visual beat indicators."""
        return False  # EndingState doesn't have visual beats
    
    def main(self, pose_landmarks, clock_manager):
        # TODO: Add ending logic 
        return State.ENDING.value  # Use enum value 
   