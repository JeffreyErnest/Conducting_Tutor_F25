# This will house the logic
# for the live start of the program, 
# as well have infomation on the body ouline
# so people know how far to stand from the camera. 

import threading
import time
import cv2
from enum import Enum

# Import components
from shared.sway import SwayDetection
from shared.mirror import MirrorDetection
from shared.sound_manager import SoundManager
from live.visual_manager import ConductingGuide
from live.beat_manager import BeatManager

class State(Enum): # Set Enum values
    SETUP = "setup"
    COUNTDOWN = "countdown"
    PROCESSING = "processing"
    ENDING = "ending"

class SystemState:
    def __init__(self, settings):
        self.current_state = SetupState()  # Start with setup state
        self.settings = settings  # Store settings for lazy BeatManager initialization
        self.beat_manager = None  # Lazy initialization - only create when needed
    
    # -------------------- State Accessor --------------------
    
    def get_current_state(self):
        return self.current_state
    
    # -------------------- Delegated Getters --------------------
    
    def is_swaying(self):
        return self.current_state.is_swaying()

    def get_sway_thresholds(self):
        return self.current_state.get_sway_thresholds()

    def get_reference_midpoint(self):
        return self.current_state.get_reference_midpoint()

    def is_mirroring(self):
        return self.current_state.is_mirroring()
    
    # -------------------- Beat Manager --------------------
    
    def get_beat_manager(self):
        """Lazy initialization: create BeatManager only when first requested."""
        if self.beat_manager is None:
            print("Initializing BeatManager...")
            self.beat_manager = BeatManager(self.settings)
        return self.beat_manager

    # -------------------- State Transition --------------------
    
    def change_state(self, new_state, clock_manager):
        """Change to a new state and perform any necessary initialization."""
        if new_state == State.COUNTDOWN.value:
            self.current_state = CountdownState()
        elif new_state == State.PROCESSING.value:
            self.current_state = ProcessingState(clock_manager)
            clock_manager.start_session_clock()
        elif new_state == State.ENDING.value:
            self.current_state = EndingState()
        print(f"State changed to: {new_state}")

class SetupState:
    def __init__(self):
        # Values initialized on first frame
        self.left_wrist_y15 = None
        self.right_wrist_y16 = None
        self.previous_y_left = None
        self.previous_y_right = None
        
        # State tracking
        self.processing_active = False
        self.movement_tracking = False
        self.movement_start_time = None
        self.first_frame = True
        
        # Configuration
        self.movement_hold_duration = 1.0  # Hold hands up for 1 second
        self.significant_movement_threshold = 0.1
        
        print("=== SETUP PHASE ===")
    
    # -------------------- State Identifier --------------------
    
    def get_state_name(self):
        return State.SETUP.value
    
    def has_visual_beats(self):
        return False  # SetupState doesn't have visual beats
    
    # -------------------- Frame Processing --------------------
   
    def _initialize_first_frame(self, pose_landmarks):
        """Initialize values on the first frame only."""
        (_, self.left_wrist_y15) = pose_landmarks.get_pose_landmark_15()
        (_, self.right_wrist_y16) = pose_landmarks.get_pose_landmark_16()
        
        # Set initial previous positions if we have valid data
        self.previous_y_left = self.left_wrist_y15
        self.previous_y_right = self.right_wrist_y16

    def wait_for_start_movement(self, pose_landmarks, clock_manager):
        """Detect user's hands-up gesture to start countdown."""
        (_, self.left_wrist_y15) = pose_landmarks.get_pose_landmark_15()
        (_, self.right_wrist_y16) = pose_landmarks.get_pose_landmark_16()

        # Check to make sure we have data for both wrists
        if self.left_wrist_y15 is None or self.right_wrist_y16 is None: 
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
            if not self.movement_tracking:
                self.movement_start_time = clock_manager.get_current_timestamp()
                self.movement_tracking = True
            elif clock_manager.get_current_timestamp() - self.movement_start_time >= self.movement_hold_duration:
                print("Starting Countdown")
                self.processing_active = True
                return State.COUNTDOWN.value
        elif left_dropped_down or right_dropped_down:
            self.movement_tracking = False
            self.movement_start_time = None  # Reset movement timer
        
        return State.SETUP.value
    
    # -------------------- Main Entry Point --------------------
    
    def main(self, pose_landmarks, clock_manager):
        """Main setup loop - wait for user to start."""
        if self.first_frame:
            self._initialize_first_frame(pose_landmarks)
            self.first_frame = False
            return State.SETUP.value
        else:
            return self.wait_for_start_movement(pose_landmarks, clock_manager)

class CountdownState:
    def __init__(self):
        # Values initialized on first frame
        self.beats_per_measure = None
        
        # State tracking
        self.first_frame = True
        
        print("=== COUNTDOWN PHASE ===")
    
    # -------------------- State Identifier --------------------
    
    def get_state_name(self):
        return State.COUNTDOWN.value
    
    # -------------------- Frame Processing --------------------
    
    def _initialize_first_frame(self, beat_manager):
        """Initialize countdown on the first frame."""
        if beat_manager:
            beat_manager.start()
            self.beats_per_measure = beat_manager.beats_per_measure
            print(f"Countdown started: {self.beats_per_measure} beats")
    
    def _check_countdown_complete(self, beat_manager):
        """Check if one full measure has completed."""
        if beat_manager and beat_manager.get_measure_count() >= 1:
            print("GO! Transition to Processing")
            return State.PROCESSING.value
        return State.COUNTDOWN.value
    
    # -------------------- Main Entry Point --------------------
    
    def main(self, pose_landmarks, clock_manager, beat_manager=None):
        """Countdown: Start BeatManager and wait for one full measure."""
        if self.first_frame:
            self._initialize_first_frame(beat_manager)
            self.first_frame = False
            return State.COUNTDOWN.value
        else:
            return self._check_countdown_complete(beat_manager)

class ProcessingState:
    def __init__(self, clock_manager):
        # Values initialized on first frame
        self.reference_midpoint = None
        self.last_midpoint_checked = None
        self.live_midpoint = None
        
        # State tracking
        self.midpoint_stable_count = 0
        self.first_frame = True
        
        # Components
        self.sway = SwayDetection()
        self.mirror = MirrorDetection()
        self.clock_manager = clock_manager

        print("=== PROCESSING PHASE ===")

    # -------------------- State Identifier --------------------

    def get_state_name(self):
        return State.PROCESSING.value
    
    # -------------------- Getter Methods --------------------

    def get_reference_midpoint(self):
        return self.reference_midpoint
    
    def is_swaying(self):
        return self.sway.get_sway_flag()

    def get_sway_thresholds(self):
        return self.sway.get_threshold_left(), self.sway.get_threshold_right()

    def is_mirroring(self):
        return self.mirror.get_mirroring_flag()
    
    # -------------------- Midpoint Processing --------------------
    
    def update_current_midpoint(self, pose_landmarks):
        """Update the live midpoint from current pose data."""
        if pose_landmarks.left_shoulder_12 and pose_landmarks.right_shoulder_11:
            pose_landmarks.calculate_midpoint() 
            self.live_midpoint = pose_landmarks.get_midpoint()
    
    def should_update_midpoint(self, current_time, interval_seconds=3.0):
        """Check if enough time has passed to update the midpoint."""
        return (current_time - self.last_midpoint_checked) >= interval_seconds
    
    def update_midpoint_check(self, pose_landmarks, clock_manager):
        """Check and update reference midpoint if needed."""
        current_time = clock_manager.get_current_timestamp() # Get current time
        
        # Check if it's time to update (every 3 seconds)
        if not self.should_update_midpoint(current_time, 3.0):
            return False  # Not time to check yet

        # 3 seconds have passed update reference midpoint; require a valid live midpoint
        if self.live_midpoint is None:
            self.last_midpoint_checked = current_time
            return False

        # Delegate the evaluation logic to a dedicated method
        updated = self.evaluate_reference_update(current_time)
        self.last_midpoint_checked = current_time
        return updated

    def evaluate_reference_update(self, current_time):
        """Evaluate whether to update reference midpoint based on movement."""
        midpoint_difference = abs(self.live_midpoint - self.reference_midpoint)

        # Micro-adjust when close to reference (smooth small drift)
        if midpoint_difference <= 0.02:
            self.reference_midpoint = self.live_midpoint
            self.midpoint_stable_count = 0
            print("Reference midpoint micro-adjusted")
            return True

        # Large movement: require stability across 2 checks (6s total) before updating
        if midpoint_difference > 0.05:
            self.midpoint_stable_count += 1
            if self.midpoint_stable_count >= 2:
                self.reference_midpoint = self.live_midpoint
                self.midpoint_stable_count = 0
                print("Reference midpoint updated (stable large move)")
                return True
        else:
            # Small-to-medium movement: do not update reference; reset stability counter
            self.midpoint_stable_count = 0
        return False
    
    # -------------------- Frame Processing --------------------
    
    def _initialize_first_frame(self, pose_landmarks):
        """Initialize processing state on first frame."""
        self.update_current_midpoint(pose_landmarks)
        if self.live_midpoint is not None:
            self.reference_midpoint = self.live_midpoint
            self.last_midpoint_checked = self.clock_manager.get_current_timestamp()
            print("Reference midpoint initialized")
    
    def _process_frame(self, pose_landmarks):
        """Process a single frame of data."""
        self.update_current_midpoint(pose_landmarks)
        self.update_midpoint_check(pose_landmarks, self.clock_manager)
        self.sway.main(self.reference_midpoint, self.live_midpoint)
        self.mirror.main(pose_landmarks, self.clock_manager, self.live_midpoint)
        # TODO: Check for ending movement
        print("End of Frame Cycle")
    
    # -------------------- Main Entry Point --------------------

    def main(self, pose_landmarks, clock_manager):
        """Main data processing loop"""
        if self.first_frame:
            self._initialize_first_frame(pose_landmarks)
            self.first_frame = False
            return State.PROCESSING.value
        else:
            self._process_frame(pose_landmarks)
            return State.PROCESSING.value

class EndingState:
    def __init__(self):
        print("=== ENDING PHASE ===")
    
    # -------------------- State Identifier --------------------
    
    def get_state_name(self):
        return State.ENDING.value
    
    # -------------------- Main Entry Point --------------------
    
    def main(self, pose_landmarks, clock_manager):
        """Ending phase processing."""
        # TODO: Add ending logic 
        return State.ENDING.value 
   