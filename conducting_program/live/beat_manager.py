import threading
import time
from shared.sound_manager import SoundManager
from live.visual_manager import ConductingGuide

class BeatManager:
    """
    Independent beat manager that handles timing, sound, and visual indicators.
    Runs in dedicated thread and spawns daemon threads for sound/visual playback.
    """
    
    def __init__(self, settings):
        # Get configuration from settings
        self.settings = settings
        self.bpm = settings.get_beats_per_minute()
        self.time_signature = settings.get_time_signature()
        self.beats_per_measure = int(self.time_signature.split('/')[0])
        
        # Beat timing
        self.beat_interval = 60 / self.bpm
        self.is_running = False
        self.bpm_thread = None
        
        # Beat tracking
        self.current_beat = 0
        self.measure_count = 0
        
        # Visual state (checked by live_display per frame)
        self.show_visual = False
        self.visual_start_time = 0.0
        self.visual_duration = 0.1  # 100ms
        self.visual_lock = threading.Lock()
        
        # Components
        self.sound_manager = SoundManager()
        self.visual_guide = ConductingGuide()
        
        print(f"BeatManager initialized: {self.bpm} BPM, {self.time_signature}")
    
    def start(self):
        """Start the independent beat timing thread."""
        if not self.is_running:
            self.is_running = True
            self.bpm_thread = threading.Thread(target=self._beat_worker, daemon=True)
            self.bpm_thread.start()
            print("BeatManager started")
    
    def stop(self):
        """Stop the beat timing thread."""
        self.is_running = False
        if self.bpm_thread:
            self.bpm_thread.join(timeout=0.1)
        print("BeatManager stopped")

    def _beat_worker(self):
        """Main beat timing thread"""
        while self.is_running:
            self._trigger_beat(time.time())
        
            time.sleep(self.beat_interval) # Sleep for exactly one beat interval
        
    def _trigger_beat(self, beat_time):
        """Trigger beat: spawn daemon threads for sound and visual."""
        # Update beat tracking
        self.current_beat += 1
        if self.current_beat > self.beats_per_measure:
            self.current_beat = 1
            self.measure_count += 1
        
        print(f"Beat {self.current_beat}/{self.beats_per_measure} (Measure {self.measure_count})")
        
        # Only play sound and show visual after warmup measure (measure 0)
        if self.measure_count >= 1:
            # Spawn daemon thread for sound (non-blocking)
            threading.Thread(target=self.sound_manager.play_metronome_sound, daemon=True).start()

            # Set visual flag (checked by live_display per frame)
            with self.visual_lock:
                self.show_visual = True
                self.visual_start_time = beat_time
                
        elif self.measure_count == 0:
            # Warmup measure: initialize audio system silently
            threading.Thread(target=self._warmup_audio, daemon=True).start()
    
    def should_show_visual(self):
        """Check if visual should be shown (called by live_display each frame)."""
        with self.visual_lock:
            if self.show_visual:
                # Check if visual duration has elapsed
                if time.time() - self.visual_start_time >= self.visual_duration:
                    self.show_visual = False
                return True
            return False
    
    def get_current_beat(self):
        """Get the current beat number."""
        return self.current_beat
    
    def get_measure_count(self):
        """Get the current measure number."""
        return self.measure_count
    
    def _warmup_audio(self):
        """Warmup the audio system during the warmup measure."""
        # Pre-initialize the audio system by calling the sound manager's warmup
        try:
            self.sound_manager.warmup_audio_system()
            print("Audio system warming up...")
        except Exception as e:
            print(f"Audio warmup failed: {e}")
