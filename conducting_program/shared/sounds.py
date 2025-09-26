import os
import threading
import time
from pydub import AudioSegment
from pydub.playback import play

# Simple sound file paths
SOUNDS_PATH = "assets/sounds/"

class Metronome():
    def __init__(self, bpm, clock_manager):
        self.bpm = int(bpm)
        self.clock_manager = clock_manager
        self.beat_interval = 60 / self.bpm  # Delay in seconds between beats
        self.last_beat_session_time = 0.0 # Tracks when the last beat should have occurred

        # Load the metronome sound once
        sound_file_path = os.path.abspath(os.path.join(SOUNDS_PATH, "metronome-85688.mp3"))
        if os.path.exists(sound_file_path):
            self.metronome_sound = AudioSegment.from_mp3(sound_file_path)
        else:
            print(f"Metronome sound not found: {sound_file_path}")
            self.metronome_sound = None

    def _play_sound_non_blocking(self, sound):
        """Helper to play sound in a separate thread."""
        if sound:
            play(sound)

    def play_metronome_continuous(self):
        """Play the metronome sound continuously, synchronized with session time."""
        if self.metronome_sound is None:
            print("Metronome sound not loaded, cannot play.")
            return

        while True:
            session_elapsed_time = self.clock_manager.get_session_elapsed_time()
            expected_next_beat_session_time = self.last_beat_session_time + self.beat_interval

            if session_elapsed_time >= expected_next_beat_session_time:
                # Play sound in a new thread to prevent blocking
                threading.Thread(target=self._play_sound_non_blocking, args=(self.metronome_sound,)).start()
                self.last_beat_session_time = expected_next_beat_session_time

                # Catch up if significantly behind (e.g., if a frame took too long)
                while session_elapsed_time >= self.last_beat_session_time + self.beat_interval:
                    self.last_beat_session_time += self.beat_interval
            
            time.sleep(0.001) # Small sleep to prevent 100% CPU usage

    def play_sound(self, filename):
        """Play any sound file by filename."""
        sound_path = os.path.abspath(os.path.join(SOUNDS_PATH, filename))
        if os.path.exists(sound_path):
            try:
                play(AudioSegment.from_file(sound_path))
            except Exception as e:
                print(f"Error playing sound with pydub: {e}")
        else:
            print(f"Sound file not found: {sound_path}")
