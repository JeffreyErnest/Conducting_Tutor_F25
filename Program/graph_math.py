import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks

# Function to process hand path data
def process_hand_path_data(x_proc, y_proc):
    """
    Process hand path data for visualization.
    
    Parameters:
    -----------
    x_proc, y_proc : array-like
        X and Y coordinates of the conducting hand
        
    Returns:
    --------
    x_valid, y_valid : array
        Filtered and processed coordinates with NaN values removed
        and y-coordinates inverted for proper visualization
        
    Notes:
    ------
    We invert y-coordinates (multiply by -1) to match the traditional 
    conducting visualization where the gesture goes upward physically 
    for higher notes, but in computer coordinates, higher y values 
    are lower on the screen.
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(x_proc) | np.isnan(y_proc))
    x_valid = x_proc[valid_mask]
    
    # IMPORTANT: We invert the y-coordinates here for visualization
    # This is to match physical conducting space where up is positive,
    # but in image coordinates, up is negative (y increases downward)
    y_valid = -y_proc[valid_mask]
    
    return x_valid, y_valid

# Function to create color gradient segments
def create_color_gradient_segments(x_valid, y_valid):
    points = np.array([x_valid, y_valid]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['blue', 'red']
    n_bins = len(x_valid)
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    return segments, custom_cmap

# Function to normalize data and detect peaks and valleys
def normalize_and_detect_peaks(y):
    """
    Normalize the y-coordinate data and detect peaks and valleys.
    
    Parameters:
    -----------
    y : array-like
        Y-coordinate values
        
    Returns:
    --------
    y_normalized : array
        Normalized y-coordinates (0-1 range)
    y_peaks : array
        Indices of peaks (maxima) in the y-coordinates
    y_valleys : array
        Indices of valleys (minima) in the y-coordinates
        
    Notes:
    ------
    - For peaks: we use find_peaks on -y because the find_peaks function
      finds local maxima, and we want to find the highest points in y.
      By using -y, we effectively find the lowest points in -y, which
      correspond to the highest points in y.
    - For valleys: we use find_peaks on y to find the lowest points in y.
    """
    # Normalize data to 0-1 range
    y_min, y_max = min(y), max(y)
    y_normalized = [(val - y_min) / (y_max - y_min) for val in y]
    
    # Set detection parameters
    prominence = (max(y_normalized) - min(y_normalized)) * 0.1
    distance = 5
    
    # Find peaks (maxima) - these are the highest points in y
    # Using -y normalized finds the highest points in the original y
    y_peaks, _ = find_peaks(-np.array(y_normalized), prominence=prominence, distance=distance)
    
    # Find valleys (minima) - these are the lowest points in y
    # Using y normalized directly finds the lowest points in the original y
    y_valleys, _ = find_peaks(y_normalized, prominence=prominence, distance=distance)
    
    return y_normalized, y_peaks, y_valleys

# Function to estimate time signature from peaks
def estimate_time_signature(y_peaks, y_normalized):
    peak_heights = [-y_normalized[i] for i in y_peaks]
    if peak_heights:
        large_wave_threshold = np.percentile(peak_heights, 75)
        large_wave_indices = [i for i in y_peaks if -y_normalized[i] > large_wave_threshold]
        small_wave_counts = []
        time_signatures = []
        
        for i in range(1, len(large_wave_indices)):
            small_wave_count = sum(1 for j in y_peaks if large_wave_indices[i-1] < j < large_wave_indices[i] and -y_normalized[j] <= large_wave_threshold)
            small_wave_counts.append(small_wave_count)
            time_signature = small_wave_count + 1
            time_signatures.append(time_signature)
            print(f"Estimated Time Signature at frame {large_wave_indices[i]}: {time_signature}/4")
        
        # Return the most common time signature if we have any
        if time_signatures:
            from collections import Counter
            counter = Counter(time_signatures)
            most_common_ts = counter.most_common(1)[0][0]
            print(f"Most common time signature: {most_common_ts}/4")
            return most_common_ts
        return None
    else:
        print("No significant peaks detected to determine time signature.")
        return None

# Function to identify beat positions in a pattern
def identify_beat_positions(beat_frames, y, time_signature=4):
    """
    Classify beats by their position in a conducting pattern (1st, 2nd, 3rd, 4th beat)
    even if some beats are missed during detection.
    
    Parameters:
    -----------
    beat_frames : list
        Frame numbers where beats were detected
    y : list or array
        Y-coordinate values for the conducting hand
    time_signature : int, optional
        Time signature numerator (default is 4 for 4/4 time)
        
    Returns:
    --------
    list
        Beat positions (0-indexed, so 0=1st beat, 1=2nd beat, etc.)
    """
    if not beat_frames:
        return []
    
    # Get normalized data and peaks/valleys
    y_normalized, y_peaks, y_valleys = normalize_and_detect_peaks(y)
    
    # Try to get time signature from the data
    detected_ts = estimate_time_signature(y_peaks, y_normalized)
    if detected_ts is not None and detected_ts > 0:
        time_signature = detected_ts
    
    # Find the extremes in y values at beats
    beat_y_values = [y[frame] if frame < len(y) else 0 for frame in beat_frames]
    beat_positions = []
    
    # For measures where we can identify clear patterns
    # Find potential downbeats (typically lowest y value in the pattern)
    # Pattern recognition based on vertical position and movement
    
    # Calculate the amplitude of each beat
    amplitudes = []
    for i, frame in enumerate(beat_frames):
        # Find nearest peak or valley
        nearest_peak = min(y_peaks, key=lambda p: abs(p - frame)) if len(y_peaks) > 0 else None
        nearest_valley = min(y_valleys, key=lambda v: abs(v - frame)) if len(y_valleys) > 0 else None
        
        if nearest_peak is not None and nearest_valley is not None:
            # Get the closer one
            if abs(nearest_peak - frame) < abs(nearest_valley - frame):
                amplitude = abs(y_normalized[nearest_peak])
            else:
                amplitude = abs(y_normalized[nearest_valley])
        elif nearest_peak is not None:
            amplitude = abs(y_normalized[nearest_peak])
        elif nearest_valley is not None:
            amplitude = abs(y_normalized[nearest_valley])
        else:
            amplitude = 0
            
        amplitudes.append(amplitude)
    
    # If we have enough beats, try to identify patterns
    if len(amplitudes) >= time_signature:
        # Group into potential measures
        potential_measures = []
        for i in range(0, len(amplitudes), time_signature):
            if i + time_signature <= len(amplitudes):
                potential_measures.append(amplitudes[i:i+time_signature])
        
        # Find the pattern by analyzing typical conducting patterns
        # Downbeat (1) is usually strongest, followed by beat 3 in 4/4
        pattern_template = None
        if time_signature == 4:  # 4/4 time
            # Expected relative strengths: 1st is strongest, 3rd is second strongest
            pattern_template = [0, 0, 0, 0]  # Will be filled with position indices
        elif time_signature == 3:  # 3/4 time
            # Expected relative strengths: 1st is strongest
            pattern_template = [0, 0, 0]
        elif time_signature == 2:  # 2/4 time
            pattern_template = [0, 0]
            
        # Fallback if we can't determine pattern
        if pattern_template is None:
            # Simple sequential assignment
            beat_positions = [i % time_signature for i in range(len(beat_frames))]
        else:
            # Use the amplitude to identify beat positions
            for i, frame in enumerate(beat_frames):
                measure_position = i % time_signature
                beat_positions.append(measure_position)
    else:
        # If we don't have enough data, use simple sequential assignment
        beat_positions = [i % time_signature for i in range(len(beat_frames))]
    
    return beat_positions 