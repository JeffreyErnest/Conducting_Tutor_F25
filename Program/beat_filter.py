from imports import *

# filters points based on minimum distance threshold
def filter_significant_points(points, threshold):
    if len(points) == 0:
        return []
        
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if points[i] - filtered_points[-1] > threshold:
            filtered_points.append(points[i])
            
    return filtered_points

# analyzes movement data to detect conducting beats
def filter_beats(frame_array, processed_frame_array):
    print("\n=== Beat Filter Debug Information ===")
    print(f"Input frame array length: {len(frame_array)}")
    print(f"Processed frame array length: {len(processed_frame_array)}")

    # extract y and x coordinates from frame arrays (if flipped)
    y = [coord[0] for coord in frame_array]  # Assuming y is first
    x = [coord[1] for coord in frame_array]  # Assuming x is second
    
    # convert to numpy arrays for processing
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    # find peaks and valleys in raw coordinates
    y_peaks, _ = find_peaks(y)
    y_valleys, _ = find_peaks(-y)  # negate y for valleys

    # process filtered coordinates
    y_proc = np.array([coord[1] for coord in processed_frame_array]).flatten()

    # find peaks and valleys in processed coordinates
    y_peaks_proc, _ = find_peaks(y_proc, prominence=0.005)
    y_valleys_proc, _ = find_peaks(-y_proc, prominence=0.005)

    # convert peak/valley indices to lists
    y_peaks_proc = list(y_peaks_proc)
    y_valleys_proc = list(y_valleys_proc)


    filtered_significant_beats = list(y_peaks)
    # Get the x,y coordinates for each beat
    beat_coordinates = [(x[i], y[i]) for i in filtered_significant_beats]

    # Before return, add debug info
    print(f"Number of y peaks: {len(y_peaks)}")
    print(f"Number of y valleys: {len(y_valleys)}")
    print(f"Number of filtered beats: {len(filtered_significant_beats)}")
    print("==================================\n")

    return filtered_significant_beats, beat_coordinates, y_peaks, y_valleys, y, x