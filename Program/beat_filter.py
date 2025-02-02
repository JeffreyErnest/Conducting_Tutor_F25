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

proximity_threshold = 5
def filter_significant_points(x_peaks_proc, x_valleys_proc, y_peaks_proc, y_valleys_proc, threshold, proximity_threshold):
    # Combine all points but keep track of which are Y points
    all_points = []
    for frame in x_peaks_proc:
        all_points.append((frame, False))  # False indicates not a Y point
    for frame in x_valleys_proc:
        all_points.append((frame, False))
    for frame in y_peaks_proc:
        all_points.append((frame, True))   # True indicates Y point
    for frame in y_valleys_proc:
        all_points.append((frame, True))
    
    # Sort by frame number
    all_points.sort(key=lambda x: x[0])
    
    filtered_points = []
    last_added_frame = -threshold
    
    i = 0
    while i < len(all_points):
        current_frame = all_points[i][0]
        
        # Skip if too close to last added beat
        if current_frame - last_added_frame < threshold:
            i += 1
            continue
        
        # Look ahead for nearby points
        nearby_points = []
        j = i
        while j < len(all_points) and all_points[j][0] - current_frame <= proximity_threshold:
            nearby_points.append(all_points[j])
            j += 1
        
        # Check if any nearby points are Y points
        y_points = [p for p in nearby_points if p[1]]  # p[1] is True for Y points
        
        if y_points:
            # Use the first Y point in the group
            filtered_points.append(y_points[0][0])
            last_added_frame = y_points[0][0]
        else:
            # No Y points nearby, use the first point in the group
            filtered_points.append(nearby_points[0][0])
            last_added_frame = nearby_points[0][0]
        
        i = j  # Skip past all nearby points
            
    return filtered_points

# analyzes movement data to detect conducting beats
def filter_beats(frame_array, processed_frame_array):
    print("\n=== Beat Filter Debug Information ===")
    print(f"Input frame array length: {len(frame_array)}")
    print(f"Processed frame array length: {len(processed_frame_array)}")

    # extract x and y coordinates from frame arrays
    x = [coord[0] for coord in frame_array]
    y = [coord[1] for coord in frame_array]
    
    # convert to numpy arrays for processing
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    # find peaks and valleys in raw coordinates
    x_peaks, _ = find_peaks(x)
    x_valleys, _ = find_peaks(-x)  # negate x for valleys
    y_peaks, _ = find_peaks(y)
    y_valleys, _ = find_peaks(-y)  # negate y for valleys

    # process filtered coordinates
    x_proc = np.array([coord[0] for coord in processed_frame_array]).flatten()
    y_proc = np.array([coord[1] for coord in processed_frame_array]).flatten()

    # find peaks and valleys in processed coordinates
    x_peaks_proc, _ = find_peaks(x_proc)
    x_valleys_proc, _ = find_peaks(-x_proc)
    y_peaks_proc, _ = find_peaks(y_proc)
    y_valleys_proc, _ = find_peaks(-y_proc)

    # convert peak/valley indices to lists
    x_peaks_proc = list(x_peaks_proc)
    x_valleys_proc = list(x_valleys_proc)
    y_peaks_proc = list(y_peaks_proc)
    y_valleys_proc = list(y_valleys_proc)

    # combine all detected beats and filter by threshold
    threshold = 10  # minimum frames between beats can be adjusted, but 5 works well
    significant_beats = sorted(set(x_peaks_proc + x_valleys_proc + y_peaks_proc + y_valleys_proc))
    filtered_significant_beats = filter_significant_points(significant_beats, threshold)

    # Get the x,y coordinates for each beat
    beat_coordinates = [(x[i], y[i]) for i in filtered_significant_beats]

    # Before return, add debug info
    print(f"Number of x peaks: {len(x_peaks)}")
    print(f"Number of x valleys: {len(x_valleys)}")
    print(f"Number of y peaks: {len(y_peaks)}")
    print(f"Number of y valleys: {len(y_valleys)}")
    print(f"Number of filtered beats: {len(filtered_significant_beats)}")
    print("==================================\n")

    return filtered_significant_beats, beat_coordinates, x_peaks, x_valleys, y_peaks, y_valleys, x, y