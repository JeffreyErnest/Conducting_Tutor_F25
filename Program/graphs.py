from imports import *

# generates all analysis graphs from the collected data
def generate_all_graphs(cycle_one):

    beat_plot_graph(cycle_one.processing_intervals, cycle_one.filtered_significant_beats, cycle_one.y_peaks, cycle_one.y_valleys, cycle_one.y)
    
    hand_path_graph(cycle_one.x, cycle_one.y)

    time_signature = overtime_graph(cycle_one.y)  # Calculate time signature, also calls graph to be displayed
    
    # Print the time signature used
    print(f"Time Signature Used: {time_signature}/4")

    # Pass time_signature to cluster_graph
    cluster_graph(cycle_one.beat_coordinates, time_signature)  # Use the calculated time signature

    swaying_graph(cycle_one.swaying_detector.midpoints_x, cycle_one.swaying_detector.default_midpoint_history, cycle_one.swaying_detector.sway_threshold)
    
    mirror_x_coordinate_graph(cycle_one.mirror_detector.left_hand_x, cycle_one.mirror_detector.right_hand_x)
        
    mirror_y_coordinate_graph(cycle_one.mirror_detector.left_hand_y, cycle_one.mirror_detector.right_hand_y)


# generates plot showing beat detection and coordinate data
def beat_plot_graph(intervals, beats, y_peaks, y_valleys, y):
    plt.figure(figsize=(12, 6))
    
    # plot coordinate data
    plt.plot(range(len(y)), y, label='Y Coordinates', color='g', alpha=0.7)

    # highlight processing intervals
    if intervals:
        for start, end in intervals:
            plt.axvspan(start, end, color='yellow', alpha=0.3, 
                       label="Processed Range" if start == intervals[0][0] else None)

    # plot beat markers and peaks/valleys
    all_beats = sorted(beats)
    for beat in all_beats:
        plt.axvline(x=beat, color='purple', linestyle='--', 
                   label="Beats" if beat == all_beats[0] else None)
    plt.plot(y_peaks, [y[i] for i in y_peaks], "o", label="Y Peaks")
    plt.plot(y_valleys, [y[i] for i in y_valleys], "o", label="Y Valleys")
    
    # set plot attributes and save
    plt.title('X and Y Coordinates Over Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Coordinate Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(video_beat_plot_name() + '.png')
    plt.show()

# generates visualization of conducting pattern with color gradient
def hand_path_graph(x_proc, y_proc):
    
    # prepare valid data points
    valid_mask = ~(np.isnan(x_proc) | np.isnan(y_proc))
    x_valid = x_proc[valid_mask]
    y_valid = y_proc[valid_mask]
    y_valid = -y_valid  
    
    # create color gradient segments
    points = np.array([x_valid, y_valid]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['blue', 'red']
    n_bins = len(x_valid)
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # plot path with gradient
    norm = plt.Normalize(0, len(x_valid))
    lc = LineCollection(segments, cmap=custom_cmap, norm=norm)
    lc.set_array(np.arange(len(x_valid)))
    plt.gca().add_collection(lc)
    
    # set plot attributes and save
    plt.xlim(np.nanmin(x_valid), np.nanmax(x_valid))
    plt.ylim(np.nanmin(y_valid), np.nanmax(y_valid))
    cbar = plt.colorbar(lc)
    cbar.set_label('Frame Number')
    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")
    plt.title("Conducting Pattern")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(video_conduct_path_name() + '.png', bbox_inches='tight')
    plt.show()

# generates the plot for showing the clusters of the beats
def cluster_graph(beat_coordinates, time_signature):
    num_clusters = time_signature
    coordinates = np.array(beat_coordinates)

    # Invert the y-coordinates
    coordinates[:, 1] = -coordinates[:, 1]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(coordinates)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Plot the clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    for i in range(num_clusters):
        cluster_points = coordinates[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

    # Set plot attributes
    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")
    plt.title("Hand Cluster Plot")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(video_cluster_plot_name() + '.png', bbox_inches='tight')
    plt.show()

    # Print the time signature used
    print(f"Time Signature Used: {time_signature}/4")


# generates the plot for the y over the whole video
def overtime_graph(y):
    plt.figure(figsize=(12, 6))

    # Plot inverted Y coordinates for visual consistency
    plt.plot(range(len(y)), [-value for value in y], label="Y-Coords", color='g', alpha=0.7)

    # Normalize the data
    y_min, y_max = min(y), max(y)
    y_normalized = [(val - y_min) / (y_max - y_min) for val in y]

    # Set dynamic parameters for peak detection
    prominence = (max(y_normalized) - min(y_normalized)) * 0.1
    distance = 5

    # Detect peaks and valleys
    y_peaks, _ = find_peaks(-np.array(y_normalized), prominence=prominence, distance=distance)
    y_valleys, _ = find_peaks(y_normalized, prominence=prominence, distance=distance)

    # Calculate time signature
    time_signature = estimate_time_signature(y_peaks, y_normalized)

    # Mark peaks and valleys on the plot
    for valley in y_valleys:
        plt.plot(valley, -y_normalized[valley], 'o', color='purple', label="Downbeat" if valley == y_valleys[0] else None)
        plt.text(valley, -y_normalized[valley], 'Downbeat', color='purple', fontsize=8, ha='right')
    for peak in y_peaks:
        plt.plot(peak, -y_normalized[peak], 'o', color='blue', label="Peak" if peak == y_peaks[0] else None)
        plt.text(peak, -y_normalized[peak], 'Peak', color='blue', fontsize=8, ha='right')

    # Print detected peaks for debugging
    print("Detected Peaks and Heights:")
    for i in y_peaks:
        print(f"Frame {i}: Height {y[i]}")

    # Finalize and show the plot
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.title("Overtime Graph")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the plot
    plt.savefig(video_overtime_plot_name() + '.png', bbox_inches='tight')
    plt.show()

    return time_signature

def estimate_time_signature(y_peaks, y_normalized):
    # Check if y_peaks is empty using .size
    if y_peaks.size == 0:
        return 4  # Default to 4 if no peaks detected

    # Example logic: Count the number of beats in a certain window
    large_wave_threshold = np.percentile([-y_normalized[i] for i in y_peaks], 75)
    large_wave_indices = [i for i in y_peaks if -y_normalized[i] > large_wave_threshold]
    small_wave_counts = []

    for i in range(1, len(large_wave_indices)):
        small_wave_count = sum(1 for j in y_peaks if large_wave_indices[i-1] < j < large_wave_indices[i] and -y_normalized[j] <= large_wave_threshold)
        small_wave_counts.append(small_wave_count)

    time_signature = small_wave_count + 1 if small_wave_counts else 4  # Default to 4 if no small waves detected
    print(f"Estimated Time Signature: {time_signature}/4")
    return time_signature


# generates plot showing swaying detection data
def swaying_graph(mid, default_mid, threshold):

    if not mid:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot all midpoints
    plt.plot(range(len(mid)), mid, label='Current Midpoint X', color='b', alpha=0.7)
    
    # Plot the default midpoints
    plt.plot(range(len(default_mid)), default_mid, label='Default Midpoint X', color='r', alpha=0.7)
    
    # Plot threshold lines based on default_mid values
    upper_threshold = [value + threshold for value in default_mid]  # Calculate upper threshold
    lower_threshold = [value - threshold for value in default_mid]  # Calculate lower threshold
    
    plt.plot(range(len(default_mid)), upper_threshold, color='r', linestyle='--', label='Upper Threshold X')  # Updated line
    plt.plot(range(len(default_mid)), lower_threshold, color='r', linestyle='--', label='Lower Threshold X')  # Updated line
    
    # Set plot attributes and save
    plt.title('Swaying Detection Over Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Midpoint X Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(video_sway_plot_Name() + '.png')
    plt.show()

# generates plot showing x-coordinate mirror movement
def mirror_x_coordinate_graph(left_hand_x, right_hand_x):
    if not left_hand_x or not right_hand_x:
        return
        
    plt.figure(figsize=(12, 6))
    
    # plot normalized hand coordinates
    plt.plot(range(len(left_hand_x)), [x - left_hand_x[0] for x in left_hand_x], 
             label='Left Hand X', color='b', alpha=0.7)
    plt.plot(range(len(right_hand_x)), [x - right_hand_x[0] for x in right_hand_x], 
             label='Right Hand X', color='g', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', label='Default Line')
    
    # set plot attributes and save
    plt.title('Hands X Coordinates Over Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Coordinate Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(video_hands_plot_x_name() + '.png')
    plt.show()

# generates plot showing y-coordinate mirror movement
def mirror_y_coordinate_graph(left_hand_y, right_hand_y):
    if not left_hand_y or not right_hand_y:
        return

    y_corrected_left = [-y for y in left_hand_y]
    y_corrected_right = [-y for y in right_hand_y]
        
    plt.figure(figsize=(12, 6))
    
    # plot normalized hand coordinates
    plt.plot(range(len(y_corrected_left)), [y - y_corrected_left[0] for y in y_corrected_left], 
             label='Left Hand Y', color='r', alpha=0.7)
    plt.plot(range(len(y_corrected_right)), [y - y_corrected_right[0] for y in y_corrected_right], 
             label='Right Hand Y', color='m', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', label='Default Line')
    
    # set plot attributes and save
    plt.title('Hands Y Coordinates Over Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Coordinate Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(video_hands_plot_y_name() + '.png')
    plt.show()