from imports import *

# generates all analysis graphs from the collected data
def generate_all_graphs(cycle_one):

    beat_plot_graph(cycle_one.processing_intervals, cycle_one.filtered_significant_beats, cycle_one.x_peaks, cycle_one.x_valleys, cycle_one.y_peaks, cycle_one.y_valleys, cycle_one.x, cycle_one.y)
    
    hand_path_graph(cycle_one.x, cycle_one.y)

    cluster_graph(cycle_one.beat_coordinates)

    overtime_graph(cycle_one.x, cycle_one.y)

    swaying_graph(cycle_one.swaying_detector.midpoints_x, cycle_one.swaying_detector.midpoint_all_history, cycle_one.swaying_detector.sway_threshold)
    
    mirror_x_coordinate_graph(cycle_one.mirror_detector.left_hand_x, cycle_one.mirror_detector.right_hand_x)
        
    mirror_y_coordinate_graph(cycle_one.mirror_detector.left_hand_y, cycle_one.mirror_detector.right_hand_y)


# generates plot showing beat detection and coordinate data
def beat_plot_graph(intervals, beats, x_peaks, x_valleys, y_peaks, y_valleys, x, y):
    plt.figure(figsize=(12, 6))
    
    # plot coordinate data
    plt.plot(range(len(x)), x, label='X Coordinates', color='b', alpha=0.7)
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
    plt.plot(x_peaks, [x[i] for i in x_peaks], "o", label="X Peaks")
    plt.plot(x_valleys, [x[i] for i in x_valleys], "o", label="X Valleys")
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
    plt.figure(figsize=(12, 6))
    
    # prepare valid data points
    valid_mask = ~(np.isnan(x_proc) | np.isnan(y_proc))
    x_valid = x_proc[valid_mask]
    # x_valid = -x_valid # turn on iff the x cord is inverted
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
def cluster_graph(beat_coordinates):  # Updated to accept beat_coordinates

    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")

    # Plot the beats on the graph
    if beat_coordinates:  # Check if there are any beat coordinates
        x_beats, y_beats = zip(*beat_coordinates)  # Unzip the beat coordinates
        plt.scatter(x_beats, [-y for y in y_beats], color='red', label='Beats')  # Inverted Y coordinates

    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")
    plt.title("Hand Cluster Plot")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # Add legend to show beats
    # plt.savefig(video_conduct_path_name() + '.png', bbox_inches='tight')
    plt.show()

# generates the plot for the x and y over the whole video
def overtime_graph(x, y):

    plt.figure(figsize=(12, 6))  # Add figure size for better visibility
    plt.plot(x, label="X-Coords", color='b', alpha=0.7)  # Plot X coordinates
    plt.plot(range(len(y)), [-value for value in y], label="Y-Coords", color='g', alpha=0.7)  # Inverted Y coordinates

    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.title("Overtime Graph")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # Add legend to show beats
    plt.show()


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