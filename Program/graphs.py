from imports import *
import os
import json

# Get export path from configuration
def get_export_path():
    try:
        with open("interface_config.json", "r") as f:
            config = json.load(f)
            export_path = config.get("export_path", "output")
            # Ensure directory exists
            os.makedirs(export_path, exist_ok=True)
            return export_path
    except:
        # Fallback to default
        default_path = "output"
        os.makedirs(default_path, exist_ok=True)
        return default_path

# Helper function to get video name from configuration
def get_video_name():
    try:
        with open("interface_config.json", "r") as f:
            config = json.load(f)
            video_path = config.get("video_path", "unknown_video")
            return os.path.basename(video_path).split('.')[0]
    except:
        return "unknown_video"

# Video name generation functions
def video_beat_plot_name():
    return f"{get_video_name()}_beat_plot"

def video_conduct_path_name():
    return f"{get_video_name()}_conduct_path"

def video_cluster_plot_name():
    return f"{get_video_name()}_cluster_plot"

def video_overtime_plot_name():
    return f"{get_video_name()}_overtime_plot"

def video_sway_plot_Name():
    return f"{get_video_name()}_sway_plot"

def video_hands_plot_x_name():
    return f"{get_video_name()}_hands_x_plot"

def video_hands_plot_y_name():
    return f"{get_video_name()}_hands_y_plot"

def video_out_name():
    return f"{get_video_name()}_analyzed"

# generates all analysis graphs from the collected data
# Add this updated function to graphs.py

def generate_all_graphs(cycle_one, graph_options=None, segment_info=None):
    """Generate all analysis graphs from the collected data.
    
    Parameters:
    cycle_one (CycleOne): The cycle one instance with detection data
    graph_options (dict): Options for which graphs to generate
    segment_info (tuple): Optional tuple of (start_frame, end_frame) for segment-specific naming
    """
    # Default all options to True if none provided
    if graph_options is None:
        graph_options = {
            "generate_beat_plot": True,
            "generate_hand_path": True,
            "generate_cluster_graph": True,
            "generate_overtime_graph": True,
            "generate_swaying_graph": True,
            "generate_mirror_x_graph": True,
            "generate_mirror_y_graph": True
        }
    
    # Create segment suffix for filenames if segment_info is provided
    segment_suffix = ""
    if segment_info and len(segment_info) == 2:
        start_frame, end_frame = segment_info
        segment_suffix = f"_segment_{start_frame}_{end_frame}"
    
    print(f"\n=== Generating Analysis Graphs{' for segment' if segment_suffix else ''} ===")
    
    if graph_options.get("generate_beat_plot", True):
        print("Generating beat plot...")
        beat_plot_graph(cycle_one.processing_intervals, cycle_one.filtered_significant_beats, 
                       cycle_one.y_peaks, cycle_one.y_valleys, cycle_one.y, segment_suffix)
    
    if graph_options.get("generate_hand_path", True):
        print("Generating hand path graph...")
        hand_path_graph(cycle_one.x, cycle_one.y, segment_suffix)

    if graph_options.get("generate_cluster_graph", True):
        print("Generating cluster graph...")
        cluster_graph(cycle_one.beat_coordinates, segment_suffix)

    if graph_options.get("generate_overtime_graph", True):
        print("Generating overtime graph...")
        overtime_graph(cycle_one.y, segment_suffix)

    if graph_options.get("generate_swaying_graph", True):
        print("Generating swaying graph...")
        swaying_graph(cycle_one.swaying_detector.midpoints_x, 
                     cycle_one.swaying_detector.default_midpoint_history, 
                     cycle_one.swaying_detector.sway_threshold,
                     segment_suffix)
    
    if graph_options.get("generate_mirror_x_graph", True):
        print("Generating mirror X coordinate graph...")
        mirror_x_coordinate_graph(cycle_one.mirror_detector.left_hand_x, 
                                cycle_one.mirror_detector.right_hand_x,
                                segment_suffix)
    
    if graph_options.get("generate_mirror_y_graph", True):
        print("Generating mirror Y coordinate graph...")
        mirror_y_coordinate_graph(cycle_one.mirror_detector.left_hand_y, 
                                cycle_one.mirror_detector.right_hand_y,
                                segment_suffix)
    
    print("=== Graph Generation Complete ===\n")
# Modify these functions in graphs.py to correctly apply the segment_suffix

# Updated beat_plot_graph function for graphs.py

def beat_plot_graph(intervals, beats, y_peaks, y_valleys, y, segment_suffix=""):
    export_path = get_export_path()
    plt.figure(figsize=(12, 6))
    
    # plot coordinate data
    plt.plot(range(len(y)), y, label='Y Coordinates', color='g', alpha=0.7)

    # Handle different behavior for segment graphs vs full video graph
    if not segment_suffix:
        # For the full video graph, highlight all processing intervals
        if intervals:
            for start, end in intervals:
                # Make sure intervals are within the data range
                if start < len(y) and end < len(y):
                    plt.axvspan(start, end, color='yellow', alpha=0.3, 
                               label="Processed Range" if start == intervals[0][0] else None)
    else:
        # For segment graphs, highlight the entire range since it's all processed
        plt.axvspan(0, len(y)-1, color='yellow', alpha=0.2, label="Processed Range")

    # plot beat markers if any exist
    if beats:
        all_beats = [b for b in sorted(beats) if b < len(y)]
        if all_beats:
            for beat in all_beats:
                plt.axvline(x=beat, color='purple', linestyle='--', 
                           label="Beats" if beat == all_beats[0] else None)
    
    # Only plot peaks and valleys within the data range
    if y_peaks is not None and len(y_peaks) > 0:
        valid_peaks = [p for p in y_peaks if p < len(y)]
        if valid_peaks:
            plt.plot(valid_peaks, [y[i] for i in valid_peaks], "o", label="Y Peaks", markersize=6)
    
    if y_valleys is not None and len(y_valleys) > 0:
        valid_valleys = [v for v in y_valleys if v < len(y)]
        if valid_valleys:
            plt.plot(valid_valleys, [y[i] for i in valid_valleys], "o", label="Y Valleys", markersize=6)
    
    # Set plot title based on whether it's a segment or full video
    if segment_suffix:
        segment_parts = segment_suffix.split("_")
        if len(segment_parts) >= 3:
            segment_start = segment_parts[-2]
            segment_end = segment_parts[-1]
            title = f'Segment Analysis - Frames {segment_start}-{segment_end}'
        else:
            title = f'Segment Analysis{segment_suffix}'
            
        # For segment plots, adjust the x-axis to show frame numbers relative to segment
        plt.xlim(0, len(y)-1)
    else:
        title = 'X and Y Coordinates Over Frame Number - Full Video'
    
    # set plot attributes and save
    plt.title(title)
    plt.xlabel('Frame Number')
    plt.ylabel('Coordinate Value')
    plt.legend()
    plt.grid(True)
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_beat_plot_name() + segment_suffix + '.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=100)
    plt.close()

def hand_path_graph(x_proc, y_proc, segment_suffix=""):
    export_path = get_export_path()
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
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_conduct_path_name() + segment_suffix + '.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def cluster_graph(beat_coordinates, segment_suffix=""):
    export_path = get_export_path()  
    plt.figure(figsize=(12, 6))
    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")

    # Define colors for the beats
    colors = ['red', 'blue', 'green', 'orange']  # List of colors for beats

    # Plot the beats on the graph
    if beat_coordinates:  # Check if there are any beat coordinates
        x_beats, y_beats = zip(*beat_coordinates)  # Unzip the beat coordinates
        
        # Plot each beat with a color based on its index
        for i in range(len(x_beats)):
            # Calculate the color index based on the total number of colors
            color_index = i % len(colors)  # Cycle through colors
            plt.scatter(x_beats[i], y_beats[i], color=colors[color_index])  # Use the color for the current beat

    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")
    plt.title("Hand Cluster Plot")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_cluster_plot_name() + segment_suffix + '.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def overtime_graph(y, segment_suffix=""):
    export_path = get_export_path()
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

    # Mark peaks and valleys on the plot
    for valley in y_valleys:
        plt.plot(valley, -y_normalized[valley], 'o', color='purple', label="Downbeat" if valley == y_valleys[0] else None)
        plt.text(valley, -y_normalized[valley], 'Downbeat', color='purple', fontsize=8, ha='right')
    for peak in y_peaks:
        plt.plot(peak, -y_normalized[peak], 'o', color='blue', label="Peak" if peak == y_peaks[0] else None)
        plt.text(peak, -y_normalized[peak], 'Peak', color='blue', fontsize=8, ha='right')

    # Estimate time signature from peaks
    peak_heights = [-y_normalized[i] for i in y_peaks]
    
    if peak_heights:
        large_wave_threshold = np.percentile(peak_heights, 75)
        large_wave_indices = [i for i in y_peaks if -y_normalized[i] > large_wave_threshold]
        small_wave_counts = []

        for i in range(1, len(large_wave_indices)):
            small_wave_count = sum(1 for j in y_peaks if large_wave_indices[i-1] < j < large_wave_indices[i] and -y_normalized[j] <= large_wave_threshold)
            small_wave_counts.append(small_wave_count)

            time_signature = small_wave_count + 1
            print(f"Estimated Time Signature at frame {large_wave_indices[i]}: {time_signature}/4")

    else:
        print("No significant peaks detected to determine time signature.")

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

    # Save the plot with segment suffix if provided
    output_file = os.path.join(export_path, video_overtime_plot_name() + segment_suffix + '.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def swaying_graph(mid, default_mid, threshold, segment_suffix=""):
    export_path = get_export_path()
    if not mid:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot all midpoints
    plt.plot(range(len(mid)), mid, label='Current Midpoint X', color='b', alpha=0.7)
    
    # Ensure default_mid has consistent data types
    default_mid_normalized = []
    for value in default_mid:
        try:
            if isinstance(value, (list, tuple, np.ndarray)):
                # If it's a sequence, take the first element or average
                if len(value) > 0:
                    default_mid_normalized.append(float(value[0]))
                else:
                    default_mid_normalized.append(0.0)
            else:
                default_mid_normalized.append(float(value))
        except (ValueError, TypeError):
            # If conversion fails, use a default value
            default_mid_normalized.append(0.0)
    
    # Plot the default midpoints only if we have normalized values
    if default_mid_normalized:
        plt.plot(range(len(default_mid_normalized)), default_mid_normalized, 
                label='Default Midpoint X', color='r', alpha=0.7)
        
        # Plot threshold lines based on default_mid values
        upper_threshold = [value + threshold for value in default_mid_normalized]
        lower_threshold = [value - threshold for value in default_mid_normalized]
        
        plt.plot(range(len(default_mid_normalized)), upper_threshold, color='r', 
                linestyle='--', label='Upper Threshold X')
        plt.plot(range(len(default_mid_normalized)), lower_threshold, color='r', 
                linestyle='--', label='Lower Threshold X')
    
    # Set plot attributes and save
    plt.title('Swaying Detection Over Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Midpoint X Value')
    plt.legend()
    plt.grid(True)
    
    # Ensure export path exists and create full path for the file
    os.makedirs(export_path, exist_ok=True)
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_sway_plot_Name() + segment_suffix + '.png')
    
    print(f"Saving swaying graph to: {output_file}")
    plt.savefig(output_file)
    plt.close()

def mirror_x_coordinate_graph(left_hand_x, right_hand_x, segment_suffix=""):
    export_path = get_export_path()
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
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_hands_plot_x_name() + segment_suffix + '.png')
    plt.savefig(output_file)
    plt.close()

def mirror_y_coordinate_graph(left_hand_y, right_hand_y, segment_suffix=""):
    export_path = get_export_path()
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
    
    # Add segment suffix to filename if provided
    output_file = os.path.join(export_path, video_hands_plot_y_name() + segment_suffix + '.png')
    plt.savefig(output_file)
    plt.close()