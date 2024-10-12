import cv2
import matplotlib.pyplot as plt
import pickle
import mediapipe as mp
from scipy.signal import find_peaks
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2

# mp_pose = mp.solutions.pose
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 2D array to store coordinates
frame_array = []
frame_rate = 24

# draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

# Create a PoseLandmarker object
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO
)

# Creates / initializing the poselandmarker
detector = PoseLandmarker.create_from_options(options)

# For video input:
cap = cv2.VideoCapture('4-4stacatto(3).mp4')

# Initialize flags and variables
processing_active = False
start_frame = None
end_frame = None
frame_number = 0

# Running to do calculations- the frist time
# x15 = []
# y15 = []


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ending processing.")
        break

    # Crop a region of interest (ROI) from the frame
    roi = image[450:900, 600:1500]
    roi_resized = cv2.resize(roi, (1280, 720))
    image = roi_resized

    frame_timestamp_ms = round(cap.get(cv2.CAP_PROP_POS_MSEC))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    annotated_image = draw_landmarks_on_image(image_rgb, detection_result)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Only process frames between start_frame and end_frame
    pose_landmarks_list = detection_result.pose_landmarks

    if processing_active and start_frame is not None and (end_frame is None or frame_number <= end_frame):
        if pose_landmarks_list:  # Ensure there are pose landmarks detected
            for landmarks in pose_landmarks_list:  # Loop through each set of detected landmarks
                if len(landmarks) > 15:  # Ensure the 15th landmark exists
                    # Capture x and y coordinates of the 15th landmark (index 14)
                    x15 = landmarks[15].x  # x coordinate of the 15th landmark
                    y15 = landmarks[15].y  # y coordinate of the 15th landmark
                    frame_array.append((x15, y15))  # Store the coordinates as a tuple
                    

    # After the loop, prepare the x and y lists from frame_array
    x = [coord[0] for coord in frame_array]  # x coordinates of 15th landmark
    y = [coord[1] for coord in frame_array]  # y coordinates of 15th landmark
    print(x, y)


    # Display the current frame number
    cv2.putText(image, f'Frame: {frame_number}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow('Video Feed', annotated_image_bgr)

    # Check for user input to start/stop processing
    key = cv2.waitKey(5) & 0xFF
    if key == ord('s'):  # Press 's' to mark the start frame
        if not processing_active:
            start_frame = frame_number
            processing_active = True
            print(f"Processing started at frame {start_frame}")
    elif key == ord('e'):  # Press 'e' to mark the end frame
        if processing_active:
            end_frame = frame_number
            processing_active = False
            print(f"Processing stopped at frame {end_frame}")
    elif key == 27:  # Press 'ESC' to exit
        break

    frame_number += 1

cap.release()

# x = []
# y = []

# for frame_landmarks in frame_array:
#     if frame_landmarks:
#         x.append(frame_landmarks[16].x)
#         y.append(frame_landmarks[16].y)

# Now that you have start_frame and end_frame, apply beat detection only within that range
# Extract high and low peaks for x and y coordinates

# Find peaks and valleys in x and y coordinates
x_peaks, _ = find_peaks(x, prominence=0.0001)
x_valleys, _ = find_peaks([-val for val in x], prominence=0.0001)
# Find peaks and valleys in y coordinates
y_peaks, _ = find_peaks(y, prominence=0.01)
y_valleys, _ = find_peaks([-val for val in y], prominence=0.01)

# Combine both x and y peaks and valleys
combined_peaks = sorted(set(x_peaks).union(set(y_peaks)))  # Combine x and y peaks
combined_valleys = sorted(set(x_valleys).union(set(y_valleys)))  # Combine x and y valleys

# Convert combined_peaks and combined_valleys to standard Python integers for safe processing
combined_peaks = [int(x) for x in combined_peaks]
combined_valleys = [int(x) for x in combined_valleys]

filtered_significant_beats = [combined_peaks[0]]  # Start with the first peak
threshold = 10  # Define a threshold in number of frames

# threshold = 10  # Define a threshold in number of frames

# # Filter beats within the selected frame range
# if combined_peaks:  # Ensure it's not empty before accessing
#     filtered_significant_beats = [combined_peaks[0]]  # Start with the first peak
#     for i in range(1, len(combined_peaks)):
#         if combined_peaks[i] - filtered_significant_beats[-1] > threshold:
#             filtered_significant_beats.append(combined_peaks[i])
# else:
#     filtered_significant_beats = []

# # Similarly, process valleys if needed
# if combined_valleys:  # Ensure it's not empty before accessing
#     filtered_significant_valleys = [combined_valleys[0]]
#     for i in range(1, len(combined_valleys)):
#         if combined_valleys[i] - filtered_significant_valleys[-1] > threshold:
#             filtered_significant_valleys.append(combined_valleys[i])
#             filtered_significant_beats.append(combined_valleys[i])  # Ensure all are ints
# else:
#     filtered_significant_valleys = []
    
# Filter beats within the selected frame range
# filtered_significant_beats = [combined_peaks[0]]  # Start with the first peak

for i in range(1, len(combined_peaks)):
    # Check if the filtered_significant_beats is not empty before accessing its last element
    if len(filtered_significant_beats) == 0 or combined_peaks[i] - filtered_significant_beats[-1] > threshold:
        filtered_significant_beats.append(combined_peaks[i])

# Similarly, process valleys if needed
filtered_significant_valleys = [combined_valleys[0]]
for i in range(1, len(combined_valleys)):
    if combined_valleys[i] - filtered_significant_valleys[-1] > threshold:
        filtered_significant_valleys.append(combined_valleys[i])
        filtered_significant_beats.append(filtered_significant_valleys)

# OpenCV window name for displaying annotated frames
window_name = 'Annotated Frames'

frame_index = 0
fps = 24
frame_skip = 1  # Process every frame

# For video input
video = cv2.VideoCapture('4-4stacatto(1).mp4')
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

out = cv2.VideoWriter('4-4stacatto(1).mp4.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2  # Adjusted for better visibility
font_thickness = 2
font_color = (255, 255, 255)

text_display_duration = 3  # Adjust this value as needed
text_display_counter = 0

# Initialize variables for beat detection
prev_beat = None
prev_bpm = 0
bpm = None  # Initialize bpm to avoid the 'name not defined' error


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ending processing.")
        break

    # Crop a region of interest (ROI) from the frame
    roi = image[450:900, 600:1500]
    roi_resized = cv2.resize(roi, (1280, 720))
    image = roi_resized

    frame_timestamp_ms = round(cap.get(cv2.CAP_PROP_POS_MSEC))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    annotated_image = draw_landmarks_on_image(image_rgb, detection_result)
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Only apply beat detection between start_frame and end_frame
    if start_frame <= frame_index <= end_frame and frame_index in filtered_significant_beats:
        text = "Beat!"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (image_rgb.shape[1] - text_size[0]) // 2
        text_y = (image_rgb.shape[0] + text_size[1]) // 2
        cv2.putText(image_rgb, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        if prev_beat is not None:
            frames_between_beats = frame_index - prev_beat
            time_between_beats = frames_between_beats / frame_rate
            bpm = 60 / time_between_beats
        else:
            bpm = prev_bpm

        bpm_info = f'Beats per minute (BPM) at frame {frame_index}: {bpm}\n'
        output_file = '4-4stacatto(1).mp4_auto_BPM.txt'
        with open(output_file, 'a') as file:
            print(bpm_info, end='')
            file.write(bpm_info)

        text_display_counter = text_display_duration

    if text_display_counter > 0:
        cv2.putText(image_rgb, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        text_display_counter -= 1

    # Write the frame with the overlayed landmarks and annotations
    out.write(image_rgb)
    cv2.imshow('Annotated Frames', image_rgb)  # Show the full frame with landmarks

    # Handle user input for starting/stopping processing
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press ESC to exit
        break

    frame_index += 1

# Release the video capture and writer resources
video.release()
out.release()

# Plot x and y coordinates over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(x)), x, label='X Coordinates', color='b', alpha=0.7)
plt.plot(range(len(y)), y, label='Y Coordinates', color='g', alpha=0.7)
plt.title('X and Y Coordinates Over Frame Number')
plt.xlabel('Frame Number')
plt.ylabel('Coordinate Value')
plt.legend()
plt.grid()
plt.savefig('coordinates_plot.png')
plt.show()

cv2.destroyAllWindows()
