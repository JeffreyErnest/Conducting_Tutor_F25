from imports import *

# returns the name of the video file being processed
def initialize_video():
    videoFileName = 'videos/Marchingband(2).mp4'
    return videoFileName

# returns name for the final output video
def video_out_name():
    videoFileName = initialize_video()
    outNames = videoFileName + "_Fill_output"
    return outNames

# returns name for the main coordinates plot
def video_plot_name():
    videoFileName = initialize_video()
    plotName = videoFileName + '_Full_coordinates_plot'
    return plotName

# returns name for the sway analysis plot
def video_sway_plot_Name():
    videoFileName = initialize_video()
    swayPlotName = videoFileName + '_Full_Sway_plot'
    return swayPlotName

# returns name for the x-axis hand movement plot
def video_hands_plot_x_name():
    videoFileName = initialize_video()
    handsPlotName_X = videoFileName + '_Full_Hands_plot_X'  
    return handsPlotName_X

# returns name for the y-axis hand movement plot
def video_hands_plot_y_name():
    videoFileName = initialize_video()
    handsPlotName_Y = videoFileName + '_Full_Hands_plot_Y'
    return handsPlotName_Y

# returns name for the beat detection plot
def video_beat_plot_name():
    videoFileName = initialize_video()
    beatPlotName = videoFileName + '_Full_coordinates_plot'
    return beatPlotName

# returns name for the conducting path visualization
def video_conduct_path_name():
    videoFileName = initialize_video()
    conductPath = videoFileName + '_Full_conducting_path' 
    return conductPath

# returns name for the bpm text file
def video_bpm_output_name():
    videoFileName = initialize_video()
    bpmOutputName = videoFileName + '_auto_BPM.txt'
    return bpmOutputName

        