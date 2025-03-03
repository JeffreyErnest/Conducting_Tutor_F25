# Standard library imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks
from numpy import mean
from sklearn.cluster import KMeans  # Add this import for clustering


# Mediapipe imports
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Local imports
from cueing import *
from names import *
from pattern import *
from mp_declaration import mediaPipeDeclaration
from p_stage1 import process_video
from p_stage2 import output_process_video
from beat_filter import filter_beats
from sway import swayingDetection
from mirror import mirrorDetection
from graphs import generate_all_graphs, overtime_graph
