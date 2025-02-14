import pygame
import cv2
import numpy as np
from imports import *

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
# Get screen size
WIDTH, HEIGHT = info.current_w, info.current_h
window_size = (WIDTH, HEIGHT)  
screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)  
pygame.display.set_caption("Pose Detection UI")

def get_window_size():
    # Returns the current window size
    return window_size

def get_screen():
    # Returns the Pygame screen object
    return screen

def display_frame(frame):
    # Converts an OpenCV frame to a Pygame surface and displays it
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     # Rotate for correct orientation if needed
    frame = np.rot90(frame)
    # Convert NumPy array to Pygame surface
    frame = pygame.surfarray.make_surface(frame)
    # Draw frame on the screen
    screen.blit(frame, (0, 0))

