# This will house the logic for the 
# movment that will end the program
# as well call the video_manager to 
# save the video if requested.


# OLD CODE

#  def end_processing(self, left_hand_x, right_hand_x, left_hand_y, right_hand_y ):

#         # Invert the y-axis values
#         left_hand_y = -left_hand_y
#         right_hand_y = -right_hand_y

#         left_significant_movement = (abs(left_hand_y - self.previous_y_left) > self.slight_movement_threshold or
#                                      abs(left_hand_x - self.previous_x_left) > self.slight_movement_threshold)
#         right_significant_movement = (abs(right_hand_y - self.previous_y_right) > self.slight_movement_threshold or
#                                       abs(right_hand_x - self.previous_x_right) > self.slight_movement_threshold)
        
#         no_movement_check = not (left_significant_movement or right_significant_movement)  # Check for no movement

#         # print (f" Left: {left_significant_movement} right: {right_significant_movement}")

#         if no_movement_check:
#             self.movement_counter += 1  # Increment counter if no movement detected
#         else:
#             self.movement_counter = 0  # Reset counter if movement is detected
            
#         # print (f"no movement check: {no_movement_check} counter is at {self.movement_counter}")

#         # Check if there is no movement for 2 seconds (60 frames)
#         no_movement_for_2_seconds = self.movement_counter >= 60

#         # Check if hands have crossed
#         hands_crossed = left_hand_x > right_hand_x
        
#         # Use the flags in the condition
#         if self.processing_active and (hands_crossed or no_movement_for_2_seconds):
#             print("Ended processing.")
#             self.processing_active = False

#         self.previous_y_left = left_hand_y
#         self.previous_x_left = left_hand_x
#         self.previous_y_right = right_hand_y
#         self.previous_x_right = right_hand_x
        
#         return self.processing_active