# Imports

import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings 
class Gest(IntEnum):
    # Binary Encoded
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    
    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    
    # Initialize instance variables
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    # Update the hand_result instance variable
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    # Calculate the signed distance between two landmarks given their indices
    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist*sign
    
    # Calculate the distance between two landmarks given their indices
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x)**2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y)**2
        dist = math.sqrt(dist)
        return dist
    
    # Calculate the absolute difference in z-coordinates between two landmarks given their indices
    def get_dz(self,point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    # Function to find Gesture Encoding using current finger_state.
    # Finger_state: 1 if finger is open, else 0
    def set_finger_state(self):

        # Check if hand landmarks have been detected
        if self.hand_result == None:
            return

        # Define points for each finger on the hand
        points = [[8,5,0],[12,9,0],[16,13,0],[20,17,0]]

        # Initialize finger state to 0
        self.finger = 0

        # Set the thumb state to open (1)
        self.finger = self.finger | 0

        # Loop through each finger on the hand
        for idx,point in enumerate(points):
            # Calculate the signed distance between two points on each finger
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            
            try:
                # Calculate the ratio of the two distances
                ratio = round(dist/dist2,1)
            except:
                # Handle division by zero error
                ratio = round(dist/0.01,1)

            # Shift the finger state one bit to the left and set the finger state to open (1) if the ratio is greater than 0.5
            self.finger = self.finger << 1
            if ratio > 0.5 :
                self.finger = self.finger | 1
    

    # Handling Fluctuations due to noise
    def get_gesture(self):
        # Check if hand landmarks have been detected
        if self.hand_result == None:
            return Gest.PALM

        # Initialize the current gesture to palm
        current_gesture = Gest.PALM

        # Check if the hand is making a pinch gesture
        if self.finger in [Gest.LAST3,Gest.LAST4] and self.get_dist([8,4]) < 0.05:

            if self.hand_label == HLabel.MINOR :
                # Set the current gesture to minor pinch if the hand is the left hand
                current_gesture = Gest.PINCH_MINOR
            else:
                # Set the current gesture to major pinch if the hand is the right hand
                current_gesture = Gest.PINCH_MAJOR

        # Check if the hand is making a V-gesture or two-finger closed gesture
        elif Gest.FIRST2 == self.finger :
            point = [[8,12],[5,9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2

            # Set the current gesture to V-gesture if the ratio of the distances is greater than 1.7
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                # Set the current gesture to two-finger closed if the distance between the fingers in the z-axis is less than 0.1
                if self.get_dz([8,12]) < 0.1:
                    current_gesture =  Gest.TWO_FINGER_CLOSED
                else:
                    # Set the current gesture to mid if none of the above conditions are met
                    current_gesture =  Gest.MID
            
        else:
            # Set the current gesture to the finger state if none of the above conditions are met
            current_gesture =  self.finger
        
        # If the current gesture is the same as the previous gesture, increment the frame count
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4 :
            self.ori_gesture = current_gesture
        return self.ori_gesture

# Executes commands according to detected gestures
class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3
    
    # This function calculates the vertical pinch distance between the index finger and thumb.
    # It takes the hand result object as input and returns the vertical pinch distance.
    def getpinchylv(hand_result):
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y)*10,1)
        return dist

    # This function calculates the horizontal pinch distance between the index finger and thumb.
    # It takes the hand result object as input and returns the horizontal pinch distance.
    def getpinchxlv(hand_result):
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord)*10,1)
        return dist
    
    # This function changes the brightness of the system based on the pinch distance calculated.
    # It first gets the current brightness level, adds the pinch level divided by 50 to it, and then sets the new brightness level.
    def changesystembrightness():
        currentBrightnessLv = sbcontrol.get_brightness()[0]/100.0
        currentBrightnessLv += Controller.pinchlv/50.0
        if currentBrightnessLv > 1.0:
            currentBrightnessLv = 1.0
        elif currentBrightnessLv < 0.0:
            currentBrightnessLv = 0.0       
        sbcontrol.fade_brightness(int(100*currentBrightnessLv) , start = sbcontrol.get_brightness()[0])
    
    # This function changes the volume of the system based on the pinch distance calculated.
    # It first gets the current volume level, adds the pinch level divided by 50 to it, and then sets the new volume level.
    def changesystemvolume():
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        currentVolumeLv = volume.GetMasterVolumeLevelScalar()
        currentVolumeLv += Controller.pinchlv/50.0
        if currentVolumeLv > 1.0:
            currentVolumeLv = 1.0
        elif currentVolumeLv < 0.0:
            currentVolumeLv = 0.0
        volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)
    
    # This function scrolls the content vertically based on the pinch distance calculated.
    # It scrolls up if pinch distance is positive and down if it's negative.
    def scrollVertical():
        pyautogui.scroll(120 if Controller.pinchlv>0.0 else -120)
        
    # This function scrolls the content horizontally based on the pinch distance calculated.
    # It scrolls left if pinch distance is positive and right if it's negative.
    # It also holds down the shift and ctrl keys while scrolling horizontally.
    def scrollHorizontal():
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv>0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    # Locate Hand to get Cursor Position
    # Stabilize cursor by Dampening
    def get_position(hand_result):
        # The landmark point index used to determine the cursor position
        point = 9
        # Get the x and y positions of the landmark point and scale it to the screen size   
        position = [hand_result.landmark[point].x ,hand_result.landmark[point].y]
        sx,sy = pyautogui.size()
        x_old,y_old = pyautogui.position()
        x = int(position[0]*sx)
        y = int(position[1]*sy)

        # Initialize variables used for dampening the cursor movement
        if Controller.prev_hand is None:
            Controller.prev_hand = x,y
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]

        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = [x,y]

        # Calculate the dampening ratio based on the distance moved by the cursor
        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1/2))
        else:
            ratio = 2.1
        x , y = x_old + delta_x*ratio , y_old + delta_y*ratio
        return (x,y)

    # Initialize variables for pinch gesture detection
    def pinch_control_init(hand_result):
        # Set the starting coordinates of the pinch gesture
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y

        # Set the pinch level and the previous pinch level to zero
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        
        # Set the frame count to zero
        Controller.framecount = 0

    # Hold final position for 5 frames to change status
    def pinch_control(hand_result, controlHorizontal, controlVertical):
        # If the frame count reaches 5, change the status
        if Controller.framecount == 5:
            # Reset the frame count and set the pinch level to the previous level
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            # If the pinch direction flag is true, scroll horizontally
            if Controller.pinchdirectionflag == True:
                controlHorizontal() #x

            # If the pinch direction flag is false, scroll vertically
            elif Controller.pinchdirectionflag == False:
                controlVertical() #y

        # Get the pinch levels in the x and y directions
        lvx =  Controller.getpinchxlv(hand_result)
        lvy =  Controller.getpinchylv(hand_result)

        # If the pinch is in the vertical direction and above the threshold
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            # Set the pinch direction flag to false for vertical
            Controller.pinchdirectionflag = False
            # If the previous pinch level is close to the current level, increase the frame count
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            
            # Otherwise, update the previous pinch level and reset the frame count
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0

        # If the pinch is in the horizontal direction and above the threshold
        elif abs(lvx) > Controller.pinch_threshold:
            # Set the pinch direction flag to true for horizontal
            Controller.pinchdirectionflag = True
            # If the previous pinch level is close to the current level, increase the frame count
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1

            # Otherwise, update the previous pinch level and reset the frame count
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    # Handle different types of gestures
    def handle_controls(gesture, hand_result):  
        # Initialize x and y coordinates as None
        x,y = None,None

        # If the gesture is not a palm gesture, get the position of the hand
        if gesture != Gest.PALM :
            x,y = Controller.get_position(hand_result)
        
        # Reset the grab flag if the gesture is not a fist
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            # Release the left mouse button
            pyautogui.mouseUp(button = "left")

        # Reset the pinch major flag if the gesture is not a pinch major
        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        # Reset the pinch minor flag if the gesture is not a pinch minor
        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # For each detected gesture, the appropriate action is performed using pyautogui.
        # When V_GEST is detected, set the flag to true and move the cursor to the current x, y position with a 0.1 sec duration.
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration = 0.1)

        # When FIST is detected, check if the grabflag is not set to true, set it to true, and perform a left mouse click.
        # Then move the cursor to the current x, y position with a 0.1 sec duration.
        elif gesture == Gest.FIST:
            if not Controller.grabflag : 
                Controller.grabflag = True
                pyautogui.mouseDown(button = "left")
            pyautogui.moveTo(x, y, duration = 0.1)

        # When MID is detected and the flag is set to true, perform a left mouse click and set the flag to false.
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        # When INDEX is detected and the flag is set to true, perform a right mouse click and set the flag to false.
        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False

        # When TWO_FINGER_CLOSED is detected and the flag is set to true, perform a double click and set the flag to false.
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

        # When PINCH_MINOR is detected and the pinchminorflag is not set, initialize the pinch control and set the flag to true.
        # Then call pinch_control with the scrollHorizontal and scrollVertical functions.
        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result,Controller.scrollHorizontal, Controller.scrollVertical)
        
        # When PINCH_MAJOR is detected and the pinchmajorflag is not set, initialize the pinch control and set the flag to true.
        # Then call pinch_control with the changesystembrightness and changesystemvolume functions.
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result,Controller.changesystembrightness, Controller.changesystemvolume)
        
'''
----------------------------------------  Main Class  ----------------------------------------
    Entry point of Gesture Controller
'''


class GestureController:
    # Declare class-level variables
    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None # Right Hand by default
    hr_minor = None # Left hand by default
    dom_hand = True

    # Constructor method for initializing class-level variables
    def __init__(self):
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    # A static method to classify hands from the detected results
    def classify_hands(results):
        # Initialize left and right hand landmarks as None
        left , right = None,None
        try:
            # Retrieve handedness dictionary for the first hand and assign to right or left accordingly
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else :
                left = results.multi_hand_landmarks[0]
        except:
            pass

        try:
            # Retrieve handedness dictionary for the second hand and assign to right or left accordingly
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else :
                left = results.multi_hand_landmarks[1]
        except:
            pass
        
        # Assign major and minor hands based on the user's dominant hand
        if GestureController.dom_hand == True:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else :
            GestureController.hr_major = left
            GestureController.hr_minor = right

    # Start method to begin processing the camera frames and detecting gestures
    def start(self):
        # Instantiate two HandRecog objects for major and minor hands
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        # Begin capturing frames and detecting hands with MediaPipe Hands
        with mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                # Read a frame from the camera
                success, image = GestureController.cap.read()

                # Ignore empty camera frame
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Flip the image horizontally, convert it to RGB format, and process with MediaPipe Hands
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Update major and minor hand landmarks if hands are detected
                if results.multi_hand_landmarks:                   
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    # Determine the current gesture based on the dominant hand
                    handmajor.set_finger_state()
                    handminor.set_finger_state()
                    gest_name = handminor.get_gesture()

                    # Call the handle_controls method of the Controller class to take actions based on the detected gesture
                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)
                    
                    # Draw landmarks on the image
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    Controller.prev_hand = None

                # Show the processed image on the screen
                cv2.imshow('Gesture Controller', image)

                # Exit loop if enter key is pressed
                if cv2.waitKey(5) & 0xFF == 13:
                    break

        # Release the camera and destroy all windows
        GestureController.cap.release()
        cv2.destroyAllWindows()

#Calling the functions to start the program
gc1 = GestureController()
gc1.start()
