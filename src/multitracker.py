# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
import traceback
import glob
# from collections import deque

import PyQt5 
# from PyQt5 import QtCore
# from PyQt5 import QtWidgets

import cv2
import exiftool
import numpy as np
import pandas as pd

# from PyQt5.QtCore import QCoreApplication
# from PyQt5.QtWidgets import QApplication, QInputDialog
import crashlogger
import datalogger

#import mbox
import qt_dialog
# from Video import STFileVideoStream
import Video
import filters
import regression
from Regions import Regions
import time

import evaluate
CPU_COUNT = multiprocessing.cpu_count()

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

from copy import deepcopy

#start tracking version at 1.0

PEOPLETRACKER_VERSION = 2.8


class MultiTracker():
    '''!
        @brief The `MultiTracker` class is a Python class that represents a multi-object tracker and contains
        various attributes and methods for tracking objects.
        
        @param tab is a designated tab in the UI which will populate provided values.
        @param colour is a 255/255/255 RGB value that defines a tracking colour for easy distinction.

        Many values are retrieved from `tab` defined in TrackerTab.py which is generated in qt_dialog.py's App.

    '''
    def __init__(self, tab, colour=(255,255,255)):    
        ## Recorded name is grabbed from tab's name line text box
        self.name = tab.name_line 

        ## Recorded id is grabbed from tab's id line text box
        self.pid = tab.id_line

        ##(int, int, int) colour of the tracking box (not used) (rgb) 
        self.colour = colour

        # self.location_data = [] #(x, y)

        ## (CLOSE, MED, FAR) (estimated)
        self.distance_data = []

        ## frames tracked, the list structure is used to define the amount of time in the scene.
        self.time_data = []
        # self.region_dict = dict

        ## Recorded data will be stored in this dictionary for every recorded frame for every tracker
        self.data_dict = dict() 

        ## Initial previous time, used for time_data
        self.previous_time = 0 

        #Set these variables to getter functions (to update whenever accessed)

        ## STRING: gender applied if known, grabbed from tab 
        self.sex = tab.sex_line 

        ## INT: group is the number of trackers that arrive in the same group
        self.group = tab.group_line 

        ## STRING description of the individual
        self.description = tab.desc_line 

        ## TRUE/FALSE values defining if tracker is there in the beginning of the video
        self.beginning = tab.get_beginning

        ## A Region is an area that will flag true if other trackers are in it's proximity. The region as a tracker will track through a video rather than being stagnant (TRUE/FALSE)
        self.is_region = tab.get_is_region 

        ## Returns TRUE/FALSE if tracker 'is' a chair (or in one?)
        self.is_chair = tab.get_is_chair
        
        ## Read Only is a TRUE/FALSE button that turns off new tracking from being recorded and displays already recorded information.
        self.read_only = tab.get_read_only 

        ## ID is a unique id that differentiates trackers even if individuals have the same name.
        self.id = tab.get_uuid 
        
        ## TRUE/FALSE determines if the data should be recorded into data_dict or not.
        self.record_state = False

        ## This will be defined as one of OpenCV's implemented Trackers. (assign by using create() function)
        self.tracker = None 

        ## The initial bounding box is where the tracker will start from. (User Defined from UI or from MaskRCNN's assignment)
        self.init_bounding_box = None

        ## resets the tracker and asks for a new init_bounding_box
        self.reset = False 

        # self.state_tracking = False # UN-USED, MAYBE DELETE

        ## AutoAssign is a state flagged to allow the tracker to be re-assigned to the closest bounding box created by MaskRCNN's predictions. (IOU measure)
        self.auto_assign_state = True 

        ## The value that stores the current state of tracker's box, the successor values of init_bounding_box (Extract tracking values from this)
        self.box = None 

        ## The center predictor created by KalmanFilter (Filters.py) (center arrow)
        self.predictor = filters.KalmanPred() 

        ## One point predictor for two points on box (Top Left, Bottom Right) Creates a prediction box.
        self.box_predictor = (filters.KalmanPred(white=True), filters.KalmanPred(white=True))

        ## The red line that appears on the screen is a regression prediction of the current path of tracker's center
        self.regression = regression.rolling_regression(10, "Linear") 

        ## Predictions generated by MaskRCNN is stored here
        self.predicted_bbox = None

        ## Center of MaskRCNN is stored here.
        self.predicted_centroid = None 

        
        # self.recording_video = False # Not Implemented
        # self.svm_model = None # Not Implemented
        # self.bayes_model = None # Not Implemented

        # self.self_aware_gallery = [] # Not Implemented
        # self.normalized_gallery = [] # Not Implemented

    # def compare_predicted(self, location, frame_num, predictions):
    #     """
    #     Compares predicted file to tracked file in frame using IOU and overlapping measures.
    #     """

    ## OpenCV's Grabcut on tracker's position
    def grab_cut(self, frame, rect, export_path=None):
        """!
        Performing image segmentation using the GrabCut algorithm in OpenCV.

        @param frame int
        @param rect (int, int, int, int)

        @returns None
        """
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)


        mask = np.zeros(frame.shape[:2],np.uint8)

        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iterCount=2, mode=cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = frame*mask2[:,:,np.newaxis] 
        cv2.imshow("Cropped Image", img)
        
    ## Name of tracker
    def get_name(self):
        """!
        @returns string
        """
        return self.name.text()
    
    ## Tracker ID
    def get_id(self):
        """! @returns string """
        return self.pid.text()

    # def get_loc(self, frame):
    #     """ Returns pixel (x,y) location of person tracked: returns tuple(int,int)"""
    #     return self.location_data(frame)
    
    ## Tacker sex
    def get_sex(self):
        """! @returns string """
        return self.sex.text()

    ## Tracker description
    def get_description(self):
        """! @returns description of tracker string"""
        return self.description.toPlainText()

    ## Tracker's group size
    def get_group(self):
        """!
        Number of individuals in defined group returns a number in string form.

        @returns string
        """
        return self.group.text()

    ## Time tracker has been tracking relative to the video
    def get_time_tracked(self, framerate):
        """!
        Returns total time being tracked in video. The value is returned as a list. Idk why, the only place we call it we access it's first (and only) value...

        @note maybe dont return a list...

        @returns  list
        """
        
        # if len(self.data_dict) > 2:
        total_time = [self.calculate_total_time(self.part_time_to_segments(self.data_dict.keys()), framerate)]
        return total_time

    ## Creates base cv2.Tracker
    def create(self, tracker_type='CSRT'):
        """ !
        The creation of the Opencvs Tracking mechanism.
        By default we use the CSRT, or DCF-CSR @cite csrt
        Please refer to CSRT reference for any work since this is what we use.

        @param tracker_type ["BOOSTING", "MIL","KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]

        @cite opencv_library
        <a href="https://docs.opencv.org/3.4/d9/df8/group__tracking.html"> https://docs.opencv.org/3.4/d9/df8/group__tracking.html </a>

        @returns CV2.TRACKER
        """
            # tracker = cv2.Tracker_create(tracker_type)
        if tracker_type == 'BOOSTING':
            #Based on the same algorithm used to power the machine learning behind Haar cascades (AdaBoost), but like Haar cascades, is over a decade old. This tracker is slow and doesn’t work very well. Interesting only for legacy reasons and comparing other algorithms.
            self.tracker = cv2.TrackerBoosting_create()

        if tracker_type == 'MIL':
            # Better accuracy than BOOSTING tracker but does a poor job of reporting failure.
            self.tracker = cv2.TrackerMIL_create()

        if tracker_type == 'KCF':
            #Kernelized Correlation Filters. Faster than BOOSTING and MIL. Similar to MIL and KCF, does not handle full occlusion well.
            self.tracker = cv2.TrackerKCF_create()
            
        if tracker_type == 'TLD':
            #prone to false positives. I do not recommend using this OpenCV object tracker.
            self.tracker = cv2.TrackerTLD_create()

        if tracker_type == 'MEDIANFLOW':
            # Does a nice job reporting failures; however, if there is too large of a jump in motion, such as fast moving objects, or objects that change quickly in their appearance, the model will fail.
            self.tracker = cv2.TrackerMedianFlow_create()

        if tracker_type == 'GOTURN':
            #The only deep learning-based object detector included in OpenCV. Reportedly handles viewing changes well 
            #NOTE must download goturn.protext from https://github.com/Mogball/goturn-files.
            #This is already downloaded under the models/ folder, will include.
            self.tracker = cv2.TrackerGOTURN_create()

        if tracker_type == 'MOSSE':
            #Very, very fast. Not as accurate as CSRT or KCF but a good choice if you need pure speed
            self.tracker = cv2.TrackerMOSSE_create()

        if tracker_type == "CSRT":
            #Discriminative Correlation Filter (with Channel and Spatial Reliability). Tends to be more accurate than KCF but slightly slower.
            self.tracker = cv2.TrackerCSRT_create()


    ## UNUSED  renames tracker
    def rename(self, name):
        """
        Renames the tracking object
        """
        self.name = name
        
    ## UNUSED sets tracker's box colour
    def set_colour(self, color=(255,255,255)):
        self.colour = color

    ## Creates bounding box of tracker (user draws the box)
    def assign(self, frame, tracker_type="CSRT"):
        """
        Assigns the box of a tracked person.
        Takes care of reassigning as well.

        Leverages cv2.selectROI and creates init_bounding_box value.
        """
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        self.init_bounding_box = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
        #if not selected, stop until it is
        # while self.init_bounding_box[0] is 0 and self.init_bounding_box[1] is 0 and self.init_bounding_box[2] is 0 and self.init_bounding_box[3] is 0:
        # while self.init_bounding_box[2] == 0 or self.init_bounding_box[3] == 0:
        #     input_dialog.log("No object selected, draw a rectangle with an area larger than 0.")
            # self.init_bounding_box = cv2.selectROI("Frame", frame, fromCenter=False,
            #     showCrosshair=True)
        if self.init_bounding_box[2] == 0 or self.init_bounding_box[3] == 0:
            input_dialog.log("No object selected, draw a rectangle with an area larger than 0, Try again")
            self.init_bounding_box = None
            return

        input_dialog.log("Bounding Box Coordinates: " + str(self.init_bounding_box))
        self.box = self.init_bounding_box

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        if self.init_bounding_box is not None:
            # self.tracker.init(frame, self.init_bounding_box)
            input_dialog.log("Testing bounding box exists: " + str(self.reset))
            self.reset = True

        if self.reset is True:
            # try:
            input_dialog.log("Setting Location")
            del self.tracker
            self.create(tracker_type)
            self.tracker.init(frame, self.init_bounding_box)
            input_dialog.log(("Setting Location Successful. " + "Bounding Box Coordinates: " + str(self.init_bounding_box)))
            
            # Tell the predictors to reset
            self.box_predictor[0].reset()
            self.box_predictor[1].reset()
            self.predictor.reset()

            # except Exception as e:
            #     crashlogger.log(str(e))
            #     input_dialog.log("Setting Location Failed.")
    

    ## assign() equivelant for automatic tracking (MaskRCNN)
    def auto_assign(self, frame, bounding_box=None, xywh=None, tracker_type="CSRT"):
        '''!
        Auto assign creates a new tracker at position defined.
        Depending on the data at hand, we use either a bounding box as 2 locations or a top left with width and height.
        Use bounding_box if you have 2 locations, and xywh if you have the top left and dimensions.

        Uses opencv's tracker's init function <a href="https://docs.opencv.org/4.x/d0/d0a/classcv_1_1Tracker.html#a7793a7ccf44ad5c3557ea6029a42a198"> defined here</a>

        @param frame <b>array</b> The cv2 frame in the video
        @param bounding_box <b>tuple</b> the top left and bottom right locations of the box.
        @param xywh <b>tuple</b> The top left corner and the width and height as the last 2 values.
        @param tracker_type string

        @returns None
        '''
        # try:
        # print(self.init_bounding_box)
        # print(bounding_box)
        if xywh is None:
            width = bounding_box[2] - bounding_box[0]
            height = bounding_box[3] - bounding_box[1]
            bounding_box = [bounding_box[0], bounding_box[1], width, height]
        else:
            bounding_box = xywh
        # print(bounding_box)
        input_dialog.log("Setting Location")
        del self.tracker
        self.create(tracker_type)
        try:
            #
            self.tracker.init(frame, bounding_box)
            input_dialog.log("Setting Location Successful.")
        except Exception as e:
            crashlogger.log(str(e))
            input_dialog.log("Setting Location Failed.")

    ## Cv2 tracker to update for next frame, and we draw the new location onto the frame.
    def update_tracker(self, frame):
        """
        track and draw box on the frame (image)
        
        This function is only done on the one tracker defined on self.

        @param frame <b>array</b> cv2 image
        """
        success = False
        box = None
        # check to see if we are currently tracking an object
        if self.init_bounding_box is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = self.tracker.update(frame)
            self.box = box
    
            # check to see if the tracking was a success
            if success:
                frame = self.draw_tracker(frame, box)

                # (x, y, w, h) = [int(v) for v in box]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), self.colour, 2)
                # cv2.rectangle(frame, (x , y - 1), (x + 10 * (len(self.get_name())) , y - 15),(255,255,255),-1)
                # cv2.putText(frame,self.get_name(), (x , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)
                
                # cv2.putText(frame,self.get_name(), (x , y - 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,0,0),1)
        return success, box, frame

    ## Draws the bounding box of the tracker and places the name ontop.
    def draw_tracker(self, frame, box, opacity=0.5, name=True):
        '''!
        Draws the bounding box of the tracker and places the name over top with an white opacity background for readability.

        @param frame <b>array</b> cv2 image
        @param box <b>tuple</b> (x,y,width,height) x and y are top left coordinates
        @param opacity <b>float</b> The opacity of the white background behind the name
        @param name <b>bool</b> Places the self.name value over top left corner of the box
        '''
        
        (x, y, w, h) = [int(v) for v in box]

        # Apply translucent layer
        alpha_layer = np.zeros_like(frame, np.uint8)
        cv2.rectangle(alpha_layer, (x , y - 1), (x + 10 * (len(self.get_name())) , y - 15),(255,255,255),-1)
        out = frame.copy()
        alpha = opacity
        mask = alpha_layer.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, alpha_layer, 1 - alpha, 0)[mask]

        # Apply opaque name and bounding box ontop of translucent layer
        cv2.rectangle(out, (x, y), (x + w, y + h), self.colour, 2)
        if name is True:
            cv2.putText(out, self.get_name(), (x , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)

        return out

    ## Not called anywhere
    def remove(self):
        """!
        Removes the tracker from being recorded. DOES NOT YET DELETE INFO
        """
        del self.tracker
        self.init_bounding_box = None
        # del self

    ## Not implemented
    def predict(self,):
        """!
        Applies KNN taking into consideration a Hidden Markov Model
        @Warning Not implemented or used
        """
        pass

    def record_data(self, frame, total_people, x=-1, y=-1, w=-1, h=-1, regions=[], other_room=False, image_frame=None):
        """!
        Appends location and time to a list of data in data_dict
        
        @param frame <b>int</b> frame is the integer number of the current frame
        @param total_people <b>int</b> a spinbox value that is the TOTAL number of people in the room.
        @param x <b>int</b> top left x value of the bounding box
        @param y <b>int</b> top left y value of the bounding box
        @param w <b>int</b> width of the bounding box
        @param h <b>int</b> height of the bounding box
        @param regions <b>list</b> a list of regions (strings) that the current tracker is intersecting with
        @param other_room <b>bool</b> Other room is defined by a checkbox. Used when individual is still in the scene but occluded.
            This is exactly the same as read only but continues counting time recording.

        @param image_frame <b>array</b> A cv2 image frame
        
        @note X,Y,W,H No Data are described as -1, and regions is described as [] or empty
        """
        # input_dialog.log("recording Frame" + str(frame))
        # self.location_data.append((int(x),int(y))) #(x, y)
        # self.time_data.append(frame) #(frames tracked)
        point = (int(x),int(y))
        dimensions = (int(w), int(h))
        self.data_dict[frame] = (point, regions, dimensions, other_room, total_people, self.is_chair())

        if image_frame is not None:
            x = int(x)
            y = int(y)
            h = int(h/2)
            w = int(w/2)
            image = image_frame[y-h:y+h, x-w:x+w]
            cv2.imshow("SELF AWARE", image)
            
            # self.self_aware_gallery.append()
            self.run_bayes_model(image, image_frame)
        # if frame:
        #     if not self.recording_video:
        #         video_name = self.get_name() + "mp4"
        #         self.get_name()
        #         self.video_recorder = cv2.VideoWriter(video_name, -1, 30, (input_dialog.width, input_dialog.height))
                
        #     if self.recording_video:
        #         self.video_recorder.write(image_frame)

    ## Do not use.
    def run_bayes_model(self, new_img, whole_image):
        '''!
        @warning DO NOT USE - Experimental naive bayes predictor for the visual similarity of current tracker score
        '''

        cv2.imshow("WHOLE", whole_image)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY).reshape(-1,1) 
        whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2GRAY).reshape(-1,1) 

        joint_img = np.append(new_img, whole_image).reshape(-1,1) 
        # print(joint_img)

        try:
            if self.bayes_model is None:
                self.bayes_model = Perceptron()

            # print(new_img.shape)
            scaler = StandardScaler().fit(joint_img)
            joint_img = scaler.transform(joint_img)

            name_array = np.array([self.get_name()]*new_img.shape[0])
            other_array = np.array(["None"]*whole_image.shape[0])
            joint_labels = np.append(name_array, other_array)

            self.bayes_model.partial_fit(joint_img, joint_labels, classes=np.unique([self.get_name(), "None"]))

            pred_2 = self.bayes_model.predict(new_img)
            # probability = self.bayes_model.predict_proba(new_img)
            # print(pred_1, pred_2)
            # print("pred:", pred_2)
            # print(pred_2)
            # a = np.sum(pred_2 == '')
            # b = len(pred_2)

            # if a/b is not 0 or 1:
                # print(a / b)
            # print(np.mean(probability))
            # print("Probability", np.mean(probability))
        except Exception as e:
            print(e)

        # self.distance_data.append(self.estimate_distance(size)) #(CLOSE, MED, FAR) (estimated)

    ## NOT IMPLEMENTED
    def append_data(self, dataframe):
        """!
        Appends the data into a given dataframe categoriezed with the name given
        @warning Not implemented
        """
        pass
    
    ## Exports tracked data into csv of the given name and directory of the video
    def export_data(self, vid_width, vid_height, vid_name, fps, new_version=None):
        """!
        Exports the recorded data and appends information such as name, box locations, total time recorded, and pixel percent locations, etc. into a csv the same name of the video
        
        Uses tracker information saved in data_dict and extends some information passed in from the video which may change such as vid_height and vid_width which is affected by resizeing the video at runtime

        @param vid_width <b>int</b> resized video width
        @param vid_height <b>int</b> resized video height
        @param vid_name <b>string</b> video name
        @param fps <b>float/int</b> fps to reference the rate the video runs at
        @param new_version <b>int</b> a version to define the revision of the tracker you're working with. Automatically has a new version when you save and then load the tracker back up. This is important if you import a tracker, and re-export the same tracker into the same dataframe, this version will differentiate the two, even if majority if the data is the same. Good to highlight differences between states.
        
        @note new_version will export a new csv with the version name attatched.
        """

        

        # if not os.path.exists(("./data/" + vid_name[:-4])):
        #     os.makedirs(("./data/" + vid_name[:-4]))
        # self.data_dict[frame] = (point, regions, dimensions, other_room, total_people, self.is_chair())
        # input_dialog.log(self.data_dict)

        #Generate the base dataframe to fill and file
        export_filename = str(vid_name[:-4]) + ".csv"
        
        frames = list(self.data_dict.keys())
        location = list(self.data_dict.values())

        #elaborate on the location, record it in percent
        perc_x_list = []
        perc_y_list = []
        pixel_location_x = []
        pixel_location_y = []

        bounding_box_top_left_x = []
        bounding_box_top_left_y = []
        bounding_box_bottom_right_x = []
        bounding_box_bottom_right_y = []

        region_list = []
        other_room_list = []
        total_people_list = []
        is_chair_list = []
        
        #Iterate through data and record
        for data in location:
            
            center = data[0]        # center -> (x,y)
            # print("Center", center)


            corrected_center = (center[0], (vid_height - center[1]))
            # print(corrected_center)

            dimentions = data[2]    # dimentions -> (width, height) where (0,0) is top left

            pixel_location_x.append(center[0])
            pixel_location_y.append((vid_height - center[1]))

            #Get top left and bottom right corner
            bounding_box_top_left_x.append( (center[0] - dimentions[0]/2) )
            bounding_box_top_left_y.append( (vid_height - center[1]) + dimentions[1]/2)

            bounding_box_bottom_right_x.append( (center[0] + dimentions[0]/2) )
            bounding_box_bottom_right_y.append((vid_height - center[1]) - dimentions[1]/2)

            #Handle invalid data to maintain consistency
            if corrected_center == -1:
                perc_x = -1
                perc_y = -1
                perc_x_list.append(round(perc_x,2))
                perc_y_list.append(round(perc_y,2))

                # top_perc.append(round(-1,2))
                # bottom_perc.append(round(-1,2))

            else:
                perc_x = (corrected_center[0]/vid_width)*100
                perc_y = (corrected_center[1]/vid_height)*100
                perc_x_list.append(round(perc_x,2))
                perc_y_list.append(round(perc_y,2))

                # top_perc.append(round( ((top[1]/vid_height)*100), 2))
                # bottom_perc.append(round( ((bottom[1]/vid_height)*100), 2))



            region_string = ""
            for index, text in enumerate(data[1]):
                #If there is no item, 1 item or the item is the last item in the list, do not append comma and space
                if len(data[1]) <= 1 or index == len(data[1]) -1:
                    region_string += (text)
                else:
                    region_string += (text + ", ")

            region_list.append(region_string)
            other_room_list.append(data[3])
            total_people_list.append(data[4])
            is_chair_list.append(data[5])

        # input_dialog.log(len(frames))
        # input_dialog.log(pixel_location)
        # input_dialog.log(len(pixel_location))
        # input_dialog.log(region_list)
        # input_dialog.log(len(region_list))

        #extend all the data so it can be exported. All data should be of the same length.
        MAX_LEN = len(frames)
        sex = [self.get_sex()]
        name = [self.get_name()]
        pid = [self.get_id()]
        group = [self.get_group()]
        description = [self.get_description()]
        beginning = [self.beginning()]

        max_width = [vid_width]
        max_height = [vid_height]

        # total_time = [self.(self.part_time_to_segments(self.time_data))]
        total_time = self.get_time_tracked(fps)
        total_time[0] += self.previous_time
        self.previous_time = total_time[0]
        sex.extend([sex[0]]*(MAX_LEN-1))
        name.extend([name[0]]*(MAX_LEN-1))
        pid.extend([pid[0]]*(MAX_LEN-1))
        group.extend([group[0]]*(MAX_LEN-1))
        description.extend([description[0]]*(MAX_LEN-1))
        beginning.extend([beginning[0]]*(MAX_LEN-1))

        max_width.extend([max_width[0]]*(MAX_LEN-1))
        max_height.extend([max_height[0]]*(MAX_LEN-1))

        total_time.extend([total_time[0]]*(MAX_LEN-1))
        # total_people.extend([total_people[0]]*(MAX_LEN-1))

        time_rec = total_time[0]
        print("Time Recorded", time_rec)
        seconds = [round((time_rec)%60,2)]
        minutes = [int(((time_rec)/60)%60)]
        hours = [int(((minutes[0])/60)%60)]

        seconds.extend([seconds[0]]*(MAX_LEN-1))
        minutes.extend([minutes[0]]*(MAX_LEN-1))
        hours.extend([hours[0]]*(MAX_LEN-1))


        print(hours[0], minutes[0], seconds[0])
        
        
        #Create the dataframe
        export_df = {"Frame_Num":frames,#self.time_data,

            "Pixel_Loc_x": pixel_location_x,
            "Pixel_Loc_y": pixel_location_y,
            "Perc_X": perc_x_list, "Perc_Y": perc_y_list,

            "Max_Pixel_x":max_width,
            "Max_Pixel_y":max_height,

            "BBox_TopLeft_x":bounding_box_top_left_x,
            "BBox_TopLeft_y":bounding_box_top_left_y,
            "BBox_BottomRight_x":bounding_box_bottom_right_x,
            "BBox_BottomRight_y":bounding_box_bottom_right_y,

            "Region": region_list,
            # "TimeInRegion":,
            "Name": name,
            "ID":pid, 
            "Sex":sex,
            "Group_Size":group,
            "Total_Sec_Rec":total_time,
            "Time_Rec(Hour)":hours,
            "Time_Rec(Min)":minutes,
            "Time_Rec(Sec)":seconds,
            "Description":description,
            "Beginning":beginning,
            "Other_Room":other_room_list,
            "Chair":is_chair_list,
            "Total_People":total_people_list
            }
        
        df = pd.DataFrame(export_df)

        
        # If version, check for new versions
        if new_version is not None:
            out_path, old_filename = os.path.split(vid_name)
            new_filename = (out_path + "/" + new_version[0])

            # If file does not exist, generate metadata
            files = glob.glob((out_path+"/*.csv"))
            file_exists = False
            for file in files:
                if new_version[0] in file:
                    file_exists = True
            
            if not file_exists:
                export_meta(vid_name, output_csv=new_filename)

            # change export filename to the new version
            export_filename = new_filename
        
        export_csv = df.to_csv (export_filename, index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path

        self.data_dict = dict()

        input_dialog.log("Export Data Complete!")
        
    ## NOT USED
    def evaluate_data(self):
        '''
        @warning not used
        '''
        te = evaluate.tracker_evaluation()
        
    ## Merges a list of integer segments of beginning/end time into one interval that contains the first and last values of all segments.
    def merge_intervals(self, total_segments):
        """!
        On a continous set of intervals, merges them such that intersecting inervals will become a new interval with new limits. 
        Example: [[0, 20], [10, 50]] becomes [[0,50]]
        [[0, 500]] stays the same

        @param total segments is a list of lists where the second dimension of lists is an interval of numbers (start frame and end frame)
        """
        # print("Merge_Interval", total_segments)
        
        try:
            assert(len(total_segments) >= 1)
            total_segments.sort(key=lambda interval: interval[0])
            merged = [total_segments[0]]
            for current in total_segments:
                previous = merged[-1]
                if current[0] <= previous[1]:
                    previous[1] = max(previous[1], current[1])
                else:
                    merged.append(current)
        except Exception as e:
            print("Handling single element in merge_intervals by duplicating single value", e)
            return [[total_segments[0][0], total_segments[0][0]]]

        return merged

    ## splits the time into beginning/end intervals given segment size as the largest absent gap
    def part_time_to_segments(self, time_data, segment_size=300):
        """!
        Parts the times into segments based on the distance between tracked frames.
            time_data is the entire time of the object tracked.
            segment_size is the distance between frames that should be considered as one segment. Default is 10 seconds at 30fps.
            min_segment is the minimum size of segment allowed. If it is too small, it will be ignored. Defualt is 1 second (30 frames).

        Example with default segment size of 300:
            [0, 10, 20, 30, 100, 200, 300, 400, 1000, 1010, 1020] results in [[0, 400], [1000, 1020]]
        
        @param time_data a list of integers to be split up into interval pairs. The split is defined by the gap between numbers set by segment_size
        @param segment_size the threshold to split the list
        """
        if len(time_data) < 2:
            return [[0]]

        time_data = list(sorted(time_data))

        #three segment markers, starting frame, current frame, and end frame
        try:
            seg_start = time_data[0]
            seg_last = time_data[0]
        except:
            # raise ValueError
            return [[0]]
        total_segments = [] # total segments placed in a list


        # iterate through all the times, start comparing time[0] with time[1]
        for time in time_data:
            #if the size between last segment time and current time is less than segment threshold
            if ((time - seg_last) <= segment_size):
                # input_dialog.log("Stepping from " + str(seg_last) + " to " + str(time) )
                seg_last = time
            
            #if the size between last segment time is greater than the threshold, we create a new segment and check if it has been tracked long enough
            if ((time - seg_last) > segment_size):
                # input_dialog.log("Past Threshold, segmenting time from " + str(seg_start) + " to " + str(seg_last))
                if seg_start is not seg_last:
                    #add the start and end of current segment as a pair
                    total_segments.append([seg_start, seg_last])
      
                seg_start = time
                seg_last = time
            
            #if we reach the end of the tracked data, we end the segments and close the loop
            if (time == time_data[-1]):
                if seg_start == seg_last or seg_start >= seg_last:
                    break
                total_segments.append([seg_start, seg_last])
                break
        #     print("TOTAL_SEGMENTS EMPTY")
        #     return [[time_data[0], time_data[0]]]
        # if not total_segments:
        #merge intervals overlapping to maintain efficiency
        if len(total_segments) >= 2:
            total_segments = self.merge_intervals(total_segments)
            if total_segments == None:
                return [[0]]
        elif len(total_segments) == 1:
            total_segments = [total_segments[0]]
        else:
            total_segments = [[0]]

        return total_segments

    ## NOT IMPLEMENTED
    def estimate_distance(self, size):
        '''!
        @warning not inplemented
        '''
        pass
    
    ## Calclulates time between two frames at a given framerate
    def calculate_time(self, frame_start, frame_end, fps=30):
        """!
        Calculates time between two given frames given the FPS rate.

        frame_end must be larger than frame_start or you will be given a negative number of seconds.

        @param frame_start <b>int</b> the starting frame number
        @param frame_end <b>int</b> the ending frame number
        @param fps <b>float</b> the framerate of the video

        @returns <b>float</b> a value in seconds between frames
        """
        time = (frame_end - frame_start)/fps
        return time
    
    # Total time tracked where non-tracked frames are considered.
    def calculate_total_time(self, total_frames, fps=30, segmented=True):
        """!
        Calculates total time tracked. This takes into consideration segmented time

        @param total_frames <b>list</b> list of total frames. This will segment the time given default gaps of 300 frames min, then calculate the total time of present frames.
        @param fps <b>float</b> the fps of the video to determine the rate of frames which defines the real time length between frames
        @param segmented <b>bool</b> if the data in total_frames is segmented or not.
        
        @returns <b>float</b> the total time in seconds
        """
        total_time = 0
        if segmented is False:
            time_segs = self.part_time_to_segments(total_frames)
        else: 
            time_segs = total_frames
        #sum all the segments together
        for seg in time_segs:
            if len(seg) < 2:
                seg = [seg[0], seg[0]]
            first = seg[0]
            last = seg[1]
            #each seg is a pair of start and end times
            total_time += self.calculate_time(first, last, fps)

        #May not need, no negative time allowed.
        if total_time <= 0:
            # print("Time less than or equal to Zero")
            total_time = 0

        return total_time

## Exports a dataset full of NULL valriables at the beginning. Used if metadata fails to extract.
def export_null_meta(vid_dir):
    '''
    Exports a dataset full of NULL valriables at the beginning. Used if metadata fails to extract.
    
    @param vid_dir <b>string</b> the directory which the video exists in
    '''
    #Create path string for exporting the data. It's just a change of extention.
    export_filename = None

    # if its a number assume live
    if type(vid_dir) == type(0):
        export_filename = time.strftime("%Y_%m_%d-%H_%M_%S") + ".csv"
    else:
        export_filename = str(vid_dir[:-4]) + ".csv"
    
    #Create the first 2 rows of data with title and then info. Recorded info should have '-' or 'N/A'
    if not os.path.isfile(export_filename):
        data ={
            "Frame_Num":['-'],#self.time_data,

            "Pixel_Loc_x":['-'],
            "Pixel_Loc_y":['-'],

            "Perc_X":['-'], "Perc_Y":['-'],

            "Max_Pixel_x":['-'],
            "Max_Pixel_y":['-'],

            "BBox_TopLeft_x":['-'],
            "BBox_TopLeft_y":['-'],
            "BBox_BottomRight_x":['-'],
            "BBox_BottomRight_y":['-'],

            "Region": ['-'],
            # "TimeInRegion":['-'],
            "Name":['-'],
            "ID":['-'],
            "Sex":['-'], 
            "Group_Size":['-'],

            "Total_Sec_Rec":['-'],

            "Time_Rec(Hour)":['-'],
            "Time_Rec(Min)":['-'],
            "Time_Rec(Sec)":['-'],

            
            "Description":['-'],
            "Present At Beginning":['-'],
            "Other_Room":['-'],
            "Chair":['-'],
            "Total_People":['-'],

            "FileName":['-'],
            "FileType":['-'],
            "CreateDate(YYYY:MM:DD HH:MM:SS)": ['-'],
            "CreateYear":['-'], "CreateMonth": ['-'], "CreateDay": ['-'],
            "CreateHour": ['-'], "CreateMinute": ['-'], "CreateSecond": ['-'],

            "Width(px)": ['-'],
            "Height(px)": ['-'],
            "FrameRate": ['-'],
            "VideoLength(Hour)":['-'],
            "VideoLength(Min)":['-'],
            "VideoLength(Sec)":['-'],
            "HandlerDescription": ['-'],
            "PeopleTrackerVersion":float(PEOPLETRACKER_VERSION)
        }
        input_dialog.log(data)

        #Create a dataframe and then export it.
        df = pd.DataFrame(data,index=[1])
        export_csv = df.to_csv (export_filename, index = None, header=True)

        input_dialog.log("Exported Null Metadata!")

# Exports metadata into the same csv that data is recorded. This is called first and populates the first two rows of data containing metadata.
def export_meta(vid_dir, output_csv=None):
    """!

    Exports a start line into a CSV with the same name as the video. This CSV will initialize with a set of columns on the left
    with dashed lines for recorded data, and a set of columns on the right with Metadata.
    This should only be called once, and will only initialize if the file does not exist- will not overwrite.

    @param vid_dir <b>string</b> directory of filename
    """
    #get metadata from filepath
    #NOTE: Exiftool must be in path to access, otherwise pass a string to the path.
    try:

        with exiftool.ExifToolHelper(executable="F:/Exiftool/exiftool.exe") as et:
            metadata = et.get_metadata(vid_dir)[0]
            print(metadata)
            print("collected metadata")

    except Exception as e:
        print(e)

        try:
            # if LooseVersion(exiftool.__version__) < LooseVersion("0.5.5"):
            try:
                with exiftool.ExifTool() as et:
                    metadata = et.get_metadata(vid_dir)
                    print(metadata)
            except Exception as e:
                print(e)
                with exiftool.ExifToolHelper() as et:
                    metadata = et.get_metadata(vid_dir)[0]
                    print(metadata)
        except Exception as e:
            crashlogger.log(str(e))
            print("Could not collect")
            export_null_meta(vid_dir)
            return None




            # return False

    # try:
        #Parse date data into [YYYY MM DD HH MM SS]
    date_created = metadata['QuickTime:CreateDate'].replace(":"," ")
    date = list(map(int,date_created.split()))
    
    duration = metadata['QuickTime:Duration']
    seconds = round((duration)%60,2)
    minutes = int(((duration)/60)%60)
    hours = int(((minutes)/60)%60)
    print(hours, minutes, seconds)
    #Create path string for exporting the data. It's just a change of extention.
    if output_csv is not None:
        export_filename = output_csv
    else:
        export_filename = str(vid_dir[:-4]) + ".csv"
    print(metadata)
    try: 

        handler = metadata['QuickTime:HandlerDescription']
    except:
        try:
            handler = metadata['EXIF:Make'] + ' ' + metadata['EXIF:Model']
        except:
            handler = metadata['MakerNotes:Make'] + ' ' + metadata['MakerNotes:Model']
        
    #Create the first 2 rows of data with title and then info. Recorded info should have '-' or 'N/A'
    if not os.path.isfile(export_filename):
        
        data ={
            "Frame_Num":['-'],#self.time_data,

            "Pixel_Loc_x":['-'],
            "Pixel_Loc_y":['-'],

            "Perc_X":['-'], "Perc_Y":['-'],

            "Max_Pixel_x":['-'],
            "Max_Pixel_y":['-'],

            "BBox_TopLeft_x":['-'],
            "BBox_TopLeft_y":['-'],
            "BBox_BottomRight_x":['-'],
            "BBox_BottomRight_y":['-'],

            "Region": ['-'],
            # "TimeInRegion":['-'],
            "Name":['-'],
            "ID":['-'],
            "Sex":['-'], 
            "Group_Size":['-'],

            "Total_Sec_Rec":['-'],

            "Time_Rec(Hour)":['-'],
            "Time_Rec(Min)":['-'],
            "Time_Rec(Sec)":['-'],

            
            "Description":['-'],
            "Present At Beginning":['-'],
            "Other_Room":['-'],
            "Chair":['-'],
            "Total_People":['-'],

            "FileName": metadata['File:FileName'],
            "FileType": metadata['File:FileType'],
            "CreateDate(YYYY:MM:DD HH:MM:SS)": metadata['QuickTime:CreateDate'],
            "CreateYear": date[0], "CreateMonth": date[1], "CreateDay": date[2],
            "CreateHour": date[3], "CreateMinute": date[4], "CreateSecond": date[5],

            "Width(px)": metadata['QuickTime:ImageWidth'],
            "Height(px)": metadata['QuickTime:ImageHeight'],
            "FrameRate": metadata['QuickTime:VideoFrameRate'],
            "VideoLength(Hour)":hours,
            "VideoLength(Min)":minutes,
            "VideoLength(Sec)":seconds,
            "HandlerDescription": handler,
            "PeopleTrackerVersion":float(PEOPLETRACKER_VERSION)
        }
        input_dialog.log(data)
    # except:
    #     export_null_meta(vid_dir)
    #     return False
        

        #Create a dataframe and then export it.
        df = pd.DataFrame(data,index=[1])
        export_csv = df.to_csv (export_filename, index = None, header=True)

        input_dialog.log("Export Metadata Complete!")
        # return metadata
    return metadata

## Loads data from csv into multiple trackers
def load_tracker_data(csv, input_dialog, frame):
    '''
    Populates tabs, data_dict and trackers previously labelled from csv.
    All trackers will start in Read Only mode.

    Useful for when the data needs to be saved/loaded between recording sessions.

    Please note CSV must match the video or data will be uninformative.

    If you modify the data by hand errors may occur because of different formats, and how third party programs save NULL and TRUE/FALSE values.
    
    @param input_dialog defined in qt_dialog.
    @param frame cv2 image
    '''
    print("Loading csv")
    new_trackers = []

    df = pd.read_csv(csv)
    df = df.loc[1:] # Ignore first line, this is metadata
    df[["Name"]].fillna("")
    df[["ID"]].fillna("")

    # make sure columns are the proper data types
    # df["Present At Beginning"] = df["Present At Beginning"].astype('bool')
    # df["Other_Room"] = df["Other_Room"].astype('bool')
    # df["Chair"] = df["Chair"].astype('bool')

    # df["Frame_Num"] = df["Frame_Num"].astype('int32')


    # unique_trackers = df.groupby(['Name','ID'], as_index=False).size()
    # unique_trackers = df.groupby(['Name','ID']).size().reset_index().rename(columns={0:'count'})
    # print
    unique_trackers = df[['Name', 'ID']].drop_duplicates()
    unique_trackers = unique_trackers[unique_trackers['Name'].notna()]
    for name_index in range(len(unique_trackers)):
        print(name_index)
        # get the unique ids
        name = unique_trackers["Name"].iloc[name_index]
        pid = unique_trackers["ID"].iloc[name_index]


        # grab all the columns where this data exists.
        tracker_data = df.loc[(df['Name'] == name)]
        # print(tracker_data)
        # Add a new tab to the 
        tab_index, new_tab = input_dialog.add_tab()
        input_dialog.add_tab_state = False

        # Fill string columns NaN to empty string
        tracker_data["Name"] = tracker_data["Name"].fillna("")
        tracker_data["ID"] = tracker_data["ID"].fillna("")
        tracker_data["Sex"] = tracker_data["Sex"].fillna("")
        tracker_data["Description"] = tracker_data["Description"].fillna("")
        tracker_data["Group_Size"] = tracker_data["Group_Size"].fillna(0)
        tracker_data["Region"] = tracker_data["Region"].fillna("")
        print(tracker_data.iloc[1])
        # Build tab info, this tab info is used to build the tracker
        new_tab.name_line.setText(tracker_data["Name"].iloc[1])
        new_tab.id_line.setText(tracker_data["ID"].iloc[1])
        new_tab.sex_line.setText(str(tracker_data["Sex"].iloc[1]))
        new_tab.group_line.setText(str(tracker_data["Group_Size"].iloc[1]))

        description_text = ""
        if tracker_data["Description"].iloc[1] == "":
            description_text = str(tracker_data["Description"].iloc[1]) + "V"
        else:
            description_text = str(tracker_data["Description"].iloc[1]) + "\nV"

        description_text += str(input_dialog.newest_version)

        new_tab.desc_line.setPlainText(description_text)
        new_tab.update_length_tracked(float(tracker_data["Total_Sec_Rec"].iloc[1]))
        new_tab.read_only_button.setChecked(True)
        new_tab.read_only = True
        tracker_data['Present At Beginning'] = (tracker_data['Present At Beginning'] == 'TRUE')

        try:
            new_tab.beginning_button.setChecked(eval(tracker_data["Present At Beginning"].iloc[1]))
        except:
            print(tracker_data["Present At Beginning"].iloc[1])
            print("Error occured when setting to beginning while loading. Setting it to False by default")
            new_tab.beginning_button.setChecked(False)



        
        # Build new tracker with info loaded
        tracker = MultiTracker(new_tab)
        tracker.reset = True
        
        # Assign static variables not from tab to Multitracker object
        # tracker.beginning = 

        
        # Populate data dict
        for index, row in tracker_data.iterrows():
            frame_num = int(row["Frame_Num"])
            center = (float(row["Pixel_Loc_x"]), float(row["Max_Pixel_y"]) - float(row["Pixel_Loc_y"]) )
            regions = str(row["Region"]).strip("[]").split(", ")
            height = abs(float(row["BBox_TopLeft_y"]) - float(row["BBox_BottomRight_y"]))
            width = abs(float(row["BBox_TopLeft_x"]) - float(row["BBox_BottomRight_x"]) )

            dimensions = (int(width), int(height))

            other_room = eval(str(row["Other_Room"]))
            is_chair = eval(str(row["Chair"]))
            total_people = int(row["Total_People"])


            tracker.data_dict[frame_num] = (center, regions, dimensions, other_room, total_people, is_chair)
        
        current_frame = input_dialog.get_scrollbar_value()
        tracked_frames = list(tracker.data_dict.keys())

        # Set tracker to closest frame
        closest = tracked_frames[0]
        min_difference = abs(current_frame - closest)
        for value in tracked_frames[1:]:
            difference = abs(current_frame - value)
            if difference < min_difference:
                closest = value
                min_difference = difference


        data = tracker.data_dict[closest]

        w = int(data[2][0])
        h = int(data[2][1])
        x = int(data[0][0])
        y = int(data[0][1])

        # Correct for invalid non-area data
        if w <= 0:
            w = 5
        if h <= 0:
            h = 5
        if x <= 0:
            x = 10
        if y <= 0:
            y = 10

        tracker.init_bounding_box = (x, y, w, h)
        frame = cv2.circle(frame, center=(x,y), radius=3, color=(5,5,5), thickness=2)
        frame = cv2.rectangle(frame, (x-int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), color = (255, 0, 0), thickness = 2)
        tracker.box = tracker.init_bounding_box
        tracker.auto_assign(frame, xywh=((x-int(w/2)), (y - int(h/2)), (x + int(w/2)), (y + int(h/2))))

        #cv2.imshow("Loading", frame)
        # tracker.auto_assign(frame, (p1[0], p1[1], p2[0], p2[1]))
        # self.tracker.init(frame, init_bounding_box)

        # Append new tracker to list
        new_trackers.append(tracker)


    #Close all applications.
    # sys.exit(app.exec_())
    # cap.release()
    # fvs.stop()
    # cv2.destroyAllWindows()
    print("End of loading.")
    return new_trackers, frame



## Function that runs the entire multitracker program
def run(video_path=None):
    '''
    @note this function is VERY long and needs to be refactored.
    '''
    # startup_video = "K:/Github/PeopleTracker/Evaluation/People/John Scott/GP020002.MP4"
    try:
            # initialize the log settings
        # logging.basicConfig(filename = 'app.log', level = logging.INFO)

        trackerName = 'CSRT'

        #Create QT application for the UI
        app = PyQt5.QtWidgets.QApplication(sys.argv)

        # sets input_dialog as global so we can access it from other functions (For automation)
        global input_dialog

        input_dialog = qt_dialog.App(video_path)
        

        #Get the video path from UI
        videoPath = input_dialog.filename
        
        input_dialog.log("Populating UI")

        if input_dialog.filename is None or input_dialog.filename == "":
            while input_dialog.nothing_loaded:
                 PyQt5.QtCore.QCoreApplication.processEvents()
        
                #Get the video path from UI
        videoPath = input_dialog.filename

                     

        #Given the path, export the metadata and setup the csv for data collection
        metadata = export_meta(videoPath)
        activity_logger = datalogger.DataLogger(videoPath, video_metadata=metadata)
        
        
        #initialize empty list to store trackers
        tracker_list = []

        pred_dict = None
        


        # initialize OpenCV's special multi object tracker
        # input_dialog.add_tab()
        # input_dialog.add_tab_state = False
        # tracker_list.append(MultiTracker(input_dialog.tab_list[0]))
        selected_tracker = 1

        #initialize regions object to store all regions
        regions = Regions(log=input_dialog.log)
        
        #Initialize video, get the first frame and setup the scrollbar to the video length
        cap = cv2.VideoCapture(videoPath)
        
        # Assign original resolution variable
        input_dialog.original_resolution = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        resized_ratio_x = input_dialog.resolution_x / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        resized_ratio_y = input_dialog.resolution_y / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print("WIDTH:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), " HEIGHT:", cv2.CAP_PROP_FRAME_HEIGHT)

        # print(resized_ratio_x, resized_ratio_y)

        input_dialog.set_max_scrollbar(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_dialog.resolution_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_dialog.resolution_y)
        # fvs = FileVideoStream(videoPath).start()
        fvs = Video.STFileVideoStream(videoPath)

        # ret, frame = cap.read()

        frame, frame_num = fvs.read()
        input_dialog.set_scrollbar(0)
        fvs.frame_number = input_dialog.get_scrollbar_value()
        fvs.reset = True
        input_dialog.scrollbar_changed = True

        previous_frame = frame

        frame = cv2.resize(frame, (input_dialog.resolution_x, input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)
        show_frame = frame.copy()

        previous_frame = frame
        #get the video's FPS
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        input_dialog.log("Video FPS set to " + str(vid_fps))
        input_dialog.set_fps_info(vid_fps)
        
        skip_frame = 10
        
        input_dialog.log("Gathering frames...")

        import maskrcnn
        input_dialog.splash.close()

        snap_called = False

        ## STEP1
        while True:

            PyQt5.QtCore.QCoreApplication.processEvents()

            if input_dialog.load_tracked_video_state:
                # import_filename = str(videoPath[:-4]) + ".csv"
                import_filename = input_dialog.openFileNameDialog(task="Select CSV to load Tracker Data", extensions="*.csv")
                loaded_trackers, frame = load_tracker_data(import_filename, input_dialog, frame)
                tracker_list.extend(loaded_trackers)
                input_dialog.load_tracked_video_state = False                

            if input_dialog.del_frame:
                input_dialog.snap_state = "Backward"
                selected_tracker = input_dialog.tabs.currentIndex()
                # input_dialog.tab_list[selected_tracker].read_only = True
                activity_logger.adjustment(frame_number=frame_number, 
                                            from_box=(top_left_x, top_left_y, (top_left_x + width), (top_left_y + height)), 
                                            to_box=None, 
                                            timer_id="DELETE_FRAME",
                                            tracker_id=tracker_list[tracker_num].get_name(),
                                            intervention_type="USER"
                                            )

                if input_dialog.tab_list[selected_tracker].read_only is False:
                    print("RESETTING")
                    input_dialog.tab_list[selected_tracker].toggle_read()
                # delete data dictionary key on active tracker
                # print("Deleting frame number", fvs.frame_number)
                # print(tracker_list[selected_tracker].data_dict.keys())

                if (fvs.frame_number) in tracker_list[selected_tracker].data_dict.keys():
                    del tracker_list[selected_tracker].data_dict[(fvs.frame_number)]

                    input_dialog.del_frame = False
                if (fvs.frame_number-fvs.skip_value) in tracker_list[selected_tracker].data_dict.keys():
                    del tracker_list[selected_tracker].data_dict[(fvs.frame_number-fvs.skip_value)]
                    input_dialog.del_frame = False
                else:
                    input_dialog.log("No track to remove on this frame.")
                    input_dialog.del_frame = False
            # END trigger delete pressed


            # This is needed for activity logger to end pause timers
            if input_dialog.pause_to_play:
                print("Playing...")
                activity_logger.end_pause()
                activity_logger.end_slider(fvs.frame_number, "SLIDER")
                input_dialog.pause_to_play = False
            
            if input_dialog.play_to_pause:
                if tracker_list:
                    activity_logger.paused(frame_num, "USER", "USER_Pause",  tracker_list[selected_tracker].get_name())
                input_dialog.play_to_pause = False

            if input_dialog.predict_state is True:
                #UNCOMMENT BELOW
                # frame, rois, scores = maskrcnn.predict(videoPath, step=input_dialog.skip_frames.value(), display=True, logger=input_dialog.log)
                input_dialog.predict_state = False

            if input_dialog.export_all_state is True:
                input_dialog.export_all_state = False
                for tracker in tracker_list:
                    if tracker.data_dict: # Check if dataframe is empty before exporting.
                        if input_dialog.was_loaded:
                            input_dialog.log("Exporting to new version")
                            tracker.export_data(input_dialog.resolution_x, input_dialog.resolution_y, videoPath, vid_fps, new_version=input_dialog.get_new_data_version())
                            input_dialog.data_version_updated = False
                        else:
                            tracker.export_data(input_dialog.resolution_x, input_dialog.resolution_y, videoPath, vid_fps)

                # When we export everything assume loaded data is all exported so we do not overwrite
                


            if input_dialog.load_predictions_state is True:
                pred_dict = maskrcnn.load_predicted((videoPath[:-4] + "_predict.csv"))
                print(pred_dict)
                input_dialog.load_predictions_state = False
            
            if input_dialog.track_preds_state is True and bool(pred_dict) is True:
                export_filename = str(videoPath[:-4]) + "Predictions_Ids.csv"
                input_dialog.track_preds_state = False
                prediction_dict = maskrcnn.track_predictions(pred_dict, videoPath, preview=True)
                prediction_dict.to_csv(export_filename)
            
            if input_dialog.export_charactoristics:
                print("Exporting")
                export_filename = str(videoPath[:-4])
                text, ok = PyQt5.QtWidgets.QInputDialog.getText(input_dialog, 'Video Location', 'Enter Recorded Location:')
                activity_logger.video_location = text
                activity_logger.get_video_characteristics()
                activity_logger.export_charactoristics(export_filename)
                input_dialog.export_charactoristics = False

            if input_dialog.export_activity:
                export_filename = str(videoPath[:-4])
                activity_logger.export_activity(export_filename)
                input_dialog.export_activity = False

            if input_dialog.quit_State is True:
                # sys.exit(app.exec_())
                # cap.release()
                # fvs.stop()
                # cv2.destroyAllWindows()
                
                input_dialog.log("Cap Release")
                cap.release()
                input_dialog.log("Destroy cv2")
                cv2.destroyAllWindows()
                input_dialog.log("Quitting App")
                
                input_dialog.log("System Exit")
                input_dialog.close()


                app.quit()
                input_dialog.log("Stopping FVS")
                # fvs.stop()
                os._exit(1)
                
            previous_skip = fvs.skip_value
            next_skip = input_dialog.get_frame_skip()
            if fvs.skip_value != input_dialog.get_frame_skip():
                print("Skipbo")
                fvs.reset = True
                # frame = fvs.read()
                fvs.skip_value = input_dialog.get_frame_skip()
                 
            app.processEvents()
            selected_tracker = input_dialog.tabs.currentIndex()

            #if there's no data to export, grey out export button.
            if tracker_list:
                if not tracker_list[selected_tracker].data_dict:
                    input_dialog.export_tab_btn.setEnabled(False)
                else:
                    input_dialog.export_tab_btn.setEnabled(True)
            elif selected_tracker == -1:
                input_dialog.export_tab_btn.setEnabled(False)



            # print(frame_num, skip_frame)
            if input_dialog.snap_state == "Forward":
                fvs.reset = True
                input_dialog.set_scrollbar(input_dialog.get_scrollbar_value() + input_dialog.get_frame_skip())
                fvs.frame_number = input_dialog.get_scrollbar_value() + input_dialog.get_frame_skip()
                
                input_dialog.scrollbar_changed = True
                input_dialog.snap_state = None
                snap_called = True
                fvs.reset = True


            if input_dialog.snap_state == "Backward":
                fvs.reset = True
                if frame_num - input_dialog.get_frame_skip() > 0:
                    input_dialog.set_scrollbar(input_dialog.get_scrollbar_value() - input_dialog.get_frame_skip())
                else:
                    input_dialog.set_scrollbar(0)
                    fvs.frame_number = 0

                input_dialog.scrollbar_changed = True
                input_dialog.snap_state = None
                snap_called = True
                fvs.reset = True
                

            #When at the end, go to the beginning and pause
            if input_dialog.get_scrollbar_value() >= input_dialog.vidScroll.maximum() or (input_dialog.get_scrollbar_value() + skip_frame) >= input_dialog.vidScroll.maximum():
                fvs.frame_number = input_dialog.get_scrollbar_value()
                if not input_dialog.record_live:
                    input_dialog.set_pause_state()
                    input_dialog.set_all_tabs("Read")

                    input_dialog.set_scrollbar(0)
                    activity_logger.paused(fvs.frame_number, "END_VIDEO", "END_VIDEO", None)
                    fvs.reset = True
                input_dialog.scrollbar_changed = True

                    # When we reach the end of the video, actively selected trackers turn to read only (trackerID = None then) and we assign END_VIDEO tag to it.
                
                # activity_logger.end_timer(activity_logger.start_time_ID)
            
            # if input_dialog.get_scrollbar_value() == 0 and fvs.frame_number != input_dialog.get_scrollbar_value():
            #     fvs.frame_number = input_dialog.get_scrollbar_value()
            #     input_dialog.scrollbar_changed = True
                # input_dialog.
                # fvs.frame_number = input_dialog.get_scrollbar_value()

                # fvs.reset = True
                # input_dialog.scrollbar_changed = True
                # input_dialog.set_pause_state()
                # input_dialog.set_all_tabs("Read")
                
            # else:
            #     # input_dialog.set_scrollbar(skip_frame)
                # input_dialog.set_scrollbar(frame_num)
                # print(input_dialog.get_scrollbar_value(), frame_num)
                # frame, frame_num = fvs.read()
                # frame = fvs.read()
                # input_dialog.scrollbar_changed = False

            #When we add a tab, finish initializing it before anything else can continue
            if input_dialog.add_tab_state == True:
                input_dialog.log("Adding Tab!")
                input_dialog.tabs.setCurrentIndex(len(tracker_list))
                selected_tracker = input_dialog.tabs.currentIndex()
                tracker_list.append(MultiTracker(input_dialog.tab_list[selected_tracker]))
                
                input_dialog.tabs.setEnabled(False)
                input_dialog.add_tab_btn.setEnabled(False)
                input_dialog.del_tab_btn.setEnabled(False)
                input_dialog.export_tab_btn.setEnabled(False)
                input_dialog.tabs.setEnabled(True)
                input_dialog.add_tab_state = False

            
            # if input_dialog.snap_state is not None and tracker_list[selected_tracker].data_dict:
            #     #grabs frames that tracker was tracked
            #     # frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            #     frame_number = fvs.frame_number
            #     frames_tracked = tracker_list[selected_tracker].data_dict
            #     closest_frame = frames_tracked.get(frame_number, frames_tracked[min(frames_tracked.keys(), key=lambda k: abs(k-frame_number))])
            #     print(closest_frame)
            #     if frame_number - closest_frame[0][0] > frame_number-closest_frame[0][1]:
            #         closest_frame = closest_frame[0][0]
            #     else:
            #         closest_frame = closest_frame[0][1]

            #     print(type(closest_frame))
            #     input_dialog.log("Closest: " + str(closest_frame))
            #     # cap.set(1 , closest_frame)
            #     input_dialog.play_state = False
            #     input_dialog.snap_state = None
            #if playing, increment frames by skip_frame count

            #if the scrollbar is changed, update the frame, else continue with the normal frame
            if input_dialog.scrollbar_changed == True:
                if tracker_list:
                    activity_logger.slider_moved(frame_num, "SLIDER", tracker_list[selected_tracker].get_name())
                #If Snapping enabled, snap the scrollbar to the nearest multiple of skip_frame
                # print("MOD", (input_dialog.get_scrollbar_value() % input_dialog.get_frame_skip()))
                if input_dialog.get_scrollbar_value() % input_dialog.get_frame_skip() != 0 and input_dialog.snap_to_frame_skip:
                    
                    rounded = round(input_dialog.get_scrollbar_value()/input_dialog.get_frame_skip())*input_dialog.get_frame_skip()
                    # print("Rounding!", rounded)
                    input_dialog.set_scrollbar(rounded)

                # input_dialog.mediaStateChanged()
                input_dialog.play_state = False
                fvs.frame_number = input_dialog.get_scrollbar_value()
                # input_dialog.log("Scrolled.")  
                fvs.reset = True
                frame, frame_num = fvs.read()
                # print(frame_num)
                previous_frame = frame
                
                input_dialog.scrollbar_changed = False

                # segmask, frame = custom_model.segmentFrame(frame,True)

            if input_dialog.play_state == True and not input_dialog.scrollbar_changed:
                
                # fvs.stopped = False
                frame, frame_num = fvs.read()
                previous_frame = frame

                input_dialog.set_scrollbar(frame_num)
                if frame_num != input_dialog.get_scrollbar_value():
                    print("Trying Again:", frame_num, fvs.frame_number, input_dialog.get_scrollbar_value())
                    continue
                # print("Reading Play:", frame_num)
                # previous_frame = frame
                # segmask, frame = custom_model.segmentFrame(frame,True)
                # skip_frame += input_dialog.get_frame_skip()
                
            else:
                # frame = input_dialog.get_scrollbar_value()
                frame = previous_frame
                
            #set tne next frame to skip frame and read it
            # cap.set(1 , skip_frame)
            try:
                # ret, frame = cap.read()
                frame = cv2.resize(frame, (input_dialog.resolution_x, input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)

            # if input_dialog.image_options.brightness.value() != 0:
                # frame = input_dialog.image_options.add_brightness(frame)


                if input_dialog.image_options.roi_normalize_flag:
                    input_dialog.image_options.set_normalized_region(frame)
                    input_dialog.image_options.roi_normalize_flag = False

                # if input_dialog.image_options.get_equalize_clahe_hist() is True:
                #     frame = input_dialog.image_options.enhance_normalized_roi(frame)

                if input_dialog.image_options.get_equalize_hist() is True:
                    frame = input_dialog.image_options.equalize_hist(frame)
                
                if input_dialog.image_options.get_equalize_clahe_hist():
                    frame = input_dialog.image_options.equalize_clahe_hist(frame)

                if input_dialog.image_options.get_alpha() != 10 or input_dialog.image_options.get_beta() != 10: 
                    frame = input_dialog.image_options.enhance_brightness_contrast(frame)
                
                if input_dialog.image_options.get_gamma() != 10:
                    frame = input_dialog.image_options.enhance_gamma(frame)

                # frame = input_dialog.image_options.enhance_brightness(frame)
                # _, frame = input_dialog.image_options.auto_enhance(frame)

                show_frame = frame.copy()
                


            except Exception as e:
                crashlogger.log(str(e))
                input_dialog.snap_state = "Forward"
                print("No resize")
                continue


            #crash the program if no frame exists
            if frame is None:
                break
            
            #Keep tab names up to date
            input_dialog.set_tab_names()

            # E is for Export
            key = cv2.waitKey(1) & 0xFF
            # if key == ord("e") or input_dialog.export_state == True:
            if input_dialog.export_state == True:
                input_dialog.export_state = False
                input_dialog.log("Exporting " + tracker_list[selected_tracker].get_name() + "'s data recorded.")
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
                try:
                    if tracker_list[selected_tracker].data_dict: # Ensure data exists in dictionary before exporting
                        if input_dialog.was_loaded:
                            tracker_list[selected_tracker].export_data(input_dialog.resolution_x, input_dialog.resolution_y, videoPath, vid_fps, new_version=input_dialog.get_new_data_version())
                            input_dialog.data_version_updated = False
                        else:
                            tracker_list[selected_tracker].export_data(input_dialog.resolution_x, input_dialog.resolution_y, videoPath, vid_fps)
                except IOError as err:
                    input_dialog.log(err)
                    input_dialog.show_warning(str(err) + "\n Please close open CSV and try again.")


            #remove the tracker that is currently selected
            # if input_dialog.del_tab_state is True and selected_tracker != -1:
            if input_dialog.del_tab_state is True:
                
                input_dialog.log("Deleting Tracker!")
                del tracker_list[selected_tracker]
                input_dialog.tabs.setCurrentIndex(len(tracker_list))
                selected_tracker = input_dialog.tabs.currentIndex()
                input_dialog.del_tab_state = False
            # elif key == ord("w"):
            #     input_dialog.log("Nudge Up")
            # elif key == ord("a"):
            #     skip_frame -= input_dialog.get_frame_skip() * 2
            #     input_dialog.play_state = True
            #     input_dialog.mediaStateChanged(True)
            # #R is for Radius
            
            if input_dialog.region_state is True:
                input_dialog.log("Adding region... Write name and then draw boundaries")
                regions.add_region(show_frame)
                input_dialog.region_state = False
                input_dialog.log("Adding region complete.")
            if input_dialog.del_region_state is True:
                input_dialog.log("Select a region to remove...")
                regions.del_region()
                input_dialog.del_region_state = False
                input_dialog.log("Removing region complete.")

            # elif key == ord("d"):
            #     input_dialog.log("Nudge Right")
            selected_tracker = input_dialog.tabs.currentIndex()

            #Set the selected Tracker to Red
            for tracker in range(len(tracker_list)):
                app.processEvents()
                if tracker == selected_tracker:
                    tracker_list[tracker].colour = (0,0,255)
                else:
                    tracker_list[tracker].colour = (255,255,255)
            app.processEvents()
            
            #Display all regions on screen if they exist
            if len(regions.region_dict) > 0:
                show_frame = regions.display_region(show_frame)

            #Loop through every tracker and update
            for index, tracker in enumerate(tracker_list):
                tracker_num = index
                tracker = tracker

                frame_number = frame_num
                # try:
                if input_dialog.tab_list[tracker_num].other_room:
                    tracker.record_data(frame_number, input_dialog.num_people.value(), other_room=True)
                    regions.del_moving_region(tracker.get_name(), id=tracker.id())

                # Collect active tracker data
                elif tracker.init_bounding_box is not None and input_dialog.tab_list[tracker_num].active is True and input_dialog.tab_list[tracker_num].read_only is False:
                    
                    #allocate frames on GPU, reducing CPU load.
                    cv2.UMat(frame)    

                    app.processEvents()
                    #track and draw box on the frame
                    success, box, show_frame = tracker.update_tracker(show_frame)
                    tracker.box = box
                    app.processEvents()
                    
                    #NOTE: this can be activated if you want to pause the program when trakcer fails
                    # if not success:
                    #     tracker.assign(frame, trackerName)

                    #caluclate info needed this frame
                    top_left_x = box[0]
                    top_left_y = box[1]
                    width = box[2]
                    height = box[3]


                    center_x = top_left_x + (width/2)
                    center_y = top_left_y + (height/2)

                    # cv2.circle(frame, (box[0],box[1]), 10, (0,0,255))
                    if tracker.is_region() is True and tracker.get_name().strip() == "":
                        input_dialog.tab_list[tracker_num].getText(input_name="Region Name:",line=input_dialog.tab_list[tracker_num].name_line)

                    if tracker.is_region() is True and tracker.get_name() != "":

                        regions.set_moving_region(name = tracker.get_name(), 
                                                point = (int(center_x - width), int(center_y - height)),
                                                dimensions = (int(width*2), int(height*2)),
                                                id=tracker.id()
                                                )
                        regions.display_region(show_frame)

                    

                    elif tracker.is_region() is False:
                        # If tracker region is no longer selected, delete moving region
                        regions.del_moving_region(tracker.get_name(), id=tracker.id())

                    if pred_dict and input_dialog.mcrnn_options.get_active() is True:
                        if input_dialog.get_scrollbar_value() in pred_dict.keys():
                            # print("GETTING IOU")
                            iou, show_frame = maskrcnn.compute_iou(
                                    box=(top_left_x, top_left_y, top_left_x + width, top_left_y + height), 
                                    boxes=pred_dict[input_dialog.get_scrollbar_value()][0], 
                                    boxes_area=pred_dict[input_dialog.get_scrollbar_value()][1], 
                                    ratios=(resized_ratio_x, resized_ratio_y),
                                    frame=frame
                                )
                            # print("IOUs", iou)

                            closest = max(iou)
                            index = iou.index(closest)
                            box = pred_dict[input_dialog.get_scrollbar_value()][0][index]
                            p1 = (int(box[0]*resized_ratio_x), int(box[1]*resized_ratio_y))
                            p2 = (int(box[2]*resized_ratio_x), int(box[3]*resized_ratio_y))

                            # if closest < 0.45:
                            if closest <= input_dialog.mcrnn_options.get_min_value():
                                # print("Out of range")
                                show_frame = cv2.rectangle(frame, p1, p2, (150, 150, 220), 3)
                                input_dialog.set_pause_state()
                                activity_logger.paused(frame_number, "MRCNN", "MRCNN_Pause", tracker_list[tracker_num].get_name())

                            elif closest >= input_dialog.mcrnn_options.get_auto_assign():
                                show_frame = cv2.rectangle(frame, p1, p2, (50, 200, 50), 3)
                                if tracker.auto_assign_state:
                                    # print("Auto Assigning")

                                    if input_dialog.play_state is True:
                                        # Only record auto assignment while video is playing
                                        activity_logger.adjustment(frame_number=frame_number, 
                                                                    from_box=(top_left_x, top_left_y, (top_left_x + width), (top_left_y + height)), 
                                                                    to_box=(p1[0], p1[1], p2[0], p2[1]), 
                                                                    timer_id="MRCNN_Adjust",
                                                                    tracker_id=tracker_list[tracker_num].get_name(),
                                                                    intervention_type="MRCNN"
                                                                    )

                                    tracker.auto_assign(frame, (p1[0], p1[1], p2[0], p2[1]))

                            for pred_index, pred in enumerate(iou):
                                diff = abs(pred - closest)

                                if pred != closest and diff <= input_dialog.mcrnn_options.get_similarity():
                                    # input_dialog.log("Possible ID Switch!")
                                    activity_logger.paused(frame_num, "MRCNN", "MRCNN_Adjust", tracker_list[tracker_num].get_name())
                                    pred_box = pred_dict[input_dialog.get_scrollbar_value()][0][pred_index]
                                    pred_p1 = (int(pred_box[0]*resized_ratio_x), int(pred_box[1]*resized_ratio_y))
                                    pred_p2 = (int(pred_box[2]*resized_ratio_x), int(pred_box[3]*resized_ratio_y))
                                    show_frame = cv2.rectangle(frame, pred_p1, pred_p2, (255, 0, 255), 1)
                                    show_frame = cv2.rectangle(frame, p1, p2, (50, 200, 50), 3)
                                    input_dialog.set_pause_state()
                                    # .index(closest)

                    #center dot               
                    cv2.circle(show_frame, (int(center_x),int(center_y)),1,(0,255,0),-1)

                    # top = (int(center_x), int(center_y + height/2))
                    # bottom = (int(center_x), int(center_y - height/2))
                    # cv2.circle(frame, top, 3, (0,255,255),-1)
                    # cv2.circle(frame, bottom, 3, (0,255,255),-1)
                    in_region, p = regions.test_region((center_x, center_y))
                    
                    # Defines when to record while play is active
                    play_active = input_dialog.play_state == True and input_dialog.tab_list[tracker_num].read_only is False
                    paused_snap_active = input_dialog.play_state == False and input_dialog.tab_list[tracker_num].read_only is False and snap_called == True
                    
                    # if record live always allow recording of data (cannot pause)
                    if input_dialog.record_live:
                        play_active = True
                        paused_snap_active = True

                    if play_active or paused_snap_active:
                        # tracker.grab_cut(frame,box)
                        #record all the data collected from that frame
                        # print("Recording data")

                        pred_line = tracker.predictor.predict((center_x,center_y))

                        pred_line = tracker.predictor.predict()
                        pred_line = tracker.predictor.predict()

                        box_pred_p1 = tracker.box_predictor[0].predict((top_left_x,top_left_y))
                        box_pred_p2 = tracker.box_predictor[1].predict(((top_left_x + width), (top_left_y + height)))

                        for i in range(input_dialog.get_frame_skip()):
                            box_pred_p1 = tracker.box_predictor[0].predict()
                            box_pred_p2 = tracker.box_predictor[1].predict()

                        pred_p1 = (int((box_pred_p1[0] + box_pred_p1[2])[0]), int((box_pred_p1[1] + box_pred_p1[3])[0]) )
                        pred_p2 = (int((box_pred_p2[0] + box_pred_p2[2])[0]), int((box_pred_p2[1] + box_pred_p2[3])[0]) )

                        pred_centroid = (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) )

                        tracker.predicted_bbox = (pred_p1[0],pred_p1[1], pred_p2[0],pred_p2[1])
                        tracker.predicted_centroid = (pred_centroid)
                        try:
                            if input_dialog.predictor_options.get_active_iou():
                                # activity_logger.intervention = datalogger.HUMAN_INTERVENTION_KALMAN
                                predicted_bbox_iou, _ = maskrcnn.compute_iou(
                                                    box=tracker.predicted_bbox,
                                                    boxes=[(top_left_x,top_left_y,(top_left_x + width), (top_left_y + height) )],
                                                    boxes_area=[(pred_p2[0] - pred_p1[0]), (pred_p2[1] - pred_p1[1]) ]
                                                )
                                cv2.rectangle(show_frame, pred_p1, pred_p2, (0,255,0), 1)
                                # print("P_IOU: ", predicted_bbox_iou, end=" | ")

                                if predicted_bbox_iou[0] <= input_dialog.predictor_options.get_min_IOU() and predicted_bbox_iou[0] > 0:
                                    print("Pausing")
                                    input_dialog.set_pause_state()
                                    activity_logger.paused(frame_number, "KALMAN", "KALMAN_Pause", tracker_list[tracker_num].get_name())

                                    
                            

                        except:
                            print("Cannot Compute IOU")
                        try:
                            if input_dialog.predictor_options.get_active_centroid():
                                pred_dist = regression.distance_2d((center_x,center_y), (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) ))
                                # print("POINT_DIST: ", pred_dist)
                                cv2.arrowedLine(show_frame, (int(pred_line[0]),int(pred_line[1])), (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) ),  (0,0,255), 3, tipLength=1)
                                if pred_dist >= input_dialog.predictor_options.get_min_distance():
                                    # print("Pausing")
                                    input_dialog.set_pause_state()
                                    activity_logger.paused(frame_number, "REGRESSION", "REGRESSION_Pause", tracker_list[tracker_num].get_name())
                        except:
                            print("Cannot Compute Distance")
                        

                        

                        if tracker.regression:
                            slope, intercept, correlation = tracker.regression.predict((center_x,center_y))
                            if slope is not None and correlation is not None and abs(correlation) >= 0.5:
                                # y = Ax + b, therefore x = (y - b) / A
                                try:
                                    #If moving up and right
                                    # print(tracker.regression.get_direction())
                                    # if tracker.regression.get_direction()[0] and tracker.regression.get_direction()[1]:
                                    # print(slope)
                                    if slope == 0:
                                        cv2.line(show_frame, (0,int(center_y)), (800,int(center_y)), (0,0,255), 1)
                                    else:
                                        startY = 0
                                        endY = 800
                                        startX = (startY - intercept) / slope
                                        endX = (endY - intercept) / slope

                                        cv2.arrowedLine(show_frame, (startX,startY), (endX,endY), (0,0,255), 1)
                                        # else:
                                        #     startY = int(center_y)
                                            #     endY = int(center_y) + 10
                                
                                                # print(min_point, max_point)
                                            # if tracker.regression.get_direction():
                                            # else:
                                            #     cv2.arrowedLine(frame, (startX,startY), (endX,endY), (0,0,255), 1)
                                        
                                except ZeroDivisionError as e:
                                    cv2.line(show_frame, (0,int(center_y)), (800,int(center_y)), (0,0,255), 1)
                                except:
                                    cv2.line(show_frame, (int(center_x),0), (int(center_x),800), (0,0,255), 1)
                                # cv2.line(frame, (int(center_x-30),int(center_y)), (int(center_x + 30),int(center_y)), (0,0,255), 1)
                        # print( ((pred_line[0] + pred_line[2])[0], (pred_line[1] + pred_line[3])[0]) )      
                        if input_dialog.record_live:
                            tracker.record_data(fvs.frame_number, input_dialog.num_people.value(), center_x, center_y, width, height, in_region, image_frame=show_frame)
                        else:
                            tracker.record_data(input_dialog.get_scrollbar_value(), input_dialog.num_people.value(), center_x, center_y, width, height, in_region)
            
                # except Exception as e:
                #     crashlogger.log(str(e))
                #     # print("No resize")
                    
                #     input_dialog.log("Crashed while deleting. Continuing")
                #     continue

                try:
                    if input_dialog.tab_list[tracker_num].read_only is True:
                        # print("Displaying read only")
                        #if read only, display the center
                        # frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        # print(frame_number, fvs.frame_number)
                        # frame_number = frame_num
                        # print(frame_number)
                        frame_number = input_dialog.get_scrollbar_value()
                        # regions.del_moving_region(tracker.get_name()) 
                        # dendregion
                        # print(selected_tracker, tracker_num)
                        if frame_number in tracker.data_dict:
                            # print("Exists")
                            # If key exists in data
                            # point, regions, dimensions, other_room, total_people
                            center, _, dim, other_room, total_people, is_chair = tracker.data_dict[frame_number]
                            # print(frame_number, center)
                            if tracker.is_region() is True and tracker.get_name() != "":
                                
                                point = (int(center[0] - dim[0]), int(center[1] - dim[1]))
                                dim = (int(dim[0]*2), int(dim[1]*2))
                                regions.set_moving_region(tracker.get_name(), point, dim)
                                # top = (int(center[0]) - dim[0], int(center[1], - dim[1]/2))
                                # bottom = (int(center[0]) - dim[0], int(center[1], + dim[1]/2))

                            if tracker.is_region() is False:
                                # If tracker region is no longer selected, delete moving region
                                regions.del_moving_region(tracker.get_name(), id=tracker.id())

                            
                            if selected_tracker == tracker_num:
                                # print("Green")
                                #center dot
                                
                                cv2.circle(show_frame, (int(center[0]),int(center[1])),2,(0,255,0),-1)
                                # print("Green")
                                # cv2.circle(frame, top, 3, (0,255,0),-1)
                                # cv2.circle(frame, bottom, 3, (0,255,0),-1)
                                
                            else: 
                                # print("Red Dot")
                                cv2.circle(show_frame, (int(center[0]),int(center[1])),2,(0,0,255),-1)
                                # cv2.circle(frame, top, 3, (0,0,255),-1)
                                # cv2.circle(frame, bottom, 3, (0,0,255),-1)
                        
                        #Exclude if you want regions to not exist
                        elif not input_dialog.retain_region:
                            regions.del_moving_region(tracker.get_name(), id=tracker.id())

                except Exception as e:
                    crashlogger.log(str(e))
                    input_dialog.log("Could not handle read only. List index out of range, Continuing")


                    
                app.processEvents()

            # Snap called is only false after done processing the current frame
            snap_called = False


            #If you select a tracker and it is not running, start a new one
            if selected_tracker >= 0 and len(tracker_list) > 0 and selected_tracker <= len(tracker_list):
                #If there is no assigned trakcer on selected individual, start one and not allow action until done
                if tracker_list[selected_tracker].init_bounding_box is None:
                    input_dialog.tabs.setEnabled(False)
                    #Fix no-Square created issue
                    create_success = False
                    while create_success is False:
                        try:
                            tracker_list[selected_tracker].create(trackerName)
                            tracker_list[selected_tracker].assign(frame, trackerName) #Breaks if create was not sucessful
                            create_success = True
                        except Exception as e:
                            crashlogger.log(str(e))
                            input_dialog.log("Could not create Tracker, Please Draw and select (Space) a rectangle")

                    
                    input_dialog.tabs.setEnabled(True)
                    input_dialog.add_tab_btn.setEnabled(True)
                    input_dialog.del_tab_btn.setEnabled(True)
                    input_dialog.export_tab_btn.setEnabled(True)

                #Press space bar to re-assign
                if input_dialog.set_tracker_state is True:
                    try:
                        
                        input_dialog.play_state = False
                        input_dialog.tabs.setEnabled(False)
                        
                        activity_logger.adjustment(frame_number, tracker_list[selected_tracker].box, "USER_Adjust", tracker_list[selected_tracker].get_name())
                        tracker_list[selected_tracker].assign(frame, trackerName)
                        activity_logger.end_adjustment(tracker_list[selected_tracker].box, "USER_Adjust")

                        input_dialog.tabs.setEnabled(True)
                        input_dialog.set_tracker_state = False
                    except Exception as e:
                        crashlogger.log(str(e))
                        input_dialog.log("Could not assign tracker, try again")
                        input_dialog.tabs.setEnabled(True)
                        input_dialog.set_tracker_state = False
                        
            #UNCOMMENT BELOW
            if pred_dict is not None and input_dialog.mcrnn_options.get_active():
                show_frame = maskrcnn.display_preds(show_frame, input_dialog.get_scrollbar_value(), pred_dict, (resized_ratio_x,resized_ratio_y))
                # input_dialog.play_state = True

            #NOTE this is in try-catch because initially there are not enough frames to calculate time. 
            #This could be done with if statement, though I havent found a way...
            try:
                current_tracked_time = tracker_list[selected_tracker].get_time_tracked(vid_fps)[0] + tracker_list[selected_tracker].previous_time
                input_dialog.tab_list[selected_tracker].update_length_tracked(current_tracked_time)
            except Exception as e:
                # crashlogger.log(str(e))
                pass
            


            # print("FRAMES", frame_num, fvs.frame_number, input_dialog.get_scrollbar_value())
            #When done processing each tracker, view the frame
            cv2.imshow("Frame", show_frame)
        
            # input_dialog.videoWindow.show_image(frame)
    except Exception as e:
        print(traceback.format_exc())
        crashlogger.log(str(traceback.format_exc()))
         

class Monkerunner():
    '''
    A class that contains all the methods and variables to run all other components together.
    The main driver to organize the order that things run and some of the methods to iterate through trackers and interact with the UI.
    '''
    def __init__(self, video_path, num_pool_threads=5) -> None:
        
        ## The default type of tracker is CSRT @cite csrt
        self.trackerName = 'CSRT'

        #Create QT application for the UI
        self.app = PyQt5.QtWidgets.QApplication(sys.argv)

        # sets input_dialog as global so we can access it from other functions (For automation)
        # self.input_dialog
        
        ## Defines the gobal UI that the application uses
        self.input_dialog = qt_dialog.App(video_path)
        

        #G#et the video path from UI
        self.videoPath = self.input_dialog.filename


        self.input_dialog.log("Populating UI")

        # init event process so that we can open the screen
        if self.input_dialog.filename is None or self.input_dialog.filename == "":
            while input_dialog.nothing_loaded:
                 PyQt5.QtCore.QCoreApplication.processEvents()
        
        ## Get the video path from UI
        self.videoPath = self.input_dialog.filename

        ## Given the path, export the metadata and setup the csv for data collection
        self.metadata = export_meta(self.videoPath)

        ## The object that records user activity and video characteristics
        self.activity_logger = datalogger.DataLogger(self.videoPath, video_metadata=self.metadata)
        
        ## This is a list that contains all the trackers. Use this to access trackers when created
        self.tracker_list = []

        ## Creates a list with a maximum number of trackers containing pre-assigned threads 
        self.thread_pool = multiprocessing.Pool(processes=num_pool_threads)

        ## Contains the maskrcnn prediction data. This dictionary has frame keys and each key contains the list of bounding boxes
        self.pred_dict = None
        
        # initialize OpenCV's special multi object tracker
        # input_dialog.add_tab()
        # input_dialog.add_tab_state = False
        # tracker_list.append(MultiTracker(input_dialog.tab_list[0]))

        ## The index that is the selected tracker through the UI.
        self.selected_tracker = 1

        ## Contains all of the regions that are generated.
        self.regions = Regions(log=self.input_dialog.log)
        
        ## Initialize video, get the first frame and setup the scrollbar to the video length
        self.cap = cv2.VideoCapture(self.videoPath)
        
        ## Assign original resolution variable
        self.input_dialog.original_resolution = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ## Resized ratio is updated when the user resizes the video from the original resolution
        self.resized_ratio_x = self.input_dialog.resolution_x / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ## Resized ratio is updated when the user resizes the video from the original resolution
        self.resized_ratio_y = self.input_dialog.resolution_y / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # print("WIDTH:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), " HEIGHT:", cv2.CAP_PROP_FRAME_HEIGHT)

        # print(resized_ratio_x, resized_ratio_y)

        # sets video length and resizes to resolution specified in input_dialog
        self.input_dialog.set_max_scrollbar(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_dialog.resolution_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_dialog.resolution_y)
        # fvs = FileVideoStream(videoPath).start()
        self.fvs = Video.STFileVideoStream(self.videoPath)

        # ret, frame = cap.read()

        self.frame, self.frame_num = self.fvs.read()
        self.input_dialog.set_scrollbar(0)
        self.vs.frame_number = self.input_dialog.get_scrollbar_value()
        self.fvs.reset = True
        self.input_dialog.scrollbar_changed = True

        self.previous_frame = self.frame

        self.frame = cv2.resize(self.frame, (self.input_dialog.resolution_x, self.input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)
        self.show_frame = self.frame.copy()

        self.previous_frame = self.frame
        #get the video's FPS
        self.vid_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.input_dialog.log("Video FPS set to " + str(self.vid_fps))
        self.input_dialog.set_fps_info(self.vid_fps)
        
        self.skip_frame = 10
        
        self.input_dialog.log("Gathering frames...")

        import maskrcnn

        self.input_dialog.splash.close()

        self.snap_called = False
    
        ## STEP1

    def threaded_process(self):

        arg_list = []
        for index, tracker in enumerate(self.tracker_list):
            arg_list.append((tracker, index))
        
        results = self.thread_pool.starmap(self.process_tracker, arg_list)
        


    ## Starts the process loop.
    def start_loop(self):
        '''
        This loop contains all the functions for the program.

        It loops through testing any if the triggers if activated.
        It applies any image options such as resize or visual changes from image options
        It activates the tracker and region drawing functions
        Finally it processes the trackers and updates them with the new locations and statistical filters

        It applies all changes to a new visual frame and repeats.
        '''
        try:
            while True:
                
                PyQt5.QtCore.QCoreApplication.processEvents()
                
                #tests every command that is activated through the UI
                self.test_triggers()

                continued = self.continue_reading()
                if continued:
                    continue

                # Start processing
                ret = self.apply_image_options()
                if not ret:
                    continue
                
                #crash the program if no frame exists
                if self.frame is None:
                    break

                #Keep tab names up to date
                self.input_dialog.set_tab_names()

                # E is for Export
                self.key = cv2.waitKey(1) & 0xFF

                self.trigger_export_current_tracker()
                self.trigger_delete_tab()
                self.trigger_region()
                self.trigger_delete_region()

                #update selected tracker after we are done adjusting (hopefully no more)
                self.selected_tracker = self.input_dialog.tabs.currentIndex()
                self.draw_trackers()
                self.app.processEvents()
                self.draw_regions()

                # here we start to iterate through all all the trackers and 
                # for index, tracker in enumerate(self.tracker_list):
                #     self.process_tracker(tracker, index)
                self.threaded_process()

                cv2.imshow("Frame", self.show_frame)
                
        except Exception as e:
            print(traceback.format_exc())
            crashlogger.log(str(traceback.format_exc()))

    def test_triggers(self):
        '''
        Test triggers tests many of the videos and applies any of the functions that are triggered.

        loads video
        backspace/ undo
        play
        pause
        start mask rcnn prediction
        export all trackers
        load predictions (mask rcnn)
        tracks predictions using nearest neighbours
        export video characteristis
        export user activity
        quit program
        updates video when frame skips
        disables export when no data is available
        snap forward skip_frame amount
        snap backward skip_frame amount
        resets when the end of the video is reached
        adds a new tracker tab
        updates video when the scroll bar is changed
        '''

        self.trigger_load_video()
        self.trigger_delete_pressed()
        self.trigger_play()
        self.trigger_pause()
        self.trigger_predict()
        self.trigger_export_all()
        self.trigger_load_predictions()
        self.trigger_track_predictions()
        self.trigger_characteristics()
        self.trigger_export_activity()
        self.trigger_quit()

        ## NOTE do we need these? no
        # self.previous_skip = fvs.skip_value
        # self.next_skip = input_dialog.get_frame_skip()


        ## We update UI on non-triggering events such as adjusting or changing values
        self.update_skip_frame()

        #process events
        self.app.processEvents()

        #update selected tracker
        self.selected_tracker = self.input_dialog.tabs.currentIndex()
        self.no_data_disable_export()
        self.triggered_snap_forward()
        self.triggered_snap_backward()
        self.check_end_video()
        
        self.triggered_add_tab()

        self.trigger_scrollbar_change()

    def process_tracker(self, tracker, tracker_number):
        '''
        Processes each tracker individually.

        Here we grab the current tracker's information and draw that information. The tracker is also updated to a new position
        We apply all statistical filters along side tracker's position including iou tests and distance measures to assist user in labeling
        Statistical filters include 
        -   kalman center
        -   kalman box
        -   maskrcnn box
        -   linear regression

        Kalman and regression are only measured while trackers are not in read only mode. Data is then recorded.

        Read only is displayed as a dot (green active, red inactive)
        trackers auto init manual assignment when they have no current cv2 trackers or initial bounding boxes.

        mask rcnn predictions are diplayed

        attempt to update total tracked time

        '''

        if self.input_dialog.tab_list[tracker_number].other_room:
            tracker.record_data(self.frame_number, self.input_dialog.num_people.value(), other_room=True)
            self.regions.del_moving_region(tracker.get_name(), id=tracker.id())

        # Collect active tracker data
        elif tracker.init_bounding_box is not None and self.input_dialog.tab_list[tracker_number].active is True and self.input_dialog.tab_list[tracker_number].read_only is False:
            
            #allocate frames on GPU, reducing CPU load.
            cv2.UMat(self.frame)    

            self.app.processEvents()
            #track and draw box on the frame
            success, box, show_frame = tracker.update_tracker(show_frame)
            tracker.box = box
            self.app.processEvents()
            
            #NOTE: this can be activated if you want to pause the program when trakcer fails
            # if not success:
            #     tracker.assign(frame, trackerName)

            #caluclate info needed this frame
            top_left_x = box[0]
            top_left_y = box[1]
            bb_width = box[2]
            bb_height = box[3]


            center_x = top_left_x + (bb_width/2)
            center_y = top_left_y + (bb_height/2)

            self.draw_tracker_region(tracker, tracker_number)
            self.maskrcnn_predictions(tracker, tracker_number, box)
            #center dot               
            cv2.circle(self.show_frame, (int(center_x),int(center_y)),1,(0,255,0),-1)

            #test if tracker is in any of the regions
            in_region, p = self.regions.test_region((center_x, center_y))

            # Defines when to record while play is active
            play_active = self.input_dialog.play_state == True and self.input_dialog.tab_list[tracker_number].read_only is False
            paused_snap_active = self.input_dialog.play_state == False and self.input_dialog.tab_list[tracker_number].read_only is False and self.snap_called == True
                    
            #adjust settings for live recording
            if self.input_dialog.record_live:
                play_active = True
                paused_snap_active = True

            #While play is active, apply active prediction filters
            if play_active or paused_snap_active:
                self.kalman_filter_predict(tracker, tracker_number, box)
                self.regression_predict(tracker, box)

                # we apply record_live in here since this shares a common attribute of needing to be in play state to work.
                if self.input_dialog.record_live:
                    tracker.record_data(self.fvs.frame_number, self.input_dialog.num_people.value(), center_x, center_y, bb_width, bb_height, in_region, image_frame=self.show_frame)
                else:
                    tracker.record_data(self.input_dialog.get_scrollbar_value(), self.input_dialog.num_people.value(), center_x, center_y, bb_width, bb_height, in_region)

            self.display_read_only(tracker, tracker_number)
            self.app.processEvents()

        # Snap called is only false after done processing the current frame
        self.snap_called = False
        # Make sure all values are valid before changing trackers
        if self.selected_tracker >= 0 and len(self.tracker_list) > 0 and self.selected_tracker <= len(self.tracker_list):
            self.handle_empty_tracker()
            self.trigger_assign_tracker()

        #NOTE move higher up to be with mask_rcnn?
        if self.pred_dict is not None and self.input_dialog.mcrnn_options.get_active():
            self.show_frame = self.maskrcnn.display_preds(self.show_frame, self.input_dialog.get_scrollbar_value(), self.pred_dict, (self.resized_ratio_x,self.resized_ratio_y))            

        #NOTE this is in try-catch because initially there are not enough frames to calculate time. 
        #This could be done with if statement, though I havent found a way...
        try:
            current_tracked_time = self.tracker_list[self.selected_tracker].get_time_tracked(self.vid_fps)[0] + self.tracker_list[self.selected_tracker].previous_time
            self.input_dialog.tab_list[self.selected_tracker].update_length_tracked(current_tracked_time)
        except Exception as e:
            # crashlogger.log(str(e))
            pass
    
        
        
        
    # If tracker is empty we start a new one
    def handle_empty_tracker(self):
        '''
        If you select a tracker and it is not running, start a new one
        If there is no assigned tracker on selected individual, start one and not allow action until done
        '''
        #If you select a tracker and it is not running, start a new one
        #If there is no assigned trakcer on selected individual, start one and not allow action until done
        if self.tracker_list[self.selected_tracker].init_bounding_box is None:
            self.input_dialog.tabs.setEnabled(False)
            #Fix no-Square created issue
            create_success = False
            while create_success is False:
                try:
                    self.tracker_list[self.selected_tracker].create(self.trackerName)
                    self.tracker_list[self.selected_tracker].assign(self.frame, self.trackerName) #Breaks if create was not sucessful
                    create_success = True
                except Exception as e:
                    crashlogger.log(str(e))
                    self.input_dialog.log("Could not create Tracker, Please Draw and select (Space) a rectangle")

            
            self.input_dialog.tabs.setEnabled(True)
            self.input_dialog.add_tab_btn.setEnabled(True)
            self.input_dialog.del_tab_btn.setEnabled(True)
            self.input_dialog.export_tab_btn.setEnabled(True)

    ## Assigns a new tracker when triggered
    def trigger_assign_tracker(self):
        '''!
        A new tracker assignment will pause the video, and make the user draw a new initial bounding box.

        @note an error will be thrown and the user will need to re-try to define a box until one is valid. An example of an invalid box is a box with no size

        '''
        #Press space bar to re-assign
        if self.input_dialog.set_tracker_state is True:
            try:
                
                self.input_dialog.play_state = False
                self.input_dialog.tabs.setEnabled(False)
                
                self.activity_logger.adjustment(self.frame_number, self.tracker_list[self.selected_tracker].box, "USER_Adjust", self.tracker_list[self.selected_tracker].get_name())
                self.tracker_list[self.selected_tracker].assign(self.frame, self.trackerName)
                self.activity_logger.end_adjustment(self.tracker_list[self.selected_tracker].box, "USER_Adjust")

                self.input_dialog.tabs.setEnabled(True)
                self.input_dialog.set_tracker_state = False
            except Exception as e:
                crashlogger.log(str(e))
                self.input_dialog.log("Could not assign tracker, try again")
                self.input_dialog.tabs.setEnabled(True)
                self.input_dialog.set_tracker_state = False
                        
    ## Displays centroid dots, red on inactive and green on selected trackers. Data must be present.
    def display_read_only(self, tracker, tracker_number):
        '''!
        Display the read_only tracks. 

        When read only is active trackers will no longer collect data, and their previous data is shown in forms of green and red dots.
        These dots are green when it is the active selected tracker tab, red otherwise. 

        @note These read only tabs will not have bounding boxes applied to them to demonstrate that no tracker is currently being recorded.
        '''
        try:
            if self.input_dialog.tab_list[tracker_number].read_only is True:

                #if read only, display the center
                frame_number = self.input_dialog.get_scrollbar_value()

                if frame_number in tracker.data_dict:
                    # If key exists in data
                    center, _, dim, other_room, total_people, is_chair = tracker.data_dict[frame_number]
                    
                    if tracker.is_region() is True and tracker.get_name() != "":
                        
                        point = (int(center[0] - dim[0]), int(center[1] - dim[1]))
                        dim = (int(dim[0]*2), int(dim[1]*2))
                        self.regions.set_moving_region(tracker.get_name(), point, dim)
                        # top = (int(center[0]) - dim[0], int(center[1], - dim[1]/2))
                        # bottom = (int(center[0]) - dim[0], int(center[1], + dim[1]/2))

                     # If tracker region is no longer selected, delete moving region
                    if tracker.is_region() is False:
                        self.regions.del_moving_region(tracker.get_name(), id=tracker.id())

                    #active tracker, center dot, green by default
                    if self.selected_tracker == tracker_number:
                        cv2.circle(self.show_frame, (int(center[0]),int(center[1])),2,(0,255,0),-1)
                    
                    #inactive tracker, red by default
                    else: 
                        cv2.circle(self.show_frame, (int(center[0]),int(center[1])),2,(0,0,255),-1)
                
                #Exclude if you want regions to not exist
                elif not self.input_dialog.retain_region:
                    self.regions.del_moving_region(tracker.get_name(), id=tracker.id())

        except Exception as e:
            crashlogger.log(str(e))
            self.input_dialog.log("Could not handle read only. List index out of range, Continuing")

    ## Rolling window linear regression. No pause is applied   
    def regression_predict(self, tracker, box):
        '''!
        Regression calculates the moving window linear regression of the movement. 

        Regression will draw an arrow along the direction of movement.
        The direction is defined by the newest point that has been recorded compared to the first point.

        '''
        if tracker.regression:
            center_x, center_y = self.box_centers(box)
            # top_left_x, top_left_y, width, height = self.box_to_xywh(box)
            slope, intercept, correlation = tracker.regression.predict((center_x,center_y))
            if slope is not None and correlation is not None and abs(correlation) >= 0.5:
                # y = Ax + b, therefore x = (y - b) / A
                try:
                    if slope == 0:
                        cv2.line(self.show_frame, (0,int(center_y)), (800,int(center_y)), (0,0,255), 1)
                    else:
                        startY = 0
                        endY = 800
                        startX = (startY - intercept) / slope
                        endX = (endY - intercept) / slope

                        cv2.arrowedLine(self.show_frame, (startX,startY), (endX,endY), (0,0,255), 1)
                        
                except ZeroDivisionError as e:
                    cv2.line(self.show_frame, (0,int(center_y)), (800,int(center_y)), (0,0,255), 1)
                except:
                    cv2.line(self.show_frame, (int(center_x),0), (int(center_x),800), (0,0,255), 1)

    ## Kalman filter on the box and centroid, pauses if iou or centroid are too different.
    def kalman_filter_predict(self, tracker, tracker_number, box):
        '''!
        Kalman filter uses 2 different difference measures.
        We apply and IoU (intersection/union) measure to tell if the current box is near the predicted box
        We also apply a centroid kalman filter on the center of the box, and we test the center distance with another defined threshold

        If either of these thresholds are insufficient in similarity, we pause the program.

        @note the bounding box measurment takes into account both momentum in scale and shift, while centroild is less sensitive to scale, and more to shift.
        

        '''
        #calculate values used for measuring
        center_x, center_y = self.box_centers(box)
        top_left_x, top_left_y, width, height = self.box_to_xywh(box)
        pred_line = tracker.predictor.predict((center_x,center_y))

        pred_line = tracker.predictor.predict()
        pred_line = tracker.predictor.predict()

        box_pred_p1 = tracker.box_predictor[0].predict((top_left_x,top_left_y))
        box_pred_p2 = tracker.box_predictor[1].predict(((top_left_x + width), (top_left_y + height)))

        for i in range(self.input_dialog.get_frame_skip()):
            box_pred_p1 = tracker.box_predictor[0].predict()
            box_pred_p2 = tracker.box_predictor[1].predict()

        pred_p1 = (int((box_pred_p1[0] + box_pred_p1[2])[0]), int((box_pred_p1[1] + box_pred_p1[3])[0]) )
        pred_p2 = (int((box_pred_p2[0] + box_pred_p2[2])[0]), int((box_pred_p2[1] + box_pred_p2[3])[0]) )

        pred_centroid = (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) )
        
        #save these values to tracker's instance
        tracker.predicted_bbox = (pred_p1[0],pred_p1[1], pred_p2[0],pred_p2[1])
        tracker.predicted_centroid = (pred_centroid)

        #test distance of box IoU
        try:
            if self.input_dialog.predictor_options.get_active_iou():
                # activity_logger.intervention = datalogger.HUMAN_INTERVENTION_KALMAN
                predicted_bbox_iou, _ = self.maskrcnn.compute_iou(
                                    box=tracker.predicted_bbox,
                                    boxes=[(top_left_x,top_left_y,(top_left_x + width), (top_left_y + height) )],
                                    boxes_area=[(pred_p2[0] - pred_p1[0]), (pred_p2[1] - pred_p1[1]) ]
                                )
                cv2.rectangle(self.show_frame, pred_p1, pred_p2, (0,255,0), 1)
                # print("P_IOU: ", predicted_bbox_iou, end=" | ")

                if predicted_bbox_iou[0] <= self.input_dialog.predictor_options.get_min_IOU() and predicted_bbox_iou[0] > 0:
                    print("Pausing")
                    self.input_dialog.set_pause_state()
                    self.activity_logger.paused(self.frame_number, "KALMAN_IOU", "KALMAN_Pause", self.tracker_list[tracker_number].get_name())

        except:
            print("Cannot Compute IOU")
        
        #test dinstance of predicted centers
        try:
            if self.input_dialog.predictor_options.get_active_centroid():
                pred_dist = regression.distance_2d((center_x,center_y), (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) ))
                # print("POINT_DIST: ", pred_dist)
                cv2.arrowedLine(self.show_frame, (int(pred_line[0]),int(pred_line[1])), (int((pred_line[0] + pred_line[2])[0]), int((pred_line[1] + pred_line[3])[0]) ),  (0,0,255), 3, tipLength=1)
                if pred_dist >= self.input_dialog.predictor_options.get_min_distance():
                    # print("Pausing")
                    self.input_dialog.set_pause_state()
                    self.activity_logger.paused(self.frame_number, "KALMAN_CENTER", "KALMAN_Pause", self.tracker_list[tracker_number].get_name())
        except:
            print("Cannot Compute Distance")

    ## A box into xywh format hint, box is already in xywh form, this just makes me feel better                        
    def box_to_xywh(self, box):
        """
        top_left_x, top_left_y, width and height
        """
        return box[0], box[1], box[2], box[3]
    
    ## returns the box center if box is in xywh format
    def box_centers(self, box):
        '''
        Retruns the center of the bounding box in xywh format
        '''
        center_x = box[0] - (box[2]/2)
        center_y = box[1] - (box[3]/2)
        return center_x, center_y

    ## maskrcnn either auto assigns if iou is sufficient or pauses if the closest iou is too far off
    def maskrcnn_predictions(self, tracker, tracker_number, box):
        '''!
        This function applies mask_rcnn predictions and measures IoU (intersection/union) similarity
        The best IoU value is selected to be tested if it passes the defined threshold
        If it passes the threshold, autoassign is triggered. 

        If the best IoU value fails a similarity threshold (different than IoU threshold) the tracker pauses.

        @note We define 2 thresholds and each do a different thing. The Minimum value for an auto assign is 45% IoU, this means the overlap must be very similar to auto-assign to mask-rcnn. 
            The idea here is that if they are already similar enough, we might as well set MaskRcnn. This may help with drifting values where the tracker may not adjust as well.

            The second threshold is a lower bound. If there are no sufficient nearby trackers, then we must assume that one or the other trackers failed.
            If this is the case, we force a pause so that this can be corrected by the user.

        '''

        if self.pred_dict and self.input_dialog.mcrnn_options.get_active() is True:
            if self.input_dialog.get_scrollbar_value() in self.pred_dict.keys():
                # print("GETTING IOU")

                top_left_x, top_left_y, width, height = self.box_to_xywh(box)

                iou, self.show_frame = self.maskrcnn.compute_iou(
                        box=(top_left_x, top_left_y, top_left_x + width, top_left_y + height), 
                        boxes=self.pred_dict[self.input_dialog.get_scrollbar_value()][0], 
                        boxes_area=self.pred_dict[self.input_dialog.get_scrollbar_value()][1], 
                        ratios=(self.resized_ratio_x, self.resized_ratio_y),
                        frame=self.frame
                    )
                # print("IOUs", iou)

                closest = max(iou)
                index = iou.index(closest)
                pred_box = self.pred_dict[input_dialog.get_scrollbar_value()][0][index]
                p1 = (int(pred_box[0]*self.resized_ratio_x), int(pred_box[1]*self.resized_ratio_y))
                p2 = (int(pred_box[2]*self.resized_ratio_x), int(pred_box[3]*self.resized_ratio_y))

                # if closest < 0.45:
                if closest <= self.input_dialog.mcrnn_options.get_min_value():
                    # print("Out of range")
                    self.show_frame = cv2.rectangle(self.frame, p1, p2, (150, 150, 220), 3)
                    self.input_dialog.set_pause_state()
                    self.activity_logger.paused(self.frame_number, "MRCNN", "MRCNN_Pause", self.tracker_list[tracker_number].get_name())

                elif closest >= self.input_dialog.mcrnn_options.get_auto_assign():
                    self.show_frame = cv2.rectangle(self.frame, p1, p2, (50, 200, 50), 3)
                    if tracker.auto_assign_state:
                        # print("Auto Assigning")

                        if self.input_dialog.play_state is True:
                            # Only record auto assignment while video is playing
                            self.activity_logger.adjustment(frame_number=self.frame_number, 
                                                        from_box=(top_left_x, top_left_y, (top_left_x + width), (top_left_y + height)), 
                                                        to_box=(p1[0], p1[1], p2[0], p2[1]), 
                                                        timer_id="MRCNN_Adjust",
                                                        tracker_id=self.tracker_list[tracker_number].get_name(),
                                                        intervention_type="MRCNN"
                                                        )

                        tracker.auto_assign(self.frame, (p1[0], p1[1], p2[0], p2[1]))

                #iterate through the predictions and check the difference from the closest IoU to the prediction
                for pred_index, pred in enumerate(iou):
                    diff = abs(pred - closest)

                    #if there is no sufficent prediction, we pause to allow the user to adjust.
                    if pred != closest and diff <= self.input_dialog.mcrnn_options.get_similarity():
                        # input_dialog.log("Possible ID Switch!")
                        self.activity_logger.paused(self.frame_num, "MRCNN", "MRCNN_Adjust", self.tracker_list[tracker_number].get_name())
                        pred_box = self.pred_dict[self.input_dialog.get_scrollbar_value()][0][pred_index]
                        pred_p1 = (int(pred_box[0]*self.resized_ratio_x), int(pred_box[1]*self.resized_ratio_y))
                        pred_p2 = (int(pred_box[2]*self.resized_ratio_x), int(pred_box[3]*self.resized_ratio_y))
                        self.show_frame = cv2.rectangle(self.frame, pred_p1, pred_p2, (255, 0, 255), 1)
                        self.show_frame = cv2.rectangle(self.frame, p1, p2, (50, 200, 50), 3)
                        self.input_dialog.set_pause_state()
                                    # .index(closest)

    ## a tracker has a region and will be drawn. Also deletes tracker if tracker is not regions
    def draw_tracker_regnoion(self, tracker, tracker_num):
        '''
        If the tracker is also defined as a region, the name is printed, the region is defined around an individual with said name and region is drawn.

        Also deletes tracjer region when selected off.
        '''
        if tracker.is_region() is True and tracker.get_name().strip() == "":
                        self.input_dialog.tab_list[tracker_num].getText(input_name="Region Name:",line=self.input_dialog.tab_list[tracker_num].name_line)

        if tracker.is_region() is True and tracker.get_name() != "":

            self.regions.set_moving_region(name = tracker.get_name(), 
                                    point = (int(self.center_x - self.width), int(self.center_y - self.height)),
                                    dimensions = (int(self.width*2), int(self.height*2)),
                                    id=tracker.id()
                                    )
            self.regions.display_region(self.show_frame)
        
        elif tracker.is_region() is False:
            # If tracker region is no longer selected, delete moving region
            self.regions.del_moving_region(tracker.get_name(), id=tracker.id())
    
    ## draws non-tracker regions (static)
    def draw_regions(self):
        '''
        If regions exist, draw them. This draws regions that are non-tracker based.
        '''
        #Display all regions on screen if they exist
        if len(self.regions.region_dict) > 0:
            self.show_frame = self.regions.display_region(self.show_frame)


    ## draws the tracker centers
    def draw_trackers(self):
        '''
        Draws the centers of the trackers 
        '''
        #Set the selected Tracker to Red
        for tracker in range(len(self.tracker_list)):
            self.app.processEvents()
            if tracker == self.selected_tracker:
                self.tracker_list[tracker].colour = (0,0,255)
            else:
                self.tracker_list[tracker].colour = (255,255,255)

    ## Deletes non-tracker region
    def trigger_delete_region(self):
        '''
        calls region delete and resets resets state
        '''
        if self.input_dialog.del_region_state is True:
            self.input_dialog.log("Select a region to remove...")
            self.regions.del_region()
            self.input_dialog.del_region_state = False
            self.input_dialog.log("Removing region complete.")

    ##Creates a region and resets the trigger state
    def trigger_region(self):
        '''
        Creates a region and resets trigger state
        '''
        if self.input_dialog.region_state is True:
            self.input_dialog.log("Adding region... Write name and then draw boundaries")
            self.regions.add_region(self.show_frame)
            self.input_dialog.region_state = False
            self.input_dialog.log("Adding region complete.")

    ## delete tab is triggered, we change index and delete the tracker
    def trigger_delete_tab(self):
        '''
        When delete tab is pressed, we adjust the current index to one before and remove the previously selected tracker
        '''
        if self.input_dialog.del_tab_state is True:
                
            self.input_dialog.log("Deleting Tracker!")
            del self.tracker_list[self.selected_tracker]
            self.input_dialog.tabs.setCurrentIndex(len(self.tracker_list))
            self.selected_tracker = self.input_dialog.tabs.currentIndex()
            self.input_dialog.del_tab_state = False

    ## Exports the currently selected tracker
    def trigger_export_current_tracker(self):
        '''!
        Exports the currently selected tracker

        Grabs the current width and height of the frame
        If the tracker contains data, export it and apply information required by the @ref  Multitracker.export_data() function
        Adds a version if applicable.

        @note If the csv is open by another program, an error will occur and prompt you to close it before exporting, the data will not be removed and everything will stay until this is done.
        '''
        # if key == ord("e") or input_dialog.export_state == True:
        if self.input_dialog.export_state == True:
            self.input_dialog.export_state = False
            self.input_dialog.log("Exporting " + self.tracker_list[self.selected_tracker].get_name() + "'s data recorded.")
            self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
            try:
                if self.tracker_list[self.selected_tracker].data_dict: # Ensure data exists in dictionary before exporting
                    if self.input_dialog.was_loaded:
                        self.tracker_list[self.selected_tracker].export_data(self.input_dialog.resolution_x, self.input_dialog.resolution_y, self.videoPath, self.vid_fps, new_version=self.input_dialog.get_new_data_version())
                        self.input_dialog.data_version_updated = False
                    else:
                        self.tracker_list[self.selected_tracker].export_data(self.input_dialog.resolution_x, self.input_dialog.resolution_y, self.videoPath, self.vid_fps)
            except IOError as err:
                self.input_dialog.log(err)
                self.input_dialog.show_warning(str(err) + "\n Please close open CSV and try again.")

    ## Resizes and adjusts colours
    def apply_image_options(self):
        '''
        Image must be resized and if any image modifications are done, they will be applied.

        If the image fails to apply, the frame might be invalid and will request that we continue until the next frame.
        '''
        returned = True
        try:
            # ret, frame = cap.read()
            self.frame = cv2.resize(self.frame, (self.input_dialog.resolution_x, self.input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)

        # if input_dialog.image_options.brightness.value() != 0:
            # frame = input_dialog.image_options.add_brightness(frame)


            if self.input_dialog.image_options.roi_normalize_flag:
                self.input_dialog.image_options.set_normalized_region(self.frame)
                self.input_dialog.image_options.roi_normalize_flag = False

            # if input_dialog.image_options.get_equalize_clahe_hist() is True:
            #     frame = input_dialog.image_options.enhance_normalized_roi(frame)

            if self.input_dialog.image_options.get_equalize_hist() is True:
                self.frame = self.input_dialog.image_options.equalize_hist(self.frame)
            
            if self.input_dialog.image_options.get_equalize_clahe_hist():
                self.frame = self.input_dialog.image_options.equalize_clahe_hist(self.frame)

            if self.input_dialog.image_options.get_alpha() != 10 or self.input_dialog.image_options.get_beta() != 10: 
                self.frame = self.input_dialog.image_options.enhance_brightness_contrast(self.frame)
            
            if self.input_dialog.image_options.get_gamma() != 10:
                self.frame = self.input_dialog.image_options.enhance_gamma(self.frame)

            # frame = input_dialog.image_options.enhance_brightness(frame)
            # _, frame = input_dialog.image_options.auto_enhance(frame)

            self.show_frame = self.frame.copy()
            


        except Exception as e:
            crashlogger.log(str(e))
            self.input_dialog.snap_state = "Forward"
            print("No resize")
            returned = False
        return returned

    ## Reads new images when play_state is true
    def continue_reading(self):
        '''
        Allows the frames to continue to read when play state is true. 

        We restrict read when scroll bar is changed since that should be paused after.
        Updates the scroll bar, and skips if the frame is unable to read.
        '''
        continuing = False

        if self.input_dialog.play_state == True and not self.input_dialog.scrollbar_changed:
            self.frame, self.frame_num = self.fvs.read()
            self.previous_frame = self.frame

            self.input_dialog.set_scrollbar(self.frame_num)
            if self.frame_num != self.input_dialog.get_scrollbar_value():
                print("Trying Again:", self.frame_num, self.fvs.frame_number, self.input_dialog.get_scrollbar_value())
                continuing = True
            
        else:
            self.frame = self.previous_frame
        return continuing

    ## Scrollbar changes, options are configured
    def trigger_scrollbar_change(self):
        '''
        If scrollbar changes, a set of things must be updated
        An activity is recorded in the activity logger
        
        When we scroll, we must keep snapped the the increment of frame skip. We snap to the nearest multiple of frame_skip
        The state is set to false, and we update frame number
        Fvs is reset and new frame is then read from fvs

        We end the changed flag
        '''
        if self.input_dialog.scrollbar_changed == True:
            if self.tracker_list:
                self.activity_logger.slider_moved(self.frame_num, "SLIDER", self.tracker_list[self.selected_tracker].get_name())
            #If Snapping enabled, snap the scrollbar to the nearest multiple of skip_frame
            # print("MOD", (input_dialog.get_scrollbar_value() % input_dialog.get_frame_skip()))
            if self.input_dialog.get_scrollbar_value() % self.input_dialog.get_frame_skip() != 0 and self.input_dialog.snap_to_frame_skip:
                
                rounded = round(self.input_dialog.get_scrollbar_value()/self.input_dialog.get_frame_skip())*self.input_dialog.get_frame_skip()
                # print("Rounding!", rounded)
                self.input_dialog.set_scrollbar(rounded)

            # input_dialog.mediaStateChanged()
            self.input_dialog.play_state = False
            self.fvs.frame_number = self.input_dialog.get_scrollbar_value()
            # input_dialog.log("Scrolled.")  
            self.fvs.reset = True
            self.frame, self.frame_num = self.fvs.read()
            # print(frame_num)
            self.previous_frame = self.frame
            
            self.input_dialog.scrollbar_changed = False


    ## adds a tracker
    def triggered_add_tab(self):
        '''
        When we add a tab, append a new Multitracker object.
        Pauses and disables tabs, add tab, export tab. When we finish creating the tracker, we can then add another tracker.
        '''
        #When we add a tab, finish initializing it before anything else can continue
        if self.input_dialog.add_tab_state == True:
            self.input_dialog.log("Adding Tab!")
            self.input_dialog.tabs.setCurrentIndex(len(self.tracker_list))
            self.selected_tracker = self.input_dialog.tabs.currentIndex()
            self.tracker_list.append(MultiTracker(self.input_dialog.tab_list[self.selected_tracker]))
            
            self.input_dialog.tabs.setEnabled(False)

            self.input_dialog.add_tab_btn.setEnabled(False)
            self.input_dialog.del_tab_btn.setEnabled(False)
            self.input_dialog.export_tab_btn.setEnabled(False)

            self.input_dialog.tabs.setEnabled(True)
            self.input_dialog.add_tab_state = False

    ## Checks if video is at the end
    def check_end_video(self):
        '''
        Checks if the video is at the end, if so set it to the beginning, pause and set everything to read. 
        Adjusts scrollbar and updates with fvs reset
        '''
        #When at the end, go to the beginning and pause
        if self.input_dialog.get_scrollbar_value() >= self.input_dialog.vidScroll.maximum() or (self.input_dialog.get_scrollbar_value() + self.skip_frame) >= self.input_dialog.vidScroll.maximum():
            self.fvs.frame_number = self.input_dialog.get_scrollbar_value()
            if not self.input_dialog.record_live:
                self.input_dialog.set_pause_state()
                self.input_dialog.set_all_tabs("Read")

                self.input_dialog.set_scrollbar(0)
                self.activity_logger.paused(self.fvs.frame_number, "END_VIDEO", "END_VIDEO", None)
                self.fvs.reset = True
            self.input_dialog.scrollbar_changed = True

    ## Left arrow goes backwards one set of skip-frame amount
    def triggered_snap_backward(self):
        '''
        Snap backward is called, use fvs reset, adjust the scroll bar, and check if frame number is less than or equal to original frame
        '''
        if self.input_dialog.snap_state == "Backward":
            self.fvs.reset = True
            if self.frame_num - self.input_dialog.get_frame_skip() > 0:
                self.input_dialog.set_scrollbar(self.input_dialog.get_scrollbar_value() - self.input_dialog.get_frame_skip())
            else:
                self.input_dialog.set_scrollbar(0)
                self.fvs.frame_number = 0

            self.input_dialog.scrollbar_changed = True
            self.input_dialog.snap_state = None
            snap_called = True
            self.fvs.reset = True

    ## Right arrow moves the video forward one step of skip-frame
    def triggered_snap_forward(self):
        '''
        When the forward button (right arrow) is pressed, adjust the scroll bar, pause and call fvs reset.

        '''
        # print(frame_num, skip_frame)
        if self.input_dialog.snap_state == "Forward":
            self.fvs.reset = True
            self.input_dialog.set_scrollbar(self.input_dialog.get_scrollbar_value() + self.input_dialog.get_frame_skip())
            self.fvs.frame_number = self.input_dialog.get_scrollbar_value() + self.input_dialog.get_frame_skip()
            
            self.input_dialog.scrollbar_changed = True
            self.input_dialog.snap_state = None
            snap_called = True
            self.fvs.reset = True


    ## Disables export when there is no tracker data
    def no_data_disable_export(self):
        '''
        If there is no tracker data disable the export button
        '''
        #if there's no data to export, grey out export button.
        if self.tracker_list:
            if not self.tracker_list[self.selected_tracker].data_dict:
                self.input_dialog.export_tab_btn.setEnabled(False)
            else:
                self.input_dialog.export_tab_btn.setEnabled(True)
        elif self.selected_tracker == -1:
            self.input_dialog.export_tab_btn.setEnabled(False)

    ## when skip-frame is adjusted, snap to the closest round number
    def update_skip_frame(self):
        '''
        If skip frame is changed in the UI we update the variable to new values fvs reset is set to TRUE which adjusts streaming values
        '''
        if self.fvs.skip_value != self.input_dialog.get_frame_skip():
            print("Skipbo")
            self.fvs.reset = True
            # frame = fvs.read()
            self.fvs.skip_value = self.input_dialog.get_frame_skip()

    ## Quits the program
    def trigger_quit(self):
        '''
        Quits the program, closing cv2 and UI
        '''
        if self.input_dialog.quit_State is True:
            # sys.exit(app.exec_())
            # cap.release()
            # fvs.stop()
            # cv2.destroyAllWindows()
            self.input_dialog.log("Closing trackers")
            self.thread_pool.terminate()
            self.input_dialog.log("Cap Release")
            self.cap.release()
            self.input_dialog.log("Destroy cv2")
            cv2.destroyAllWindows()
            self.input_dialog.log("Quitting App")
            
            self.input_dialog.log("System Exit")
            self.input_dialog.close()


            self.app.quit()
            self.input_dialog.log("Stopping FVS")

            # fvs.stop()
            os._exit(1)
    
    ## Records when a user exports something into datalogger.
    def trigger_export_activity(self):
        '''
        Only records a data-logger activity of export
        '''
        if self.input_dialog.export_activity:
            export_filename = str(self.videoPath[:-4])
            self.activity_logger.export_activity(export_filename)
            self.input_dialog.export_activity = False

    ## Exports video characteristics
    def trigger_characteristics(self):
        '''!
        Exports a set of video characteristics including optical flow, pan and zoom at frame wide median and mean values
        
        @note this will take a while since it must calculate the entire video.
        '''
        if self.input_dialog.export_charactoristics:
            print("Exporting")
            export_filename = str(self.videoPath[:-4])
            text, ok = PyQt5.QtWidgets.QInputDialog.getText(self.input_dialog, 'Video Location', 'Enter Recorded Location:')
            self.activity_logger.video_location = text
            self.activity_logger.get_video_characteristics()
            self.activity_logger.export_charactoristics(export_filename)
            self.input_dialog.export_charactoristics = False

    ## Uses predicted values and tracks them with numbered ids
    def trigger_track_predictions(self):
        '''
        when track predictions is triggered, the video will play through with all predictions present. 
        Each prediction will have a numbered id.
        The ids are centroids which will track along using a nearest neigbour method.

        The data will be exported into a filename video_name_Predictions_Ids.csv 
        '''
        if self.input_dialog.track_preds_state is True and bool(self.pred_dict) is True:
            export_filename = str(self.videoPath[:-4]) + "_Predictions_Ids.csv"
            self.input_dialog.track_preds_state = False
            prediction_dict = self.maskrcnn.track_predictions(self.pred_dict, self.videoPath, preview=True)
            prediction_dict.to_csv(export_filename)

    ## Exports every tracker that has data.
    def trigger_export_all(self):
        '''!
        Iterates through every tracker and exports their data individually, then returns export all state to false
        
        @warning Data will be removed from data_dict. Information will need to be re-loaded in order to re-export.
            This was set to avoid re-writing duplicate data
        '''
        if self.input_dialog.export_all_state is True:
                self.input_dialog.export_all_state = False
                for tracker in self.tracker_list:
                    if tracker.data_dict: # Check if dataframe is empty before exporting.
                        if self.input_dialog.was_loaded:
                            self.input_dialog.log("Exporting to new version")
                            tracker.export_data(self.input_dialog.resolution_x, self.input_dialog.resolution_y, self.videoPath, self.vid_fps, new_version=self.input_dialog.get_new_data_version())
                            self.input_dialog.data_version_updated = False
                        else:
                            tracker.export_data(self.input_dialog.resolution_x, self.input_dialog.resolution_y, self.videoPath, self.vid_fps)

                # When we export everything assume loaded data is all exported so we do not overwrite

    ## loads the predictions into pred_dict dataframe
    def trigger_load_predictions(self):
        '''
        Loads maskRCNN dataframe saved from predictions.

        Loads from video_name_predicted.csv

        This data will be drawn in maskrcnn_predictions() function
        '''
        if self.input_dialog.load_predictions_state is True:
            pred_dict = self.maskrcnn.load_predicted((self.videoPath[:-4] + "_predict.csv"))
            print(pred_dict)
            self.input_dialog.load_predictions_state = False

    ## DOES NOTHING
    def trigger_predict(self):
        if self.input_dialog.predict_state is True:
            #UNCOMMENT BELOW
            # frame, rois, scores = maskrcnn.predict(videoPath, step=input_dialog.skip_frames.value(), display=True, logger=input_dialog.log)
            self.input_dialog.predict_state = False

    ## 
    def trigger_play(self):
        '''
        Changes pause state to play and records activity
        '''
        # This is needed for activity logger to end pause timers
        if self.input_dialog.pause_to_play:
            print("Playing...")
            self.activity_logger.end_pause()
            self.activity_logger.end_slider(self.fvs.frame_number, "SLIDER")
            self.input_dialog.pause_to_play = False
    
    ## Turns play state to pause
    def trigger_pause(self):
        '''
        Changes play state to pause and records activity
        '''
        if self.input_dialog.play_to_pause:
            if self.tracker_list:
                self.activity_logger.paused(self.frame_num, "USER", "USER_Pause",  self.tracker_list[self.selected_tracker].get_name())
            self.input_dialog.play_to_pause = False

    ## If load video state is set, load the video
    def trigger_load_video(self):
        '''!
        Triggered if @ref load_tracked_video_state is True.

        helper to @ref load_tracker_data()
        '''
        if self.input_dialog.load_tracked_video_state:
            # import_filename = str(videoPath[:-4]) + ".csv"
            import_filename = self.input_dialog.openFileNameDialog(task="Select CSV to load Tracker Data", extensions="*.csv")
            loaded_trackers, frame = load_tracker_data(import_filename, self.input_dialog, frame)
            self.tracker_list.extend(loaded_trackers)
            self.input_dialog.load_tracked_video_state = False   
        
    ## Loads data from csv into multiple trackers
    def load_tracker_data(self, csv, frame):
        '''!
        Populates tabs, data_dict and trackers previously labelled from csv.
        All trackers will start in Read Only mode.

        Useful for when the data needs to be saved/loaded between recording sessions.

        Please note CSV must match the video or data will be uninformative.

        If you modify the data by hand errors may occur because of different formats, and how third party programs save NULL and TRUE/FALSE values.
        
        @param input_dialog defined in qt_dialog.
        @param frame cv2 image
        '''
        print("Loading csv")
        new_trackers = []

        df = pd.read_csv(csv)
        df = df.loc[1:] # Ignore first line, this is metadata
        df[["Name"]].fillna("")
        df[["ID"]].fillna("")

        # make sure columns are the proper data types
        # df["Present At Beginning"] = df["Present At Beginning"].astype('bool')
        # df["Other_Room"] = df["Other_Room"].astype('bool')
        # df["Chair"] = df["Chair"].astype('bool')
        # df["Frame_Num"] = df["Frame_Num"].astype('int32')
        # unique_trackers = df.groupby(['Name','ID'], as_index=False).size()
        # unique_trackers = df.groupby(['Name','ID']).size().reset_index().rename(columns={0:'count'})

        unique_trackers = df[['Name', 'ID']].drop_duplicates()
        unique_trackers = unique_trackers[unique_trackers['Name'].notna()]
        for name_index in range(len(unique_trackers)):
            print(name_index)
            # get the unique ids
            name = unique_trackers["Name"].iloc[name_index]
            pid = unique_trackers["ID"].iloc[name_index]

            # grab all the columns where this data exists.
            tracker_data = df.loc[(df['Name'] == name)]

            # Add a new tab to the 
            tab_index, new_tab = self.input_dialog.add_tab()
            self.input_dialog.add_tab_state = False

            # Fill string columns NaN to empty string
            tracker_data["Name"] = tracker_data["Name"].fillna("")
            tracker_data["ID"] = tracker_data["ID"].fillna("")
            tracker_data["Sex"] = tracker_data["Sex"].fillna("")
            tracker_data["Description"] = tracker_data["Description"].fillna("")
            tracker_data["Group_Size"] = tracker_data["Group_Size"].fillna(0)
            tracker_data["Region"] = tracker_data["Region"].fillna("")
            print(tracker_data.iloc[1])

            # Build tab info, this tab info is used to build the tracker
            new_tab.name_line.setText(tracker_data["Name"].iloc[1])
            new_tab.id_line.setText(tracker_data["ID"].iloc[1])
            new_tab.sex_line.setText(str(tracker_data["Sex"].iloc[1]))
            new_tab.group_line.setText(str(tracker_data["Group_Size"].iloc[1]))

            description_text = ""

            #check for version number
            if tracker_data["Description"].iloc[1] == "":
                description_text = str(tracker_data["Description"].iloc[1]) + "V"
            else:
                description_text = str(tracker_data["Description"].iloc[1]) + "\nV"

            description_text += str(self.input_dialog.newest_version)

            # assign loaded values
            new_tab.desc_line.setPlainText(description_text)
            new_tab.update_length_tracked(float(tracker_data["Total_Sec_Rec"].iloc[1]))
            new_tab.read_only_button.setChecked(True)
            new_tab.read_only = True
            tracker_data['Present At Beginning'] = (tracker_data['Present At Beginning'] == 'TRUE')

            # 
            try:
                new_tab.beginning_button.setChecked(eval(tracker_data["Present At Beginning"].iloc[1]))
            except:
                print(tracker_data["Present At Beginning"].iloc[1])
                print("Error occured when setting to beginning while loading. Setting it to False by default")
                new_tab.beginning_button.setChecked(False)



            
            # Build new tracker with info loaded
            tracker = MultiTracker(new_tab)
            tracker.reset = True
            
            # Assign static variables not from tab to Multitracker object
            # tracker.beginning = 

            
            # Populate data dict
            for index, row in tracker_data.iterrows():
                frame_num = int(row["Frame_Num"])
                center = (float(row["Pixel_Loc_x"]), float(row["Max_Pixel_y"]) - float(row["Pixel_Loc_y"]) )
                regions = str(row["Region"]).strip("[]").split(", ")
                height = abs(float(row["BBox_TopLeft_y"]) - float(row["BBox_BottomRight_y"]))
                width = abs(float(row["BBox_TopLeft_x"]) - float(row["BBox_BottomRight_x"]) )

                dimensions = (int(width), int(height))

                other_room = eval(str(row["Other_Room"]))
                is_chair = eval(str(row["Chair"]))
                total_people = int(row["Total_People"])


                tracker.data_dict[frame_num] = (center, regions, dimensions, other_room, total_people, is_chair)
            
            current_frame = self.input_dialog.get_scrollbar_value()
            tracked_frames = list(tracker.data_dict.keys())

            # Set tracker to closest frame
            closest = tracked_frames[0]
            min_difference = abs(current_frame - closest)
            for value in tracked_frames[1:]:
                difference = abs(current_frame - value)
                if difference < min_difference:
                    closest = value
                    min_difference = difference


            data = tracker.data_dict[closest]

            w = int(data[2][0])
            h = int(data[2][1])
            x = int(data[0][0])
            y = int(data[0][1])

            # Correct for invalid non-area data
            if w <= 0:
                w = 5
            if h <= 0:
                h = 5
            if x <= 0:
                x = 10
            if y <= 0:
                y = 10

            tracker.init_bounding_box = (x, y, w, h)
            frame = cv2.circle(frame, center=(x,y), radius=3, color=(5,5,5), thickness=2)
            frame = cv2.rectangle(frame, (x-int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), color = (255, 0, 0), thickness = 2)
            tracker.box = tracker.init_bounding_box
            tracker.auto_assign(frame, xywh=((x-int(w/2)), (y - int(h/2)), (x + int(w/2)), (y + int(h/2))))

            #cv2.imshow("Loading", frame)
            # tracker.auto_assign(frame, (p1[0], p1[1], p2[0], p2[1]))
            # self.tracker.init(frame, init_bounding_box)

            # Append new tracker to list
            new_trackers.append(tracker)


        #Close all applications.
        # sys.exit(app.exec_())
        # cap.release()
        # fvs.stop()
        # cv2.destroyAllWindows()
        print("End of loading.")
        return new_trackers, frame
    
    def trigger_delete_pressed(self):
        '''
        Triggered if backspace is pressed.

        pauses the video and moves one frame backwards.
        Toggles read only if it is false (sets it to true) It no longer records new data.

        Deletes current and previous frame

        '''

        # Activate if we delete a frame
        if self.input_dialog.del_frame:
            self.input_dialog.snap_state = "Backward"
            self.selected_tracker = self.input_dialog.tabs.currentIndex()
            # input_dialog.tab_list[selected_tracker].read_only = True #we set this with a toggle later on

            try:
                ## NOTE THIS IS WHERE AND ERROR HAPPENS YOOOOO (values in this activity are not available in first iteration)
                self.activity_logger.adjustment(frame_number=self.frame_number, 
                                            from_box=(None, top_left_y, (top_left_x + width), (top_left_y + height)), 
                                            to_box=None, 
                                            timer_id="DELETE_FRAME",
                                            tracker_id=self.tracker_list[self.tracker_num].get_name(),
                                            intervention_type="USER"
                                            )
            except:
                self.activity_logger.adjustment(frame_number=self.frame_number, 
                            from_box=(None, None, None, None), 
                            to_box=None, 
                            timer_id="DELETE_FRAME",
                            tracker_id=self.tracker_list[self.tracker_num].get_name(),
                            intervention_type="USER"
                            )
            
            # if read only is false
            if self.input_dialog.tab_list[self.selected_tracker].read_only is False:
                print("RESETTING")
                self.input_dialog.tab_list[self.selected_tracker].toggle_read()

            # if frame is in data dict, we delete that data
            if (self.fvs.frame_number) in self.tracker_list[self.selected_tracker].data_dict.keys():
                del self.tracker_list[self.selected_tracker].data_dict[(self.fvs.frame_number)]
                self.input_dialog.del_frame = False

            # if the previous frame is in data dict, delete that, otherwise no data and we end the trigger
            if (self.fvs.frame_number-self.fvs.skip_value) in self.tracker_list[self.selected_tracker].data_dict.keys():
                del self.tracker_list[self.selected_tracker].data_dict[(self.fvs.frame_number-self.fvs.skip_value)]
                self.input_dialog.del_frame = False
            else:
                self.input_dialog.log("No track to remove on this frame.")
                self.input_dialog.del_frame = False

#This main is used to test the time
if __name__ == "__main__":
    run()