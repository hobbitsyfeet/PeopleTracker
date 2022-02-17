# -*- coding: utf-8 -*-
import math
import multiprocessing
import os
import sys
# the input dialog
#For popup windows (DEPRECIATED)
#import tkinter as tk
import traceback
# from collections import deque
# from multiprocessing.pool import ThreadPool
from random import randint
from sys import exit
from threading import Thread

import cv2
import exiftool
import imutils
import numpy as np
import pandas as pd

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QApplication, QComboBox, QInputDialog, QLineEdit,
                             QMessageBox, QWidget,QSplashScreen)

import crashlogger

#import mbox
import qt_dialog
from Video import FileVideoStream, STFileVideoStream
import filters
import regression
from Regions import Regions

import evaluate

CPU_COUNT = multiprocessing.cpu_count()

#start tracking version at 1.0
PEOPLETRACKER_VERSION = 2.5

# For extracting video metadata
# import mutagen

class MultiTracker():
    def __init__(self, tab, name="Person", colour=(255,255,255)):    
        self.name = tab.name_line
        self.pid = tab.id_line
        self.colour = colour

        # self.location_data = [] #(x, y)
        self.distance_data = [] #(CLOSE, MED, FAR) (estimated)
        self.time_data = [] #(frames tracked)
        # self.region_dict = dict

        self.data_dict = dict()
        self.previous_time = 0

        #Set these variables to getter functions (to update whenever accessed)
        self.sex = tab.sex_line
        self.group = tab.group_line
        self.description = tab.desc_line
        self.beginning = tab.get_beginning
        self.is_region = tab.get_is_region
        # self.region_state = tab.get_region_state
        self.is_chair = tab.get_is_chair
        self.read_only = tab.get_read_only
        self.id = tab.get_uuid
        # self.other_room = tab.get_other_room
        self.record_state = False

        self.tracker = None

        # initialize the bounding box coordinates of the object we are going
        # to track
        self.init_bounding_box = None
        self.reset = False
        self.state_tracking = False
        self.auto_assign_state = True

        self.predictor = filters.KalmanPred()
        self.box_predictor = (filters.KalmanPred(white=True), filters.KalmanPred(white=True)) #One point predictor for two points on box (Top Left, Bottom Right)
        # self.regression = None
        self.regression = regression.rolling_regression(10, "Linear")
        self.predicted_bbox = None
        self.predicted_centroid = None

    # def compare_predicted(self, location, frame_num, predicitons):
    #     """
    #     Compares predicted file to tracked file in frame using IOU and overlapping measures.
    #     """

    def grab_cut(self, frame, rect, export_path=None):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)


        mask = np.zeros(frame.shape[:2],np.uint8)

        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, iterCount=2, mode=cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = frame*mask2[:,:,np.newaxis] 
        cv2.imshow("Cropped Image", img)
        

    def get_name(self):
        """ Returns name of person tracked: returns string """
        return self.name.text()
    
    def get_id(self):
        """ Returns ID of person tracked: returns string """
        return self.pid.text()

    # def get_loc(self, frame):
    #     """ Returns pixel (x,y) location of person tracked: returns tuple(int,int)"""
    #     return self.location_data(frame)
    
    def get_sex(self):
        """ Returns sex of person being tracked: returns string """
        return self.sex.text()

    def get_description(self):
        return self.description.toPlainText()

    def get_group(self):
        return self.group.text()

    def get_time_tracked(self, framerate):
        """ Returns total time being tracked in video: returns  """
        
        # if len(self.data_dict) > 2:
        total_time = [self.calculate_total_time(self.part_time_to_segments(self.data_dict.keys()), framerate)]
        return total_time

    def create(self, tracker_type='CSRT'):
        """
        The creation of the Opencvs Tracking mechanism.

        tracker_type = ["BOOSTING", "MIL","KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]
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



    def rename(self, name):
        """
        Renames the tracking object
        """
        self.name = name
        

    def set_colour(self, color=(255,255,255)):
        self.colour = color

    def assign(self, frame, tracker_type="CSRT"):
        """
        Assigns the box, name and sex of a tracked person.
        Takes care of reassigning as well.
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
            self.self.init_bounding_box = None
            return

        input_dialog.log("Bounding Box Coordinates: " + str(self.init_bounding_box))

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        if self.init_bounding_box is not None:
            # self.tracker.init(frame, self.init_bounding_box)
            input_dialog.log("Testing bounding box exists: " + str(self.reset))
            self.reset = True

        if self.reset is True:
            try:
                input_dialog.log("Setting Location")
                del self.tracker
                self.create(tracker_type)
                self.tracker.init(frame, self.init_bounding_box)
                input_dialog.log(("Setting Location Successful. " + "Bounding Box Coordinates: " + str(self.init_bounding_box)))
                
                # Tell the predictors to reset
                self.box_predictor[0].reset()
                self.box_predictor[1].reset()
                self.predictor.reset()

            except Exception as e:
                crashlogger.log(str(e))
                input_dialog.log("Setting Location Failed.")
    

    def auto_assign(self, frame, bounding_box, tracker_type="CSRT"):
        try:
            # print(self.init_bounding_box)
            # print(bounding_box)
            width = bounding_box[2] - bounding_box[0]
            height = bounding_box[3] - bounding_box[1]
            bounding_box = [bounding_box[0], bounding_box[1], width, height]
            # print(bounding_box)
            input_dialog.log("Setting Location")
            del self.tracker
            self.create(tracker_type)
            self.tracker.init(frame, bounding_box)
            input_dialog.log("Setting Location Successful.")
        except Exception as e:
            crashlogger.log(str(e))
            input_dialog.log("Setting Location Failed.")

    def update_tracker(self, frame):
        """
        track and draw box on the frame
        """
        success = False
        box = None
        # check to see if we are currently tracking an object
        if self.init_bounding_box is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = self.tracker.update(frame)
    
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    self.colour, 2)
                cv2.rectangle(frame, (x , y - 1), (x + 10 * (len(self.get_name())) , y - 15),(255,255,255),-1)
                cv2.putText(frame,self.get_name(), (x , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)
                
                # cv2.putText(frame,self.get_name(), (x , y - 1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,0,0),1)
        return success, box, frame

    def remove(self):
        """
        Removes the tracker from being recorded. DOES NOT YET DELETE INFO
        """
        del self.tracker
        self.init_bounding_box = None
        # del self

    def predict(self,):
        """
        Applies KNN taking into consideration a Hidden Markov Model
        """
        pass

    def record_data(self, frame, total_people, x=-1, y=-1, w=-1, h=-1, regions=[], other_room=False):
        """
        Appends location and time to a list of data

        X,Y,W,H No Data are described as -1, and regions is described as []
        """
        # input_dialog.log("recording Frame" + str(frame))
        # self.location_data.append((int(x),int(y))) #(x, y)
        # self.time_data.append(frame) #(frames tracked)
        point = (int(x),int(y))
        dimensions = (int(w), int(h))
        self.data_dict[frame] = (point, regions, dimensions, other_room, total_people, self.is_chair())

        # self.distance_data.append(self.estimate_distance(size)) #(CLOSE, MED, FAR) (estimated)

    def append_data(self, dataframe):
        """
        Appends the data into a given dataframe categoriezed with the name given
        """
        pass

    def export_data(self, vid_width, vid_height, vid_name, fps):
        """
        Exports the recorded data and appends constants such as name, total time recorded, and pixel percent Loc
        """
        # if not os.path.exists(("./data/" + vid_name[:-4])):
        #     os.makedirs(("./data/" + vid_name[:-4]))
        # self.data_dict[frame] = (point, regions, dimensions, other_room, total_people, self.is_chair())
        input_dialog.log(self.data_dict)

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
        total_time = self.get_time_tracked(vid_fps)
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
        data = {"Frame_Num":frames,#self.time_data,



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
        
        df = pd.DataFrame(data)
        export_csv = df.to_csv (export_filename, index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path

        self.data_dict = dict()

        input_dialog.log("Export Data Complete!")
        
    def evaluate_data(self):
        te = evaluate.tracker_evaluation()
        

    def merge_intervals(self, total_segments):
        """
        On a continous set of intervals, merges them such that intersecting inervals will become a new interval with new limits. 
        Example: [[0, 20], [10, 50]] becomes [[0,50]]
        [[0, 500]] stays the same
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

    def part_time_to_segments(self, time_data, segment_size=300):
        """
        Parts the times into segments based on the distance between tracked frames.
            time_data is the entire time of the object tracked.
            segment_size is the distance between frames that should be considered as one segment. Default is 10 seconds at 30fps.
            min_segment is the minimum size of segment allowed. If it is too small, it will be ignored. Defualt is 1 second (30 frames).

        Example with default segment size of 300:
            [0, 10, 20, 30, 100, 200, 300, 400, 1000, 1010, 1020] results in [[0, 400], [1000, 1020]]
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

    def estimate_distance(self, size):
        pass
            
    def calculate_time(self, frame_start, frame_end, fps=30):
        """
        Calculates time between two given frames and the FPS rate
        """
        time = (frame_end - frame_start)/fps
        return time
    
    def calculate_total_time(self, total_frames, fps=30, segmented=True):
        """
        Calculates total time tracked. This takes into consideration segmented time
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


def export_null_meta(vid_dir):
    #Create path string for exporting the data. It's just a change of extention.
    export_filename = str(videoPath[:-4]) + ".csv"
    
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
        
def export_meta(vid_dir):
    """
    Exports a start line into a CSV with the same name as the video. This CSV will initialize with a set of columns on the left
    with dashed lines for recorded data, and a set of columns on the right with Metadata.
    This should only be called once, and will only initialize if the file does not exist- will not overwrite.

    Input: vid_dir: directory of filename (string)
    """
    #get metadata from filepath
    #NOTE: Exiftool must be in path to access, otherwise pass a string to the path.
    try:
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata(vid_dir)
    except Exception as e:
        crashlogger.log(str(e))
        print("Could not collect")
        export_null_meta(vid_dir)
        return False

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
        return True
    
    #Close all applications.
    # sys.exit(app.exec_())
    # cap.release()
    # fvs.stop()
    # cv2.destroyAllWindows()

#This main is used to test the time
if __name__ == "__main__":
    try:
            # initialize the log settings
        # logging.basicConfig(filename = 'app.log', level = logging.INFO)

        trackerName = 'CSRT'

        #Create QT application for the UI
        app = QApplication(sys.argv)
        input_dialog = qt_dialog.App()


        #Get the video path from UI
        videoPath = input_dialog.filename

        input_dialog.log("Populating UI")
        #Given the path, export the metadata and setup the csv for data collection
        export_meta(videoPath)
        
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

        resized_ratio_x = input_dialog.resolution_x / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        resized_ratio_y = input_dialog.resolution_y / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print("WIDTH:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), " HEIGHT:", cv2.CAP_PROP_FRAME_HEIGHT)

        # print(resized_ratio_x, resized_ratio_y)

        input_dialog.set_max_scrollbar(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_dialog.resolution_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_dialog.resolution_y)
        # fvs = FileVideoStream(videoPath).start()
        fvs = STFileVideoStream(videoPath)

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

        while True:
            QCoreApplication.processEvents()
            
            if input_dialog.predict_state is True:
                #UNCOMMENT BELOW
                # frame, rois, scores = maskrcnn.predict(videoPath, step=input_dialog.skip_frames.value(), display=True, logger=input_dialog.log)
                input_dialog.predict_state = False

            if input_dialog.export_all_state is True:
                input_dialog.export_all_state = False
                for tracker in tracker_list:
                    tracker.export_data(input_dialog.resolution_x, input_dialog.resolution_y, videoPath, vid_fps)
            if input_dialog.load_predictions_state is True:
                pred_dict = maskrcnn.load_predicted((videoPath[:-4] + "_predict.csv"))
                print(pred_dict)
                input_dialog.load_predictions_state = False
            
            if input_dialog.track_preds_state is True and bool(pred_dict) is True:
                input_dialog.track_preds_state = False
                maskrcnn.track_predictions(pred_dict, videoPath, preview=True)
                


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
                fvs.reset = True
                

            #When at the end, go to the beginning and pause
            if input_dialog.get_scrollbar_value() >= input_dialog.vidScroll.maximum() or (input_dialog.get_scrollbar_value() + skip_frame) >= input_dialog.vidScroll.maximum():
                input_dialog.set_scrollbar(0)
                fvs.frame_number = input_dialog.get_scrollbar_value()
                fvs.reset = True
                input_dialog.scrollbar_changed = True
                input_dialog.set_pause_state()
                input_dialog.set_all_tabs("Read")
            
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
                print(frame_num)
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
            
            elif input_dialog.region_state is True:
                input_dialog.log("Adding region... Write name and then draw boundaries")
                regions.add_region(show_frame)
                input_dialog.region_state = False
                input_dialog.log("Adding region complete.")
            elif input_dialog.del_region_state is True:
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

            #Loop through every tracker and update
            for tracker in enumerate(tracker_list):
                tracker_num = tracker[0]
                tracker = tracker[1]

                frame_number = frame_num
                try:
                    if input_dialog.tab_list[tracker_num].other_room:
                        tracker.record_data(frame_number, input_dialog.num_people.value(), other_room=True)
                        regions.del_moving_region(tracker.get_name(), id=tracker.id())

                    elif tracker.init_bounding_box is not None and input_dialog.tab_list[tracker_num].active is True and input_dialog.tab_list[tracker_num].read_only is False:
                        
                        #allocate frames on GPU, reducing CPU load.
                        cv2.UMat(frame)    

                        app.processEvents()
                        #track and draw box on the frame
                        success, box, show_frame = tracker.update_tracker(show_frame)
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

                                elif closest >= input_dialog.mcrnn_options.get_auto_assign():
                                    show_frame = cv2.rectangle(frame, p1, p2, (50, 200, 50), 3)
                                    if tracker.auto_assign_state:
                                        # print("Auto Assigning")
                                        tracker.auto_assign(frame, (p1[0], p1[1], p2[0], p2[1]))

                                for pred_index, pred in enumerate(iou):
                                    diff = abs(pred - closest)

                                    if pred != closest and diff <= input_dialog.mcrnn_options.get_similarity():
                                        # print(diff)
                                        # input_dialog.log("Possible ID Switch!")
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
                        
                        if input_dialog.play_state == True and input_dialog.tab_list[tracker_num].read_only is False:
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

                            tracker.record_data(input_dialog.get_scrollbar_value(), input_dialog.num_people.value(), center_x, center_y, width, height, in_region)

                except Exception as e:
                    crashlogger.log(str(e))
                    # print("No resize")
                    
                    input_dialog.log("Crashed while deleting. Continuing")
                    continue

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
                        tracker_list[selected_tracker].assign(frame, trackerName)
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
            
            #Display all regions on screen if they exist
            if len(regions.region_dict) > 0:
                show_frame = regions.display_region(show_frame)

            # print("FRAMES", frame_num, fvs.frame_number, input_dialog.get_scrollbar_value())
            #When done processing each tracker, view the frame
            cv2.imshow("Frame", show_frame)
        
            # input_dialog.videoWindow.show_image(frame)
    except Exception as e:
        print(traceback.format_exc())
        crashlogger.log(str(traceback.format_exc()))
