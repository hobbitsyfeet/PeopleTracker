import os

import cv2
from random import randint
from sys import exit
import imutils
import pandas as pd

import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
from multiprocessing.pool import ThreadPool
from threading import Thread
from collections import deque

#For popup windows (DEPRECIATED)
import tkinter as tk
from tkinter import simpledialog

# the input dialog
import tkinter
import mbox

import numpy as np

import sys
from PyQt5.QtWidgets import QApplication
import qt_dialog

import math

# For extracting video metadata
# import mutagen

class MultiTracker():
    def __init__(self, tab, name="Person", colour=(255,255,255)):
        self.name = tab.name_line
        self.colour = colour

        # self.location_data = [] #(x, y)
        self.distance_data = [] #(CLOSE, MED, FAR) (estimated)
        self.time_data = [] #(frames tracked)
        # self.radius_regions = dict

        self.data_dict = dict()
        self.previous_time = 0

        self.sex = tab.sex_line

        self.description = tab.desc_line
        self.record_state = False

        self.tracker = None

        # initialize the bounding box coordinates of the object we are going
        # to track
        self.init_bounding_box = None
        self.reset = False
        self.state_tracking = False
            

    def get_name(self):
        """ Returns name of person tracked: returns string """
        return self.name.text()

    # def get_loc(self, frame):
    #     """ Returns pixel (x,y) location of person tracked: returns tuple(int,int)"""
    #     return self.location_data(frame)
    
    def get_sex(self):
        """ Returns sex of person being tracked: returns string """
        return self.sex.text()

    def get_description(self):
        return self.description.getText()

    def get_time_tracked(self, framerate):
        """ Returns total time being tracked in video: returns  """ 
        total_time = [self.calculate_total_time(self.part_time_to_segments(list(self.data_dict.keys())), framerate)]
        return total_time

        
    def create(self, tracker_type='CSRT'):
        """
        The creation of the Opencv's Tracking mechanism.

        tracker_type = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        """
        # tracker_type = tracker_types[2]
    
        # if int(minor_ver) < 3:
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
            #prone to false-positives. I do not recommend using this OpenCV object tracker.
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
        while self.init_bounding_box[0] is 0 and self.init_bounding_box[1] is 0 and self.init_bounding_box[2] is 0 and self.init_bounding_box[3] is 0:
            print("No onject selected, select an object to continue")
            self.init_bounding_box = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
        print (self.init_bounding_box)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        if self.init_bounding_box is not None:
            # self.tracker.init(frame, self.init_bounding_box)
            print(self.reset)
            self.reset = True

        if self.reset is True:
            print("Resetting Location")
            del self.tracker
            self.create(tracker_type)
            self.tracker.init(frame, self.init_bounding_box)
       
        # self.fps = imutils.video.FPS().start()


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

    def record_data(self, frame, x, y):
        """
        Appends location and time to a list of data
        """
        # print("recording Frame" + str(frame))
        # self.location_data.append((int(x),int(y))) #(x, y)
        # self.time_data.append(frame) #(frames tracked)
        self.data_dict[frame] = (int(x),int(y))

        # self.distance_data.append(self.estimate_distance(size)) #(CLOSE, MED, FAR) (estimated)

    def append_data(self, dataframe):
        """
        Appends the data into a given dataframe categoriezed with the name given
        """
        pass

    def export_data(self, vid_width, vid_height, vid_name, fps):
        """
        Exports the recorded data and appends constants such as name, total time recorded, and pixel% Loc
        """
        # if not os.path.exists(("./data/" + vid_name[:-4])):
        #     os.makedirs(("./data/" + vid_name[:-4]))
        print(self.data_dict)

        export_filename = str(vid_name[:-4]) + ".csv"
        if not os.path.isfile(export_filename):
            data = {"Frame_Num":[],#self.time_data,
                "Pixel_Loc":[],
                "Perc_X":[], "Perc_Y":[],
                "Name":[], "Sex":[],"Total_Sec_Rec":[]}

            df = pd.DataFrame(data)
            export_csv = df.to_csv (export_filename, index = None, header=True, mode='a')

        frames = list(self.data_dict.keys())
        location = list(self.data_dict.values())
        #elaborate on the location, record it in percent
        perc_x_list = []
        perc_y_list = []
        for data in location:
            perc_x = (data[0]/vid_width)*100
            perc_y = (data[1]/vid_height)*100
            perc_x_list.append(round(perc_x,2))
            perc_y_list.append(round(perc_y,2))


        #extend all the data so it can be exported
        MAX_LEN = len(frames)
        sex = [self.get_sex()]
        name = [self.get_name()]
        # total_time = [self.(self.part_time_to_segments(self.time_data))]
        total_time = self.get_time_tracked(vid_fps)
        total_time[0] += self.previous_time
        self.previous_time = total_time[0]
        sex.extend([sex[0]]*(MAX_LEN-1))
        name.extend([name[0]]*(MAX_LEN-1))
        total_time.extend([total_time[0]]*(MAX_LEN-1))


        data = {"Frame_Num":frames,#self.time_data,
            "Pixel_Loc": location,
            "Perc_X": perc_x_list, "Perc_Y": perc_y_list,
            "Name": name, "Sex":sex, 
            "Total_Sec_Rec":total_time}
        
        df = pd.DataFrame(data)


        export_csv = df.to_csv (export_filename, index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path

        self.data_dict = dict()
        

    def part_time_to_segments(self, time_data, segment_size=300):
        """
        Parts the times into segments based on the distance between tracked frames.
            time_data is the entire time of the object tracked.
            segment_size is the distance between frames that should be considered as one segment. Default is 10 seconds at 30fps.
            min_segment is the minimum size of segment allowed. If it is too small, it will be ignored. Defualt is 1 second (30 frames).
        """
        #three segment markers, starting frame, current frame, and end frame
        seg_start = seg_last =  time_data[0]
        
        total_segments = [] # total segments placed in a list


        # iterate through all the times, start comparing time[0] with time[1]
        for time in time_data:
            # print(time)
            #if the size between last segment time and current time is less than segment threshold
            if ((time - seg_last) <= segment_size):
                # print("Stepping from " + str(seg_last) + " to " + str(time) )
                seg_last = time
            
            #if the size between last segment time is greater than the threshold, we create a new segment and check if it has been tracked long enough
            if ((time - seg_last) > segment_size):
                # print("Past Threshold, segmenting time from " + str(seg_start) + " to " + str(seg_last))
                if seg_start is not seg_last:
                    #add the start and end of current segment as a pair
                    total_segments.append((seg_start, seg_last))
                seg_start = time
                seg_last = time
            
            #if we reach the end of the tracked data, we end the segments and close the loop
            if (time == time_data[-1]):
                if seg_start == seg_last:
                    break
                total_segments.append((seg_start, seg_last))
                break
        
        return total_segments

    def estimate_distance(self, size):
        pass
            
    def calculate_time(self, frame_start, frame_end, fps=30):
        """
        Calculates time between two given frames and the FPS rate
        """
        return (frame_end - frame_start)/fps
    
    def calculate_total_time(self, total_frames, fps=30, segmented=True):
        """
        Calculates total time tracked. This takes into consideration segmented time
        """
        total_time = 0
        if segmented is False:
            time_segs = self.part_time_to_segments(total_frames)
        else: 
            time_segs = total_frames
            # print(time_segs)
        #sum all the segments together
        for seg in time_segs:
            # print("Timing Segment:",end="")
            # print(seg)
            #each seg is a pair of start and end times
            total_time += self.calculate_time(seg[0],seg[1],fps)
        return total_time

    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
class Regions():
    def __init__(self):
        self.radius_regions = dict()

    def add_radius(self, name):
        """
        Creates a circle given a rectangle ROI.
        """
        self.radius_regions[name] = (cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True))


    
    def display_radius(self, frame):
        """
        Displays all radius created Radius on given frame
        """
        for key, region in self.radius_regions.items():
            x, y, w, h = region[0], region[1], region[2], region[3]
            ellipse_center = (int(x + (w/2)) ,int( y + (h/2)))

            frame = cv2.ellipse(frame, ellipse_center, (int((w/2)),(int(h/2))), 0, 0,360, (0,255,0) )
            # cv2.ellipse(frame, box=w/2,color=(0,255,0))
            cv2.rectangle(frame, (x , y - 1), (x + 10 * (len(key)) , y - 15),(255,255,255),-1)
            cv2.putText(frame,key, (x , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)
        return frame

    def test_radius(self, test_point):
        """
        Tests weather a point value exists within a radius

        Equation: Given (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
            Given a test_point (x,y), ellipse_center at (h,k), 
            radius_width, radius_height as a,b respectivly
        """
        #overlapping areas may result in multiple True tests
        within_points = []
        x = test_point[0]
        y = test_point[1]
        
        for key, region in self.radius_regions.items():

            x, y, w, h = region[0], region[1], region[2], region[3]
            ellipse_center = (x + (w/2) , y - (h/2))

            # checking the equation of
            # ellipse with the given point
            p = ((math.pow((x - ellipse_center[0]), 2) // math.pow((w/2), 2)) + 
                (math.pow((y - ellipse_center[1]), 2) // math.pow((h/2), 2)))
            if p <= 1: #point exists in or on eclipse
                within_points.append(key)
        return None

    def handle_inputs():
        pass
#This main is used to test the time
if __name__ == "__main__":
    trackerName = 'CSRT'
    # vid_dir = "videos/"
    # vid_name = "GP074188.MP4"
    # vid_dir = "C:/Users/legom/Documents/GitHub/UofLStudy/"
    # vid_name = "Complex_Trim.mp4"
    # videoPath = vid_dir + vid_name
    
    # videoPath = "C:/Users/legom/Documents/GitHub/UofLStudy/Complex_Trim.mp4"
    app = QApplication(sys.argv)
    input_dialog = qt_dialog.App()

    videoPath = input_dialog.filename

    # meta_file = mutagen.File("videos/GP074188.MP4")
    # print(meta_file)
    
    tracker_list = []
    # initialize OpenCV's special multi-object tracker
    # for new_tracker in range(CPU_COUNT):
    input_dialog.add_tab()
    input_dialog.add_tab_state = False
    tracker_list.append(MultiTracker(input_dialog.tab_list[0]))

    regions = Regions()


    selected_tracker = 0
    cap = cv2.VideoCapture(videoPath)
    input_dialog.set_max_scrollbar(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    #get the video's FPS
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    input_dialog.set_fps_info(vid_fps)
    

    i = 0
    while cap.isOpened():
        app.processEvents()
        selected_tracker = input_dialog.tabs.currentIndex()

        #if the scrollbar is changed, update the frame, else continue with the normal frame
        if input_dialog.scrollbar_changed == True:
            i = input_dialog.get_scrollbar_value()
            input_dialog.scrollbar_changed = False
        else:
            input_dialog.set_scrollbar(i)

        if input_dialog.add_tab_state == True:
            print(len(tracker_list))
            input_dialog.tabs.setCurrentIndex(len(tracker_list))
            tracker_list.append(MultiTracker(input_dialog.tab_list[len(tracker_list)]))
            input_dialog.tabs.setEnabled(False)
            input_dialog.add_tab_btn.setEnabled(False)
            input_dialog.del_tab_btn.setEnabled(False)
            input_dialog.tabs.setEnabled(True)
            input_dialog.add_tab_state = False
        
        if input_dialog.play_state == True:
            i += input_dialog.get_frame_skip()

        cap.set(1 , i)
        ret, frame = cap.read()

        if frame is None:
            break

        input_dialog.set_tab_names()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("e") or input_dialog.export_state == True:
            input_dialog.export_state = False
            print("Exporting " + tracker_list[selected_tracker].get_name() + "'s data recorded.")
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
            try:
                tracker_list[selected_tracker].export_data(width, height, videoPath, vid_fps)
            except IOError as err:
                print(err)

        #remove the tracker that is currently selected
        elif input_dialog.del_tab_state is True and selected_tracker != -1:
            del tracker_list[selected_tracker]
            selected_tracker = input_dialog.tabs.currentIndex()
            input_dialog.del_tab_state = False
        elif key == ord("w"):
            print("Nudge Up")
        elif key == ord("a"):
            i -= input_dialog.get_frame_skip() * 2
            input_dialog.play_state = True
            input_dialog.mediaStateChanged(True)
        elif key == ord("s"):
            print("Nudge Down")
            regions.add_radius(input())
        elif key == ord("d"):
            print("Nudge Right")
        selected_tracker = input_dialog.tabs.currentIndex()

        #Set the selected Tracker to Red
        for tracker in range(len(tracker_list)):
            if tracker == selected_tracker:
                tracker_list[tracker].colour = (0,0,255)
            else:
                tracker_list[tracker].colour = (255,255,255)

        #Loop through every tracker and update
        for tracker in tracker_list:
            if tracker.init_bounding_box is not None:

                #allocate frames on GPU, reducing CPU load.
                cv2.UMat(frame)    

                #track and draw box on the frame
                success, box, frame = tracker.update_tracker(frame)
                
                # if not success:
                #     tracker.assign(frame, trackerName)

                #caluclate info needed this frame
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                bottom_right = box[0]
                top_left = box[1]
                width = box[2]
                height = box[3]

                center_x = bottom_right - (width/2)
                center_y = top_left + (height/2)
                
                regions.test_radius((center_x, center_y))
                if input_dialog.play_state == True:
                    #record all the data collected from that frame
                    tracker.record_data(frame_number, center_x, center_y)


        #If you select a tracker and it is not running, start a new one

        if selected_tracker >= 0 and len(tracker_list) > 0 and selected_tracker <= len(tracker_list):
            if tracker_list[selected_tracker].init_bounding_box is None:
                input_dialog.tabs.setEnabled(False)
                tracker_list[selected_tracker].create(trackerName)
                tracker_list[selected_tracker].assign(frame, trackerName)
                input_dialog.tabs.setEnabled(True)
                input_dialog.add_tab_btn.setEnabled(True)
                input_dialog.del_tab_btn.setEnabled(True)
            if key == ord(' '):
                input_dialog.play_state = False
                input_dialog.tabs.setEnabled(False)
                tracker_list[selected_tracker].assign(frame, trackerName)
                input_dialog.tabs.setEnabled(True)
            # input_dialog.play_state = True

        try:
            current_tracked_time = tracker_list[selected_tracker].get_time_tracked(vid_fps)[0] + tracker_list[selected_tracker].previous_time
            input_dialog.tab_list[selected_tracker].update_length_tracked(current_tracked_time)
        except:
            pass

        if len(regions.radius_regions) > 0:
            frame = regions.display_radius(frame)
            #When done processing each tracker, view the frame
        cv2.imshow("Frame", frame)

            
    sys.exit(app.exec_())
    cap.release()
    cv2.destroyAllWindows()