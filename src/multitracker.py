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
import qt_dialog
import sys
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QWidget, QComboBox, QMessageBox


import math

import exiftool

from Video import FileVideoStream

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
        self.beginning = tab.get_beginning
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
        return self.description.toPlainText()

    def get_time_tracked(self, framerate):
        """ Returns total time being tracked in video: returns  """
        
        # if len(self.data_dict) > 2:
        total_time = [self.calculate_total_time(self.part_time_to_segments(list(self.data_dict.keys())), framerate)]
        return total_time

    def create(self, tracker_type='CSRT'):
        """
        The creation of the Opencv's Tracking mechanism.

        tracker_type = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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
            input_dialog.log("No onject selected, select an object to continue")
            self.init_bounding_box = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
        input_dialog.log(self.init_bounding_box)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        if self.init_bounding_box is not None:
            # self.tracker.init(frame, self.init_bounding_box)
            input_dialog.log(self.reset)
            self.reset = True

        if self.reset is True:
            input_dialog.log("Resetting Location")
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

    def record_data(self, frame, x, y, regions):
        """
        Appends location and time to a list of data
        """
        # input_dialog.log("recording Frame" + str(frame))
        # self.location_data.append((int(x),int(y))) #(x, y)
        # self.time_data.append(frame) #(frames tracked)
        point = (int(x),int(y))
        self.data_dict[frame] = (point, regions)

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
        input_dialog.log(self.data_dict)

        #Generate the base dataframe to fill and file
        export_filename = str(vid_name[:-4]) + ".csv"
        if not os.path.isfile(export_filename):
            data = {"Frame_Num":[],#self.time_data,
                "Pixel_Loc":[],
                "Perc_X":[], "Perc_Y":[],
                "Region": [],
                "Name":[], "Sex":[], "Total_Sec_Rec":[],
                "Description":[],
                "Present At Beginning":[],
                }

            df = pd.DataFrame(data)
            export_csv = df.to_csv (export_filename, index = None, header=True, mode='a')

        frames = list(self.data_dict.keys())
        location = list(self.data_dict.values())

        #elaborate on the location, record it in percent
        perc_x_list = []
        perc_y_list = []
        pixel_location = []
        region_list = []
        
        #Iterate through data and record
        for data in location:

            pixel_location.append(data[0])

            perc_x = (data[0][0]/vid_width)*100
            perc_y = (data[0][1]/vid_height)*100
            perc_x_list.append(round(perc_x,2))
            perc_y_list.append(round(perc_y,2))

            region_string = ""
            for text in data[1]:
                if text == data[1][-1]:
                    region_string += (text)
                else:
                    region_string += (text + ", ")
            region_list.append(region_string)

        input_dialog.log(len(frames))
        input_dialog.log(pixel_location)
        input_dialog.log(len(pixel_location))
        input_dialog.log(region_list)
        input_dialog.log(len(region_list))

        #extend all the data so it can be exported. All data should be of the same length.
        MAX_LEN = len(frames)
        sex = [self.get_sex()]
        name = [self.get_name()]
        description = [self.get_description()]
        beginning = [self.beginning()]
        # total_time = [self.(self.part_time_to_segments(self.time_data))]
        total_time = self.get_time_tracked(vid_fps)
        total_time[0] += self.previous_time
        self.previous_time = total_time[0]
        sex.extend([sex[0]]*(MAX_LEN-1))
        name.extend([name[0]]*(MAX_LEN-1))
        description.extend([description[0]]*(MAX_LEN-1))
        beginning.extend([beginning[0]]*(MAX_LEN-1))
        total_time.extend([total_time[0]]*(MAX_LEN-1))

        
        #Create the dataframe
        data = {"Frame_Num":frames,#self.time_data,
            "Pixel_Loc": pixel_location,
            "Perc_X": perc_x_list, "Perc_Y": perc_y_list,
            "Region": region_list,
            # "TimeInRegion":,
            "Name": name, "Sex":sex, 
            "Total_Sec_Rec":total_time,
            "Description":description,
            "Beginning":beginning,
            }
        
        df = pd.DataFrame(data)
        export_csv = df.to_csv (export_filename, index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path

        self.data_dict = dict()

        input_dialog.log("Export Data Complete!")
        

    def part_time_to_segments(self, time_data, segment_size=300):
        """
        Parts the times into segments based on the distance between tracked frames.
            time_data is the entire time of the object tracked.
            segment_size is the distance between frames that should be considered as one segment. Default is 10 seconds at 30fps.
            min_segment is the minimum size of segment allowed. If it is too small, it will be ignored. Defualt is 1 second (30 frames).
        """
        #three segment markers, starting frame, current frame, and end frame
        seg_start =  time_data[0]
        seg_last = time_data[0]
        
        total_segments = [] # total segments placed in a list


        # iterate through all the times, start comparing time[0] with time[1]
        for time in time_data:
            # input_dialog.log(time)
            #if the size between last segment time and current time is less than segment threshold
            if ((time - seg_last) <= segment_size):
                # input_dialog.log("Stepping from " + str(seg_last) + " to " + str(time) )
                seg_last = time
            
            #if the size between last segment time is greater than the threshold, we create a new segment and check if it has been tracked long enough
            if ((time - seg_last) > segment_size):
                # input_dialog.log("Past Threshold, segmenting time from " + str(seg_start) + " to " + str(seg_last))
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
            # input_dialog.log(time_segs)
        #sum all the segments together
        for seg in time_segs:
            # input_dialog.log("Timing Segment:",end="")
            # input_dialog.log(seg)
            #each seg is a pair of start and end times
            total_time += self.calculate_time(seg[0],seg[1],fps)
        return total_time

    '''
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    '''


class Regions(QWidget):

    def __init__(self):
        super().__init__()
        self.radius_regions = dict()

    def add_radius(self):
        """
        Creates a circle given a rectangle ROI.
        """
        name, okPressed = QInputDialog.getText(self, 'Region', 'Region Name:')
        if okPressed and name != '':
            input_dialog.log(name)
            self.radius_regions[name] = (cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True))

    def del_radius(self):
        items = (self.radius_regions.keys())
        # items = ("Red","Blue","Green")
        item, okPressed = QInputDialog.getItem(self, "Get item","Delete Regions:", items, 0, False)
        if okPressed and item:
            input_dialog.log(item)
            del self.radius_regions[item]
        # name, okPressed = QInputDialog.getText(self, 'Region', 'Delete Region Name:')
        
        # combo_box = QComboBox(self)
        # for item in items:
        #     combo_box.addItem(item)
        # combo_box.move(50, 250)
        # combo_box.showPopup() 
        # selected = combo_box.activated[str]
                # creating a combo box widget 

  
        # adding action to the button 
        # button.pressed.connect(self.action) 
  

        
        # comboBox.activated[str].connect(lambda parameter_list: expression)
        # del self.radius_regions[selected]
        # item, okPressed = QInputDialog.getItem(self.parent, "Get item","Region Name", items, 0, False)
        # if okPressed and item:
        #     input_dialog.log(item)
    
    def display_radius(self, frame):
        """
        Displays all radius created Radius on given frame
        """
        for key, region in self.radius_regions.items():
            x, y, w, h = region[0], region[1], region[2], region[3]
            ellipse_center = (int(x + (w/2)) ,int( y + (h/2)))

            frame = cv2.ellipse(frame, ellipse_center, (int((w/2)),(int(h/2))), 0, 0,360, (0,255,0) )
            # cv2.ellipse(frame, box=w/2,color=(0,255,0))
            cv2.rectangle(frame, (x + int(w/2.1) , y - 1), (x + int(w/2.1) + 10 * (len(key)) , y - 15),(255,255,255),-1)
            cv2.putText(frame,key, (x + int(w/2.1) , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)
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
        test_x = test_point[0]
        test_y = test_point[1]
        
        for key, region in self.radius_regions.items():

            x, y, w, h = region[0], region[1], region[2], region[3]
            ellipse_center = (x + (w/2) , y + (h/2))

                # checking the equation of
                # ellipse with the given point
            p = ((math.pow((test_x - ellipse_center[0]), 2) / math.pow((w/2), 2)) + 
                (math.pow((test_y - ellipse_center[1]), 2) / math.pow((h/2), 2)))

            if p <= 1: #point exists in or on eclipse
                within_points.append(key)
        return within_points

    def handle_inputs():
        pass

def export_meta(vid_dir):
    """
    Exports a start line into a CSV with the same name as the video. This CSV will initialize with a set of columns on the left
    with dashed lines for recorded data, and a set of columns on the right with Metadata.
    This should only be called once, and will only initialize if the file does not exist- will not overwrite.

    Input: vid_dir: directory of filename (string)
    """

    #get metadata from filepath
    #NOTE: Exiftool must be in path to access, otherwise pass a string to the path.
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(vid_dir)
    
    #Parse date data into [YYYY MM DD HH MM SS]
    date_created = metadata['QuickTime:CreateDate'].replace(":"," ")
    date = list(map(int,date_created.split()))
    
    #Create path string for exporting the data. It's just a change of extention.
    export_filename = str(vid_dir[:-4]) + ".csv"
    
    #Create the first 2 rows of data with title and then info. Recorded info should have '-' or 'N/A'
    if not os.path.isfile(export_filename):
        data ={
            "Frame_Num":['-'],#self.time_data,
            "Pixel_Loc":['-'],
            "Perc_X":['-'], "Perc_Y":['-'],
            "Region": ['-'],
            # "TimeInRegion":['-'],
            "Name":['-'], "Sex":['-'], "Total_Sec_Rec":['-'],
            "Description":['-'],
            "Present At Beginning":['-'],

            "FileName": metadata['File:FileName'],
            "FileType": metadata['File:FileType'],
            "CreateDate(YYYY:MM:DD HH:MM:SS)": metadata['QuickTime:CreateDate'],
            "CreateYear": date[0], "CreateMonth": date[1], "CreateDay": date[2],
            "CreateHour": date[3], "CreateMinute": date[4], "CreateSecond": date[5],

            "Width(px)": metadata['QuickTime:ImageWidth'],
            "Height(px)": metadata['QuickTime:ImageHeight'],
            "FrameRate": metadata['QuickTime:VideoFrameRate'],
            "HandlerDescription": metadata['QuickTime:HandlerDescription']
        }
        input_dialog.log(data)

        #Create a dataframe and then export it.
        df = pd.DataFrame(data,index=[1])
        export_csv = df.to_csv (export_filename, index = None, header=True)

        input_dialog.log("Export Metadata Complete!")
        return True


        
#This main is used to test the time
if __name__ == "__main__":
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

    

    # initialize OpenCV's special multi-object tracker
    # input_dialog.add_tab()
    # input_dialog.add_tab_state = False
    # tracker_list.append(MultiTracker(input_dialog.tab_list[0]))
    selected_tracker = 1

    #initialize regions object to store all regions
    regions = Regions()
    
    #Initialize video, get the first frame and setup the scrollbar to the video length
    cap = cv2.VideoCapture(videoPath)
    input_dialog.set_max_scrollbar(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_dialog.resolution_x);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_dialog.resolution_y);
    fvs = FileVideoStream(videoPath).start()

    # ret, frame = cap.read()

    frame = fvs.read()
    # frame = imutils.resize(frame, width=450)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.dstack([frame, frame, frame])

    frame = cv2.resize(frame, (input_dialog.resolution_x, input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)
    
    previous_frame = frame
    #get the video's FPS
    # vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_fps = 30
    input_dialog.set_fps_info(vid_fps)
    
    
    skip_frame = 10
    
    input_dialog.log("Gathering frames...")
    
    while True:

        previous_skip = fvs.skip_value
        next_skip = input_dialog.get_frame_skip()
        if fvs.skip_value != input_dialog.get_frame_skip():
            fvs.next = input_dialog.get_scrollbar_value()
            fvs.reset = True
            frame = fvs.read()
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

        #if the scrollbar is changed, update the frame, else continue with the normal frame
        if input_dialog.scrollbar_changed == True:
            # input_dialog.log("Scrolled.")
            # skip_frame = input_dialog.get_scrollbar_value()
            
            fvs.next = input_dialog.get_scrollbar_value()
            
            fvs.reset = True
            # input_dialog.set_scrollbar(fvs.next)
            frame = fvs.read()
            
            # frame = fvs.read()
            previous_frame = frame
            input_dialog.scrollbar_changed = False
            
        # else:
        # #     # input_dialog.set_scrollbar(skip_frame)
        #     input_dialog.set_scrollbar(fvs.frame_number)
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
        
        if input_dialog.snap_state is not None and tracker_list[selected_tracker].data_dict:
            #grabs frames that tracker was tracked
            # frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_number = fvs.frame_number
            frames_tracked = tracker_list[selected_tracker].data_dict
            closest_frame = frames_tracked.get(frame_number, frames_tracked[min(frames_tracked.keys(), key=lambda k: abs(k-frame_number))])
            print(closest_frame)
            if frame_number - closest_frame[0][0] > frame_number-closest_frame[0][1]:
                closest_frame = closest_frame[0][0]
            else:
                closest_frame = closest_frame[0][1]
            print(type(closest_frame))
            input_dialog.log("Closest: " + str(closest_frame))
            # cap.set(1 , closest_frame)
            input_dialog.play_state = False
            input_dialog.snap_state = None
        #if playing, increment frames by skip_frame count
        if input_dialog.play_state == True:
            input_dialog.set_scrollbar(fvs.frame_number)
            # fvs.stopped = False
            frame = fvs.read()
            previous_frame = frame
            # skip_frame += input_dialog.get_frame_skip()

            
        else:
            frame = previous_frame
            input_dialog.set_scrollbar(fvs.frame_number)



        #set tne next frame to skip frame and read it
        # cap.set(1 , skip_frame)
        try:
            # ret, frame = cap.read()
            frame = cv2.resize(frame, (input_dialog.resolution_x, input_dialog.resolution_y), 0, 0, cv2.INTER_CUBIC)
        except:
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
                tracker_list[selected_tracker].export_data(width, height, videoPath, vid_fps)
            except IOError as err:
                input_dialog.log(err)

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
            regions.add_radius()
            input_dialog.region_state = False
            input_dialog.log("Adding region complete.")
        elif input_dialog.del_region_state is True:
            input_dialog.log("Select a region to remove...")
            regions.del_radius()
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

            if tracker.init_bounding_box is not None and input_dialog.tab_list[tracker_num].active is True and input_dialog.tab_list[tracker_num].read_only is False:

                #allocate frames on GPU, reducing CPU load.
                cv2.UMat(frame)    

                app.processEvents()
                #track and draw box on the frame
                success, box, frame = tracker.update_tracker(frame)
                app.processEvents()
                
                #NOTE: this can be activated if you want to pause the program when trakcer fails
                # if not success:
                #     tracker.assign(frame, trackerName)

                #caluclate info needed this frame
                # frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                frame_number = fvs.frame_number
                bottom_right = box[0]
                top_left = box[1]
                width = box[2]
                height = box[3]

                center_x = bottom_right + (width/2)
                center_y = top_left + (height/2)
                
                #center dot
                cv2.circle(frame, (int(center_x),int(center_y)),2,(0,0,255),-1)

                in_region = regions.test_radius((center_x, center_y))
                
                if input_dialog.play_state == True:
                    #record all the data collected from that frame
                    tracker.record_data(frame_number, center_x, center_y, in_region)

            elif input_dialog.tab_list[tracker_num].read_only is True:
                #if read only, display the center
                # frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                frame_number = fvs.frame_number

                if frame_number in tracker.data_dict:
                    # If key exists in data
                    center, _ = tracker.data_dict[frame_number]

                    if selected_tracker == tracker_num:
                        #center dot
                        cv2.circle(frame, (int(center[0]),int(center[1])),2,(0,255,0),-1)
                    else: 
                        cv2.circle(frame, (int(center[0]),int(center[1])),2,(0,0,255),-1)


                
            app.processEvents()

        #If you select a tracker and it is not running, start a new one
        if selected_tracker >= 0 and len(tracker_list) > 0 and selected_tracker <= len(tracker_list):
            #If there is no assigned trakcer on selected individual, start one and not allow action until done
            if tracker_list[selected_tracker].init_bounding_box is None:
                input_dialog.tabs.setEnabled(False)
                tracker_list[selected_tracker].create(trackerName)
                tracker_list[selected_tracker].assign(frame, trackerName)
                input_dialog.tabs.setEnabled(True)
                input_dialog.add_tab_btn.setEnabled(True)
                input_dialog.del_tab_btn.setEnabled(True)
                input_dialog.export_tab_btn.setEnabled(True)
            #Press space bar to re-assign
            if key == ord(' ') or input_dialog.set_tracker_state is True:
                input_dialog.play_state = False
                input_dialog.tabs.setEnabled(False)
                tracker_list[selected_tracker].assign(frame, trackerName)
                input_dialog.tabs.setEnabled(True)
                input_dialog.set_tracker_state = False
    
            # input_dialog.play_state = True

        #NOTE this is in try-catch because initially there are not enough frames to calculate time. 
        #This could be done with if statement, though I havent found a way...
        try:
            current_tracked_time = tracker_list[selected_tracker].get_time_tracked(vid_fps)[0] + tracker_list[selected_tracker].previous_time
            input_dialog.tab_list[selected_tracker].update_length_tracked(current_tracked_time)
        except:
            pass
        
        #Display all regions on screen if they exist
        if len(regions.radius_regions) > 0:
            frame = regions.display_radius(frame)

        #When done processing each tracker, view the frame
        cv2.imshow("Frame", frame)

    #Close all applications.
    sys.exit(app.exec_())
    cap.release()
    fvs.stop()
    cv2.destroyAllWindows()