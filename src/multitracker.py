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

# For extracting video metadata
# import mutagen

class MultiTracker():
    def __init__(self, name="Person", colour=(255,255,255)):
        self.name = name
        self.colour = colour

        self.location_data = [] #(x, y)
        self.distance_data = [] #(CLOSE, MED, FAR) (estimated)
        self.time_data = [] #(frames tracked)

        self.sex = "N/A"

        self.record_state = False

        self.tracker = None

        # initialize the bounding box coordinates of the object we are going
        # to track
        self.init_bounding_box = None
        self.reset = False
        self.state_tracking = False

        def get_name(self):
            """ Returns name of person tracked: returns string """
            return self.name

        def get_loc(self, frame):
            """ Returns pixel (x,y) location of person tracked: returns tuple(int,int)"""
            return self.location_data(frame)
        
        def get_sex(self):
            """ Returns sex of person being tracked: returns string """
            return self.sex

        def get_time_tracked(self):
            """ Returns total time being tracked in video: returns  """ 
            total_time = [self.calculate_total_time(self.part_time_to_segments(self.time_data))]
            return total_time

        
    def create(self,tracker_type='CSRT'):
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
            self.tracker.init(frame, self.init_bounding_box)
            print(self.reset)
            self.reset = True
            root = tkinter.Tk()

            Mbox = mbox.MessageBox
            Mbox.root = root

            user = {}
            accept = False
            while accept is False:
                # mbox.mbox('starting in 1 second...', t=1)
                user['name'] = mbox.mbox('Enter Name or ID?', entry=True)
                if user['name']:
                    user['sex'] = mbox.mbox('male or female?', ('male', 'm'), ('female', 'f'))
                    accept = mbox.mbox(user, frame=False)
                # root.mainloop()
                self.name = user['name']
                self.sex = user['sex']
            root.withdraw()

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

        return success, box, frame

    def remove(self):
        """
        Removes the tracker from being recorded. DOES NOT YET DELETE INFO
        """
        remove_check = mbox.mbox('Delete and Remove Tracker?', ('Yes', 'y'), ('No', 'n'))
        if remove_check == 'n':
            pass
        remove_check_2 = mbox.mbox('Are you REALLLLY Sure?!?', ('Fuck it.', 'y'), ('No', 'n'))
        if remove_check == 'y' and remove_check_2 == 'y':
            del self.tracker
            self.init_bounding_box = None


    def predict(self,):
        """
        Applies KNN taking into consideration a Hidden Markov Model
        """
        pass

    def record_data(self, frame, x, y):
        """
        Appends location and time to a list of data
        """
        self.location_data.append((int(x),int(y))) #(x, y)
        self.time_data.append(frame) #(frames tracked)
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
        if not os.path.exists(("./data/" + vid_name[:-4])):
            os.makedirs(("./data/" + vid_name[:-4]))
        
        export_filename = "./data/" + str(vid_name[:-4]) + "/" + self.name + ".csv"

        #elaborate on the location, record it in percent
        perc_x_list = []
        perc_y_list = []
        for data in self.location_data:
            perc_x = (data[0]/vid_width)*100
            perc_y = (data[1]/vid_height)*100
            perc_x_list.append(round(perc_x,2))
            perc_y_list.append(round(perc_y,2))


        #extend all the data so it can be exported
        MAX_LEN = len(self.time_data)
        sex = [self.sex]
        name = [self.name]
        total_time = [self.f(self.part_time_to_segments(self.time_data))]
        
        sex.extend([sex[0]]*(MAX_LEN-1))
        name.extend([name[0]]*(MAX_LEN-1))
        total_time.extend([total_time[0]]*(MAX_LEN-1))


        data = {"Frame_Num":self.time_data,
            "Pixel_Loc": self.location_data,
            "Perc_X": perc_x_list, "Perc_Y": perc_y_list,
            "Name": name, "Sex":sex, "Total_Sec_Rec":total_time}
        
        df = pd.DataFrame(data)
        export_csv = df.to_csv (export_filename, index = None, header=True) #Don't forget to add '.csv' at the end of the path
        

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
        return (frame_end - frame_start)/fps
    
    def calculate_total_time(self, total_frames, fps=30, segmented=True):
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
            total_time += self.calculate_time(seg[0],seg[1])
        return total_time

#This main is used to test the time
if __name__ == "__main__":
    trackerName = 'BOOSTING'
    vid_dir = "videos/"
    vid_name = "GP074188.MP4"
    videoPath = vid_dir + vid_name
  
    # meta_file = mutagen.File("videos/GP074188.MP4")
    # print(meta_file)
    
    tracker_list = []
    # initialize OpenCV's special multi-object tracker
    for new_tracker in range(CPU_COUNT):
        tracker_list.append(MultiTracker())

    
    selected_tracker = 0
    

    cap = cv2.VideoCapture(videoPath)
    #get the video's FPS
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while cap.isOpened():
        i += 10
        cap.set(1 , i)
        ret, frame = cap.read()
        

        if frame is None:
            break
        frame = imutils.resize(frame, width=720)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            selected_tracker = 0
        elif key == ord("2"):
            selected_tracker = 1
        elif key == ord("3"):
            selected_tracker = 2
        elif key == ord("4"):
            selected_tracker = 3
        elif key == ord("5"):
            selected_tracker = 4
        elif key == ord("6"):
            selected_tracker = 5
        elif key == ord("7"):
            selected_tracker = 6
        elif key == ord(' '):
            tracker_list[selected_tracker].assign(frame,trackerName)
        elif key == ord("e"):
            print("Exporting " + tracker_list[selected_tracker].name + "'s data recorded.")
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
            try:
                tracker_list[selected_tracker].export_data(width,height,videoPath,vid_fps)
            except IOError as err:
                print(err)
        elif key == ord("r"):
            #remove the tracker that is currently selected
            tracker_list[selected_tracker].remove()
            #change selected tracker to the first tracker that is still tracking
            for tracker in range(len(tracker_list)):
                if tracker_list[tracker].init_bounding_box is not None:
                    selected_tracker = tracker
        elif key == ord("w"):
            print("Nudge Up")
        elif key == ord("a"):
            print("Nudge Left")
        elif key == ord("s"):
            print("Nudge Down")
        elif key == ord("d"):
            print("Nudge Right")

        #Set the selected Tracker to Red
        for tracker in range(len(tracker_list)):
            if tracker == selected_tracker:
                tracker_list[tracker].colour = (0,0,255)
            else:
                tracker_list[tracker].colour = (255,255,255)
        
        #If you select a tracker and it is not running, start a new one
        if tracker_list[selected_tracker].init_bounding_box is None:
            tracker_list[selected_tracker].create(trackerName)
            tracker_list[selected_tracker].assign(frame,trackerName)
        
        #Loop through every tracker and update
        for tracker in tracker_list:

            if tracker.init_bounding_box is not None:

                #attempt to run it on GPU
                cv2.UMat(frame)    

                #track and draw box on the frame
                success, box, frame = tracker.update_tracker(frame)

                #Return count to 0 when max is reached
                if selected_tracker > CPU_COUNT:
                    selected_tracker = CPU_COUNT

                #caluclate info needed this frame
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                bottom_right = box[0]
                top_left = box[1]
                width = box[2]
                height = box[3]

                center_x = bottom_right - (width/2)
                center_y = top_left + (height/2)
                
                #record all the data collected from that frame
                tracker.record_data(frame_number, center_x, center_y)

            #When done processing each tracker, view the frame
            cv2.imshow("Frame", frame)
        # count += 1

        # # loop over the bounding boxes and draw them on the frame
        # for box in boxes:
        #     (x, y, w, h) = [int(v) for v in box]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # if the 's' key is selected, we are going to "select" a bounding
        # # box to track
        # if key == ord("s"):
        #     colors = []
        #     # select the bounding box of the object we want to track (make
        #     # sure you press ENTER or SPACE after selecting the ROI)
        #     box = cv2.selectROIs("Frame", frame, fromCenter=False,
        #                         showCrosshair=True)
        #     box = tuple(map(tuple, box)) 
        #     for bb in box:
        #         tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
        #         trackers.add(tracker, frame, bb)

        # # if you want to reset bounding box, select the 'r' key 
        # elif key == ord("r"):
        #     trackers.clear()
        #     trackers = cv2.MultiTracker_create()

        #     box = cv2.selectROIs("Frame", frame, fromCenter=False,
        #                         showCrosshair=True)
        #     box = tuple(map(tuple, box))
        #     for bb in box:
        #         tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
        #         trackers.add(tracker, frame, bb)

        # elif key == ord("q"):
        #     break
    cap.release()
    cv2.destroyAllWindows()