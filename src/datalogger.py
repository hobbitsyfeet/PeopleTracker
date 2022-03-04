
from itertools import count
from matplotlib.pyplot import pause
import time
import pandas as pd
import cv2
import numpy as np
import os

from evaluate import tracker_evaluation

# Constants for intervention Level
NO_INTVERVENTION_TRACKER = "NONE"
HUMAN_INTERVENTION = "USER"
HUMAN_INTERVENTION_KALMAN = "USER_KALMAN"
HUMAN_INTERVENTION_REGRESSION_KALMAN = "USER_KALMAN_REGRESSION" # NOTE: USE THIS ONLY?


HUMAN_INTERVENTION_MCRNN = "USER_MRCNN"
HUMAN_INTERVENTION_KALMAN_MCRNN = "USER_KALMAN_MRCNN"
HUMAN_INTERVENTION_MRCNN_REGRESSION = "USER_REGRESSION_MRCNN"
HUMAN_INTERVENTION_MRCNN_KALMAN_REGRESSION = "USER_KALMAN_REGRESSION_MRCNN" # NOTE: USE THIS ONLY?

NO_INTERVENTION_MODEL = "MRCNN"

class DataLogger:
    def __init__(self, video, video_metadata=None, intervention_level=NO_INTVERVENTION_TRACKER, ground_truth_folder=None):
        self.intervention_level = intervention_level
        self.ground_truth_folder = ground_truth_folder
        self.metadata = video_metadata
        self.video = video

        self.start_time_id = "START_STOP"

        self.previous_frame = None
        self.logging_dict = {}
        self.timer_dict = {}

        self.paused_data = {}
        self.slider_data = {}
        self.adjust_data = {}


        self.logger_df = pd.DataFrame(columns =['Frame_Number',
                                                'Event_Time',
                                                'Event_Duration',
                                                'Event_Type',
                                                'Event_Value',
                                                'Intervention_Level',
                                                'Intervention_Type',
                                                'Tracker_ID']) # NOTE: Tracker ID is one of either: Selected (And active Tracker), or tracker acted upon by additional methods.
                                                               # This means that all USER Intervention_Types, the tracker_ID is the current tracker
                                                               # When Kalman, Regressuion or MRCNN make changes, it does not need to be the current tracker selected. 

        self.video_info_df = pd.DataFrame(columns=['Frame_Number', 
                                                    'Mean_Illumination', 
                                                    'Std_Illumination', 
                                                    'Mean_Opticalflow', 
                                                    'Median_Opticalflow', 
                                                    'Std_Opticalflow',
                                                    'Ground_Truth_Count',
                                                    'Occluded_Ground_Truths_Count',
                                                    'Entering_Ground_Truths',
                                                    'Entering_Ground_Truths_Count',
                                                    'Exiting_Ground_Truths',
                                                    'Exiting_Ground_Truths_Count',
                                                    'Video_ID', 
                                                    'Video_Location'])
    ##################################
    # User input logging starts here #
    ##################################

    def start_recording(self):
        '''
        Starts the recording timer
        '''
        start_time = time.time()
        self.timer_dict[self.start_time_id] = (start_time, None)
        
    def end_recording(self):
        '''
        '''
        self.timer_dict[self.start_time_id][1] = time.time()

    def start_timer(self, timer_id):
        """
        Starts the timer from when the project is initiated
        """

        # We start recording when the first action is done
        if self.start_time_id not in self.timer_dict:
            self.start_recording()

        start_record = self.timer_dict[self.start_time_id][0]
        start_time = time.time() - start_record
        self.timer_dict[timer_id] = (start_time, None)
        return start_time

    def get_time_elapsed(self, timer_id):
        """
        returns elapsed time of a timer in seconds
        """
        return time.time() - self.timer_dict[self.start_time_id][0] - self.timer_dict[timer_id][0]


    def end_timer(self, timer_id):
        """
        Ends the timer from when the project is finished
        """
        duration = self.get_time_elapsed(timer_id)
        self.timer_dict[timer_id] = (self.timer_dict[timer_id][0], duration)

    def adjustment(self, frame_number, from_box, timer_id, tracker_id, intervention_type="USER", to_box=None):
        """
        Records the frame number and locations where the tracker is moved from and to.

        Records what made the adjustment.
            
            Human - If the user made the adjustment
            
            Model - If the MaskRCNN model made the automatic adjustment
        """

        time_started = self.start_timer(timer_id)
        data = [frame_number,
        time_started,
        0,
        "Assignment",
        str(from_box),
        self.intervention_level,
        intervention_type,
        tracker_id
        ]
        self.adjust_data[timer_id] = data

        if to_box:
            self.end_adjustment(to_box, timer_id)

        
        # print(self.logger_df)
    
    def end_adjustment(self, to_box, timer_id):
        '''
        Records the adjustment when final changes are made. This contributes final location and duration.
        '''
        duration = self.get_time_elapsed(timer_id)
        data = self.adjust_data[timer_id]
        data[2] = duration

        data[4] = str(data[4]) + str(to_box)
        self.logger_df.loc[self.logger_df.shape[0]] = data


    def paused(self, frame_number, pause_type, timer_id, tracker_id):
        """
        Records what initiated the pause, and for how long the pause existed for.

        This requires an external timer, and will be recorded when *Play* has been selected.
        """
        time_started = self.start_timer(timer_id)
        
        data = [
            frame_number,
            time_started,
            None,
            "Pause",
            None,
            self.intervention_level,
            pause_type,
            tracker_id
        ]
        self.paused_data[timer_id] = data
        # print(self.logger_df)
        
    
    def end_pause(self):
        if self.paused_data:
            for pause_ids in self.paused_data.keys():
                if pause_ids is not self.start_time_id:
                    # print(pause_ids)
                    self.end_timer(pause_ids)
                    duration = self.get_time_elapsed(pause_ids)
                    data = self.paused_data[pause_ids]
                    data[2] = duration
                    self.logger_df.loc[self.logger_df.shape[0]] = data
                
                # print(self.logger_df)
            self.paused_data = {}

    def slider_moved(self, frame_from, timer_id, tracker_id):
        """
        Records when the slider was changed, where from, and where to.
        """
        if timer_id not in self.timer_dict.keys():
            # print("Recording Slider")
            time_started = self.start_timer(timer_id)
            data = [
            frame_from,
            time_started,
            0,
            "Slider",
            None,
            self.intervention_level,
            "USER",
            tracker_id
            ]
            self.slider_data[timer_id] = data
            # print(self.logger_df)
    
    def end_slider(self, frame_to, timer_id):
        if self.slider_data:
            # time_started, timer_ended = self.timer_dict[timer_id]
            duration = self.get_time_elapsed(timer_id)
            data = self.slider_data[timer_id]
            data[2] = duration
            data[4] = frame_to
            self.logger_df.loc[self.logger_df.shape[0]] = data
            self.slider_data = {}
            
            self.end_timer(timer_id)
            del self.timer_dict[timer_id]

            # print(self.logger_df)
    
    def record_errors(self, frame):
        pass

    def get_tracker_score(self):
        pass
    
    ################################
    # VIDEO COMPLEXITY STARTS HERE #
    ################################

    def get_video_characteristics(self, video=None):
        """
        Records all video characteristics. 
            This will be done independently from the user input 
            and will be constants between intervention types
        """

        if video is None:
            video = self.video
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        folder = os.path.dirname(video)
        te = tracker_evaluation()
        te.load_json(self.ground_truth_folder)
        print("Done")

        if (cap.isOpened()== False): 
            print("Error opening video  file")
        
        previous_gt_frame = 0

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret == True:
                ill_mean, ill_std = self.illumination(frame)
                flow_mean, flow_median, flow_std = self.optical_flow(frame)

                gt_count = te.get_ground_truth_count(int(frame_num))
                occluded_count = te.get_occlusion_count(int(frame_num))

                new_trackers = None
                nt_count = None
                leaving_trackers = None
                leaving_count = None

                if te.ground_truth_exists(frame_num):
                    if te.ground_truth_exists(frame_num):
                        new_trackers, leaving_trackers = te.get_ground_truth_difference(previous_gt_frame, frame_num)
                        nt_count = len(new_trackers)
                        leaving_count = len(leaving_trackers)
                        print("NEW_TRACKER", new_trackers, nt_count, "  LEAVING_TRACKERS", leaving_trackers, leaving_count)
                    previous_gt_frame = frame_num
                    
                
                # print(ill_mean, ill_std, flow_mean, flow_median, flow_std, gt_count, occluded_count)
                data = [frame_num, ill_mean, ill_std, flow_mean, flow_median, flow_std, gt_count, occluded_count, new_trackers, nt_count, leaving_trackers, leaving_count,  "VIDEO_ID", "VIDEO_LOCATION"]
                self.video_info_df.loc[self.video_info_df.shape[0]] = data
                # # Display the resulting frame
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
            else:
                return self.video_info_df
        return self.video_info_df

    def illumination(self, frame):
        """
        Records the average intensity of the RGB image in greyscale with values between 0 (black) and 255 (white)
        
        Intensity = (R + G + B)/3
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        return mean[0][0], std[0][0]

    def optical_flow(self, frame):
        """
        Uses optical flow to measure the average magnitude of movement from the flow of pixels.

        Also records the resolution of the image
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            return 0, 0, 0

        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Resize to 1/4 to reduce complexity
        gray_prev = cv2.resize(gray_prev, (0,0), fx=0.25, fy=0.25) 
        gray_current = cv2.resize(gray_current, (0,0), fx=0.25, fy=0.25)

        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current, flow=None, pyr_scale=0.5, levels=6, winsize=50, iterations=3, poly_n=5, poly_sigma=1.2, flags=None)
        # flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mean = np.mean(flow)
        median = np.median(flow)
        std = np.std(flow)
        self.previous_frame = frame

        return mean, median ,std

    def objects_occluded(self, folder_path, fps):
        """
        Records weather the occlusion flag has occured on that frame. Records how many IDs have been occluded.
        """
        te = tracker_evaluation()
        te.load_json(folder_path, fps=fps)
        occlusion_dict = {}
        for frame in te.get_frame_count():
            occlusion_dict[frame] = te.count_occlusion(frame)

        return occlusion_dict


    
    def check_intervention_level(self):
        intervention = pd.unique(self.logger_df['Intervention_Type'])
        if "USER" in intervention:
            self.intervention_level = HUMAN_INTERVENTION

            if "KALMAN" in intervention or "REGRESSION" in intervention:
                self.intervention_level = HUMAN_INTERVENTION_REGRESSION_KALMAN
            
                if "MRCNN" in intervention:
                    self.intervention_level = HUMAN_INTERVENTION_MRCNN_KALMAN_REGRESSION


            if "MRCNN" in intervention:
                self.intervention_level = HUMAN_INTERVENTION_MCRNN

                if "KALMAN" in intervention or "REGRESSION":
                    self.intervention_level = HUMAN_INTERVENTION_MRCNN_KALMAN_REGRESSION
        
        # If there is no human intervention
        elif "MRCNN" in intervention:
            self.intervention_level = NO_INTERVENTION_MODEL
        
        else:
            self.intervention_level = NO_INTVERVENTION_TRACKER

        return self.intervention_level

            



    def export_charactoristics(self, file_path):
        print("Saving Characteristics")

        char_filename =  file_path + "CHARACTERISTICS " + ".csv"
        self.video_info_df.to_csv(char_filename, index=False)
        
            
        print("Saving complete")

    def export_activity(self, file_path):
        if self.logger_df.shape[0] > 0:
            logger_filename =   file_path + "_ACTIVITY_LOGGER_" + self.intervention_level + ".csv"

            intervention_level = self.check_intervention_level()
            print(intervention_level)

            self.logger_df = self.logger_df.assign(Intervention_Level=intervention_level)

            self.logger_df.to_csv(logger_filename, index=False)
            print("Exported")
        else:
            print("No Data Available")

    def plot_chars_df(self):
        pass

if __name__ == "__main__":
    # logger = DataLogger("L:/.shortcut-targets-by-id/1BBxJzDfQqUGSfvkILX9LLVkQKSuQcTji/Monkey Videos (for tracker)/Yes/MVI_2975.MOV")
    # logger = DataLogger("D:/GitHub/PeopleTracker/videos/(Simple) GP014125.MP4")
    logger = DataLogger("C:/Users/legom/Downloads/ChrisCran_Labels/GOPR3814.MP4")
    logger.ground_truth_folder = "C:/Users/legom/Downloads/ChrisCran_Labels/GOPR3814/"
    
    # logger = DataLogger("L:/.shortcut-targets-by-id/1MP4p63J_OlME1O2ysxy_aSATSfgtn850/Gallery Videos/Historical Videos/Nov 18/GP044104.MP4")
    print("LOGGING")
    logger.get_video_characteristics()
    # logger.export_activity("")
    logger.export_charactoristics("C:/Users/legom/Downloads/ChrisCran_Labels/")
    # logger.video_info_df.to_csv("K:/Github/PeopleTracker/VIDEO_CHARACTERISTICS_GALLERY.csv")
