
from matplotlib.pyplot import pause
import evaluate
import time
import pandas as pd
import cv2
import numpy as np

# Constants for intervention Level
NO_INTVERVENTION_TRACKER = "NONE"
HUMAN_INTERVENTION = "USER"
HUMAN_INTERVENTION_KALMAN = "USER_KALMAN"
HUMAN_INTERVENTION_MCRNN = "USER_MRCNN"
HUMAN_INTERVENTION_KALMAN_MCRNN = "USER_KALMAN_MRCNN"
NO_INTERVENTION_MODEL = "MRCNN"

class DataLogger:
    def __init__(self, video, video_metadata=None, intervention_level=NO_INTVERVENTION_TRACKER):
        self.intervention_level = intervention_level
        self.metadata = video_metadata
        self.video = video

        self.start_time_ID = "START_STOP"

        self.previous_frame = None
        self.logging_dict = {}
        self.timer_dict = {}

        self.paused_data = {}


        self.logger_df = pd.DataFrame(columns =['Frame_Number',
                                                'Event_Time',
                                                'Event_Duration',
                                                'Event_Type',
                                                'Event_Value',
                                                'Intervention_Level',
                                                'Intervention_Type'])

        self.video_info_df = pd.DataFrame(columns=['Frame_Number', 
                                                    'Mean_Illumination', 
                                                    'Std_Illumination', 
                                                    'Mean_Opticalflow', 
                                                    'Median_Opticalflow', 
                                                    'Std_Opticalflow', 
                                                    'Video_ID', 
                                                    'Video_Location'])
    ##################################
    # User input logging starts here #
    ##################################

    def start_timer(self, id):
        """
        Starts the timer from when the project is initiated
        """
        self.timer_dict[id] = (time.time(), None)

    def get_time_elapsed(self, id):
        """
        returns elapsed time of a timer in seconds
        """
        if self.timer_dict[id][1] is None:
            #if timer is not done, give current time
            return time.time() - self.timer_dict[id][0]
        else:
            # Get the start - end time if timer has ended
            return  self.timer_dict[id][1] - self.timer_dict[id][0]

    def end_timer(self, id):
        """
        Ends the timer from when the project is finished
        """
        self.timer_dict[id] = (self.timer_dict[id][0], time.time())

    def adjustment(self, frame_number, from_box, to_box, timer_id, intervention_type="Human"):
        """
        Records the frame number and locations where the tracker is moved from and to.

        Records what made the adjustment.
            
            Human - If the user made the adjustment
            
            Model - If the MaskRCNN model made the automatic adjustment
        """
        time_started, timer_ended = self.timer_dict[timer_id]
        duration = self.get_time_elapsed(timer_id)
        data = [frame_number,
        time_started,
        duration,
        "Assignment",
        str(from_box)+str(to_box),
        self.intervention_level,
        intervention_type
        ]
        self.logger_df.loc[self.logger_df.shape[0]] = data
        print(self.logger_df)

    def paused(self, frame_number, pause_type, timer_id):
        """
        Records what initiated the pause, and for how long the pause existed for.

        This requires an external timer, and will be recorded when *Play* has been selected.
        """
        self.start_timer(timer_id)
        time_started, timer_ended = self.timer_dict[timer_id]
        
        data = [
            frame_number,
            time_started,
            None,
            "Pause",
            None,
            self.intervention_level,
            pause_type
        ]
        self.paused_data[timer_id] = data
        print(self.logger_df)
        
    
    def end_pause(self):
        if self.paused_data:
            for pause_ids in self.paused_data.keys():
                print(pause_ids)
                self.end_timer(pause_ids)
                duration = self.get_time_elapsed(pause_ids)
                data = self.paused_data[pause_ids]
                data[2] = duration
                self.logger_df.loc[self.logger_df.shape[0]] = data
                self.paused_data = {}
                print(self.logger_df)

    def slider_moved(self, frame_from, frame_to, timer_id):
        """
        Records when the slider was changed, where from, and where to.
        """
        time_started, timer_ended = self.timer_dict[timer_id]
        duration = self.get_time_elapsed(timer_id)
        data = [
        frame_from,
        time_started,
        duration,
        "Pause",
        frame_to,
        self.intervention_level,
        "USER"
        ]
        self.logger_df.loc[self.logger_df.shape[0]] = data
        print(self.logger_df)
    
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

        if (cap.isOpened()== False): 
            print("Error opening video  file")
        
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret == True:
                ill_mean, ill_std = self.illumination(frame)
                flow_mean, flow_median, flow_std = self.optical_flow(frame)
                print(ill_mean, ill_std, flow_mean, flow_median, flow_std)
                data = [frame_num, ill_mean, ill_std, flow_mean, flow_median, flow_std, "VIDEO_ID", "VIDEO_LOCATION"]
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

    def objects_occluded(self, frame_number):
        """
        Records weather the occlusion flag has occured on that frame. Records how many IDs have been occluded.
        """
        pass

    def get_tracked_count(self, dataframe):
        """
        Records the number of ground truths exist in the frame.
        
        This reflects tracker complexity
        """
        pass

    def export_charactoristics(self, file_path):
        print("Saving Characteristics")
        char_filename =  file_path + "CHARACTERISTICS " + ".csv"
        self.video_info_df.to_csv(char_filename)
        
            
        print("Saving complete")

    def export_activity(self, file_path):
        if self.logger_df.shape[0] > 0:
            logger_filename =   file_path + "ACTIVITY_LOGGER_" + self.intervention_level + ".csv"
            self.video_info_df.to_csv(logger_filename)
            print("Exported")
        else:
            print("No Data Available")

    def plot_chars_df(self):
        pass

if __name__ == "__main__":
    logger = DataLogger("L:/.shortcut-targets-by-id/1BBxJzDfQqUGSfvkILX9LLVkQKSuQcTji/Monkey Videos (for tracker)/Yes/MVI_2975.MOV")
    # logger = DataLogger("L:/.shortcut-targets-by-id/1MP4p63J_OlME1O2ysxy_aSATSfgtn850/Gallery Videos/Historical Videos/Nov 18/GP044104.MP4")
    
    logger.get_video_characteristics()
    
    # logger.video_info_df.to_csv("K:/Github/PeopleTracker/VIDEO_CHARACTERISTICS_GALLERY.csv")
