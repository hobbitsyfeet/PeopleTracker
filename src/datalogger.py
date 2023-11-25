import time
import pandas as pd
import cv2
import numpy as np
import os

# print("Datalogger import 1")
import evaluate
# print("Datalogger import 2")

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
    def __init__(self, video, video_location=None, video_metadata=None, intervention_level=NO_INTVERVENTION_TRACKER, ground_truth_folder=None):
        
        live_video = False
        #assume number is live camera
        if type(video) == type(0):
            live_video = True
        
        ## String representing constants 
        self.intervention_level = intervention_level
        self.ground_truth_folder = ground_truth_folder
        self.te = evaluate.tracker_evaluation()
        if not live_video:
            self.te.fps = float(video_metadata['QuickTime:VideoFrameRate'])
        else: 
            self.te.fps = 30

        if self.ground_truth_folder is None and not live_video:
            #Check the same folder and see if it works
            print("Checking to see if ground truths are in the same folder as video...")
            self.ground_truth_folder = os.path.dirname(video) + "/"
            self.te.load_json(self.ground_truth_folder, fps=int(self.te.fps))
            
            if not self.te.ground_truth_dict:
                print("Checking Adjacent folder with the same name...")
                self.ground_truth_folder = os.path.dirname(video) + "/" + os.path.basename(video)[:-4] + "/"
                self.te.load_json(self.ground_truth_folder, fps=int(self.te.fps))
        
        if not self.te.ground_truth_dict:
            print("Could not find ground truths")

        self.metadata = video_metadata
        self.video = video
        
        if not live_video:
            self.video_id = os.path.basename(video)[:-4]
        else:
            self.video_id = -1

        self.video_location = video_location

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

        self.video_info_df = pd.DataFrame(columns=[ 'Frame_Number', 
                                                    'Mean_Illumination', 
                                                    'Std_Illumination', 

                                                    'Mean_Opticalflow', 
                                                    'Median_Opticalflow',
                                                    'Mean_Magnitude_Opticalflow',
                                                    'Std_Opticalflow',

                                                    'Zoom_Opticalflow',
                                                    'Zoom_Dominance_Percent',
                                                    'Zoom_Dominance_Threshold',
                                                    'Zoom_Mean_Magnitude_Threshold',

                                                    'Pan_Opticalflow',
                                                    'Pan_Dominance_Percent',
                                                    'Pan_Dominance_Threshold',
                                                    'Pan_Mean_Magnitude_Threshold',
                                                    
                                                    'Ground_Truth_Count',
                                                    'Occluded_Ground_Truths_Count',
                                                    'Entering_Ground_Truths',
                                                    'Entering_Ground_Truths_Count',
                                                    'Exiting_Ground_Truths',
                                                    'Exiting_Ground_Truths_Count',
                                                    'Ground_Truth_Labels',
                                                    'Ground_Truth_Areas',
                                                    'Ground_Truth_Heights',
                                                    'Ground_Truth_Widths',

                                                    'Video_ID', 
                                                    'Video_Location'])
    ##################################
    # User input logging starts here #
    ##################################

     #starts the recording timer
    def start_recording(self):
       
        '''
        Starts the recording timer
        '''
        start_time = time.time()
        self.timer_dict[self.start_time_id] = (start_time, None)
    
    ##
    # ends recording timer 
    def end_recording(self):
        '''
        Ends recording by setting the second value in the tuple (start, end) to the current time.
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
        '''
        After paused() is called, end_pause is used to end the paused action and add the duration to the logger.
        '''
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
        '''
        When slider_moved() is called, we call end_slider which removes a duration timer and adds the duration and SLIDER action to the logger.
        '''
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
        '''
        Not implemented
        '''
        pass

    def get_tracker_score(self):
        '''
        Not implemented
        '''
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
        te_loaded = True

        folder = os.path.dirname(video)
        print("Done")

        if (cap.isOpened()== False): 
            print("Error opening video  file")
        
        previous_gt_frame = 0
        gt_count = None
        occluded_count = None
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Read until video is completed
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            while not ret:
                print(ret, frame_num)
                frame_num += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if frame_num >= total_frames:
                    break
            
            if ret == True:
                ill_mean, ill_std = self.illumination(frame)
                of_dict = self.optical_flow(frame) # Mean Median STD Mean_Magnitude Dominant_Zoom Percent_Zoom 
                                                             # Zoom_Dominance_Thresh Zoom_Magnitude_Thresh Dominant_Pan Percent_Pan 
                                                             # Pan_Dominance_Thresh Pan_Magnitude_Thresh
                print(of_dict)
                if te_loaded:
                    gt_count = self.te.get_ground_truth_count(int(frame_num))
                    occluded_count = self.te.get_occlusion_count(int(frame_num))

                    

                new_trackers = None
                nt_count = None
                leaving_trackers = None
                leaving_count = None

                labels = None
                areas = None
                heights = None
                widths = None

                if self.te.ground_truth_exists(frame_num) and te_loaded:
                    if self.te.ground_truth_exists(frame_num):
                        new_trackers, leaving_trackers = self.te.get_ground_truth_difference(previous_gt_frame, frame_num)
                        nt_count = len(new_trackers)
                        leaving_count = len(leaving_trackers)
                        labels, areas, heights, widths = self.ground_truth_characteristics(frame_num)
                        

                    previous_gt_frame = frame_num
                    
                # print(ill_mean, ill_std, flow_mean, flow_median, flow_std, gt_count, occluded_count)
                data = [frame_num, ill_mean, ill_std, of_dict["Mean"], of_dict["Median"], 
                        of_dict["STD"], of_dict["Mean_Magnitude"], of_dict["Dominant_Zoom"], 
                        of_dict["Percent_Zoom"], of_dict["Zoom_Dominance_Thresh"], of_dict["Zoom_Magnitude_Thresh"],
                        of_dict["Dominant_Pan"], of_dict["Percent_Pan"], of_dict["Pan_Dominance_Thresh"], of_dict["Pan_Magnitude_Thresh"],
                        gt_count, occluded_count, new_trackers, nt_count, leaving_trackers, leaving_count, labels, areas, heights, widths,  
                        self.video_id, self.video_location
                ]

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

        Returns values in a dictionary with names:
            Mean
            Median
            STD
            Mean_Magnitude
            Dominant_Zoom
            Percent_Zoom
            Zoom_Dominance_Thresh
            Zoom_Magnitude_Thresh
            Dominant_Pan
            Percent_Pan
            Pan_Dominance_Thresh
            Pan_Magnitude_Thresh

        [66] Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Image Analysis, pages 363–370. Springer, 2003. 
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            
            return {"Mean":0, "Median": 0, "STD":0, "Mean_Magnitude":0, "Dominant_Zoom":0, "Percent_Zoom":0, "Zoom_Dominance_Thresh":0, "Zoom_Magnitude_Thresh": 0,"Dominant_Pan": 0, "Percent_Pan": 0, "Pan_Dominance_Thresh": 0, "Pan_Magnitude_Thresh":0}

        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Resize to 1/4 to reduce complexity
        gray_prev = cv2.resize(gray_prev, (0,0), fx=0.25, fy=0.25) 
        gray_current = cv2.resize(gray_current, (0,0), fx=0.25, fy=0.25)

        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current, flow=None, pyr_scale=0.5, levels=6, winsize=50, iterations=3, poly_n=5, poly_sigma=1.2, flags=None)

        mean_magnitude = np.mean(cv2.cartToPolar(flow[..., 0], flow[..., 1])[0])

        zoom_dom_direction, zoom_perc_dom, zoom_dom_thresh, zoom_mean_mag_thresh  =  self.optical_flow_zoom(flow)
        pan_dom_direction, pan_perc_dom, pan_dom_thresh, pan_mean_mag_thresh =  self.optical_flow_pan(flow)
        # flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mean = np.mean(flow)
        median = np.median(flow)
        std = np.std(flow)
        self.previous_frame = frame

        opticalflow_dict = {"Mean":mean, "Median": median, "STD":std, "Mean_Magnitude":mean_magnitude, "Dominant_Zoom":zoom_dom_direction, "Percent_Zoom":zoom_perc_dom, "Zoom_Dominance_Thresh":zoom_dom_thresh, "Zoom_Magnitude_Thresh": zoom_mean_mag_thresh,"Dominant_Pan": pan_dom_direction, "Percent_Pan": pan_perc_dom, "Pan_Dominance_Thresh": pan_dom_thresh, "Pan_Magnitude_Thresh":pan_mean_mag_thresh}
        print(opticalflow_dict)
        return opticalflow_dict

    def optical_flow_pan(self, flow, dominance_threshold=0.4, magnitude_threshold=0.5):
        """
        Vector orientation:
        t+1=current frame, t=previous frame

        atan( 
            (yi^(t+1) - yi^(t)) / 
            (xi^(t+1) - xi^(t)) 
        )


        Calculate Dominant Orientation = Peak value in orientation histogram of the imag
        np.histogram(np.array(vector orientation))

        returns the dominant orientation given a threshold, the threshold it passed by, and the average magnitude of movement.


        Makkapati, V. (2008). Robust camera pan and zoom change detection using optical flow. 
        In National conference on computer vision, pattern recognition, 
        image processing and graphics (pp. 73-78).
        """
        # get all vector orientations
        orientation = np.rad2deg(cv2.cartToPolar(flow[..., 0], flow[..., 1])[1])
        magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]
        orientation_hist = np.histogram(orientation, bins=8, range=(0,360)) # Bin every 45 degrees

        # print(orientation_hist)
        max_value_index = np.argmax(orientation_hist[0], axis=0)
        percent = orientation_hist[0][max_value_index]/np.sum(orientation_hist[0])
        dominant_orientation = orientation_hist[1][max_value_index]
        
        
        if percent >= dominance_threshold and np.mean(magnitude) >= magnitude_threshold:
            print("PANNING:", dominant_orientation, percent, np.mean(magnitude))
            return dominant_orientation, percent, dominance_threshold, magnitude_threshold

        return None, None, None, None

        

    def optical_flow_zoom(self, flow, dominance_threshold=0.7, magnitude_threshold=1):
        """
        Describe each vector's direction. 
        
        Returns 
            0 if diverging (zoom in), 1 if converging (zoom out), otherwise none.
            percent of dominance
            domenance threshold
            mean magnitude threshold


        Makkapati, V. (2008). Robust camera pan and zoom change detection using optical flow. 
        In National conference on computer vision, pattern recognition, 
        image processing and graphics (pp. 73-78).
        """

        orientation = np.rad2deg(cv2.cartToPolar(flow[..., 0], flow[..., 1])[1])
        magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]

        height, width, dim = flow.shape
        center = (width/2, height/2) # (x,y)

        # converging = np.full((height, width), False)
        
        original_location = np.moveaxis(np.mgrid[:height,:width], 0, -1)

        original_diff_x = (original_location[...,0]**2 - center[0]**2)
        original_diff_y = (original_location[...,1]**2 - center[1]**2)

        # flow_location_x = original_location[..., 0] + flow[..., 0]
        # flow_location_y = original_location[..., 1] + flow[..., 1]
        flow_diff_x = ((original_location[...,0] + flow[..., 0])**2 - center[0]**2)
        flow_diff_y = ((original_location[...,1] + flow[..., 1])**2 - center[1]**2)

        # print(flow_diff_x)
        # print(flow_diff_y)
        flow_diff = abs((flow_diff_x + flow_diff_y)**1/2)
        original_diff = abs((original_diff_x + original_diff_y)**1/2)

        converging = flow_diff <= original_diff

        # print(converging)
        convergence_hist = np.histogram(converging, bins=2, range=(0,1))
        # print(convergence_hist)
        max_value_index = np.argmax(convergence_hist[0], axis=0) # Get the most dominant
        percent = convergence_hist[0][max_value_index]/np.sum(convergence_hist[0]) # get dominance percentage
        dominant_convergence = convergence_hist[1][max_value_index] # get the dominant convergence (True is converging)

        if percent >= dominance_threshold and np.mean(magnitude) >= magnitude_threshold:
            if dominant_convergence >= 0:
                print("ZOOM IN:", dominant_convergence, percent, np.mean(magnitude))
            else:
                print("ZOOM OUT:", dominant_convergence, percent, np.mean(magnitude))
            
            return dominant_convergence, percent, dominance_threshold, magnitude_threshold

        return None, None, None, None


        # print(flow.shape)
        # print(np.ndindex(flow.shape))
        # for iy, ix,  in np.ndindex(flow.shape):
        #     print(iy, ix)

    def ground_truth_characteristics(self, frame_num):
        '''
        Records ground truth height, width, and distance from last frame
        '''
        ground_truths = self.te.get_ground_truths(frame_num)
        labels = []
        gt_areas = []
        gt_heights = []
        gt_widths = []
        for gt in ground_truths:
            box = gt['points']
            labels.append(gt['label'])
            gt_areas.append(self.te.get_area(box))
            gt_heights.append(self.te.get_height(box))
            gt_widths.append(self.te.get_width(box))

        return labels, gt_areas, gt_heights, gt_widths

    def objects_occluded(self, folder_path, fps):
        """
        Records weather the occlusion flag has occured on that frame. Records how many IDs have been occluded.
        """
        occlusion_dict = {}
        for frame in self.te.get_frame_count():
            occlusion_dict[frame] = self.te.count_occlusion(frame)
            print(occlusion_dict[frame])

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

    def occuding_objects(self, occluding_tracks_csv, occluding_threshold=0.8):
        '''
        Given recorded occluding objects, check the precision of ground truth onto occluding object.
        If precision of ground truth is above threshold, this means that object is occluded.
        This is the same measure as check_occlusion but data in this context is different and 
        therefore a different function
        '''
        percent_occluded = {}
        
        # Load occluding tracks
        occlusion_dict = self.te.load_tracker_data(occluding_tracks_csv)

        occluding_frame = []
        occluding_name = []
        occluded_list = []
        percent_occluded_list = []

        #convert occlusions to gt form
        for frame in occlusion_dict.keys():
            occlusion_trackers = self.te.get_estimates(frame)

            for occlusion_object in occlusion_trackers:
                # Correct for comparing tracker to ground_truths based on loaded information
                corrected_occlusion = self.estimate_to_point(occlusion_object, invert_y=self.te.invert_y)

                # Get precision of occluding object onto ground truths.
                # This method looks all ground truths, and checks intersection with occlusion.
                # Precision measures how much of the E covers the GT
                occluded, percent_occluded = self.te.check_occlusion(corrected_occlusion, frame)

                occluding_frame.append(frame)
                occluding_name.append(occlusion_object['name'])
                occluded_list.append(occluded)
                percent_occluded_list.append(percent_occluded)



        # # Return Dictionary of objects and their percent occlusion
        # data = {
        #         "Frame_Num": occluding_frame,
        #         "Occluding_Name": occluding_name,
        #         "Ground_Truth": 
        #         "Occluding": occluded_list,
        #         "Percent_Occluding": percent_occluded_list,
        #         ""


        #         }
        # pass

if __name__ == "__main__":
    # logger = DataLogger("L:/.shortcut-targets-by-id/1BBxJzDfQqUGSfvkILX9LLVkQKSuQcTji/Monkey Videos (for tracker)/Yes/MVI_2975.MOV")
    # logger = DataLogger("D:/GitHub/PeopleTracker/videos/(Simple) GP014125.MP4")
    logger = DataLogger("K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2889.MOV")
    logger.ground_truth_folder = "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2889/"
    
    # logger = DataLogger("L:/.shortcut-targets-by-id/1MP4p63J_OlME1O2ysxy_aSATSfgtn850/Gallery Videos/Historical Videos/Nov 18/GP044104.MP4")
    print("LOGGING")
    logger.get_video_characteristics()
    # logger.export_activity("")
    logger.export_charactoristics("K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2889/")
    # logger.video_info_df.to_csv("K:/Github/PeopleTracker/VIDEO_CHARACTERISTICS_GALLERY.csv")
