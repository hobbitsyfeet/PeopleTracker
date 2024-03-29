
# import utils



'''
NOTE: Add additional evaluation methods to help plot data

- Centroid point distance. This is an additional measure of accuracy. This is because that's how we collect data for our study.
- region change rate (or shape over time) - this is important to measure alongside performance. This allows us to see where measures go wrong. When ground truths change drastically, how can we expect a tracker to adapt? (Ours has a hard time when objects scale or change shape a lot)


'''

import json
import glob

import pandas as pd


import matplotlib
import matplotlib.pyplot as plt
# import numpy as np

# NOTE: Ground Truth frames are One full second ahead of the estimates. This is because we calculate from 1*60 = 60 so either we subtract a constant fps from the ground truth or add 60 to our data. Adjusting Ground truth would be the best option.

##
# Based on @cite evaluation <a href="https://www.idiap.ch/~odobez/publications/SmithGaticaOdobezBa-cvpr-eemcv05.pdf"> https://www.idiap.ch/~odobez/publications/SmithGaticaOdobezBa-cvpr-eemcv05.pdf </a>
class tracker_evaluation:
    '''!
    Implementation by Justin Petluk
    @cite evaluation 
     <a href="https://www.idiap.ch/~odobez/publications/SmithGaticaOdobezBa-cvpr-eemcv05.pdf"> https://www.idiap.ch/~odobez/publications/SmithGaticaOdobezBa-cvpr-eemcv05.pdf </a> 
    '''
    def __init__(self, tracker_file = None, ground_thruth_folder = None, fps=None):
        '''
        Assigns default values
        '''
        self.score = 0
        self.colour = None

        if ground_thruth_folder is None:
            self.ground_truth_dict = {}
        else:
            self.load_json(ground_thruth_folder, fps)


        self.total_images = None

        if tracker_file is not None:
            self.estimate_dict = self.load_tracker_data(tracker_file)

        if fps is None:
            self.fps = 30
        else:
            self.fps = fps

        ##! Threshold of Precision allowed in agreement or overlap. If above threshold, an occlusion flag is noted. @ref check_occlusion
        self.threshold_to = 0.8

        ## Used for evaluating F-Measure, Higher F-value, the higher allowed error.
        self.threshold_tc = 0.5 

        ##! A map of IDs given on a single frame of data and used to save the output of @ref identification_map
        self.id_map = {}

        self.invert_y = 0

        self.correction_ratio_x = None
        self.correction_ratio_y = None

    ## Loads annotation created with @cite labelme labelme
    def load_json(self, folder, fps=None):
        '''!
        Loads all json files in a folder.

        '''
        # Load all json files
        # Points exist in [x,y] pairs. Labelling determines order of points but it should be top left to bottom right. Make sure this is the order. (If not, flip order of points)
        if fps:
            self.fps = fps
        print("Loading JSON")
        files = glob.glob(folder+"*.json")

        self.total_images = len(files)
        for file in files:
            # Parse the filename and get the image number
            json_image_number = int(file.split(".")[0].split("_")[-1])
            frame_number = json_image_number * self.fps
            with open(file) as f:
                data = json.load(f)
                for shape in data['shapes']:
                    shape['points'] = (shape['points'][0][0],shape['points'][0][1], shape['points'][1][0], shape['points'][1][1])
                self.ground_truth_dict[frame_number] = data['shapes']
        

        # Sort the dictionary
        self.ground_truth_dict = {k: self.ground_truth_dict[k] for k in sorted(self.ground_truth_dict)}
        self.labels = self.list_labels()

    def load_tracker_data(self, tracker_file):
        '''!
        Loads estimates from people tracker exported data.
        
        It loads private variables estimate_dict and estimates (a list of all estimate names)

        Returns a dataframe of tracker data

        '''
        print("Loading Tracker Data")
        df = pd.read_csv(tracker_file)
        self.fps = int(round(df.iloc[0]['FrameRate']))
        # print(df)
        self.invert_y = int(df.iloc[1]['Max_Pixel_y'])
        # self.invert_y = 0

        # try:
        try:
            video_width = int(df.iloc[0]['Width(px)'])
            video_height = int(df.iloc[0]['Height(px)'])

            recorded_width = int(df.iloc[1]['Max_Pixel_x'])
            recorded_height = int(df.iloc[1]['Max_Pixel_y'])

            # Example: Estimates are recorded at 480x720 while ground truths are recorded at 720x1280. The ratio inputed would be (720/480, 1280/720) or (1.5, 1.777778)
            self.correction_ratio_x = video_width/recorded_width
            self.correction_ratio_y = video_height/recorded_height

        except:
            print("It's likely that this data is MaskRCNN Predicted and recorded at full resolution: Setting scale ratio to 1:1")
            # Example: Estimates are recorded at 480x720 while ground truths are recorded at 720x1280. The ratio inputed would be (720/480, 1280/720) or (1.5, 1.777778)
            self.correction_ratio_x = 1
            self.correction_ratio_y = 1

        # except:
        # self.correction_ratio_x = 1
        # self.correction_ratio_y = 1

        df = df.iloc[1: , :] # Drop first row


        
        print("FRAME RATE", self.fps)
        tracker_data = df[["Frame_Num", "Name", "ID", "BBox_TopLeft_x", "BBox_TopLeft_y", "BBox_BottomRight_x", "BBox_BottomRight_y"]]
        tracker_data = tracker_data.rename(columns={'Frame_Num': 'frame', 
                                'Name': 'name',
                                'ID': 'id',
                                'BBox_TopLeft_x': 'x1',
                                'BBox_TopLeft_y': 'y2',
                                'BBox_BottomRight_x': 'x2',
                                'BBox_BottomRight_y': 'y1'})
        # print(tracker_data)
        self.estimate_dict = tracker_data
        self.estimates = self.list_estimates()
        print("Done loading Data")
        return self.estimate_dict

        
    def plot_score(self):
        '''
        Not implemented
        '''
        # Plots timeline (in frames) according to ground truth. Additionally, plots id and errors.
        pass

    def plot_scene(self):
        '''
        Not Implemented
        '''
        # Plots a colour coded point over the scene. additionally plots id and errors spatially.
        pass

    def identification_graph(self):
        '''
        
        '''
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        start_frame = 0

        data = []
        
        frame_number = []
        data = []
        
        gt_count = self.get_groundtruth_count()
        es_count = self.get_estimate_count()
        print("ES_COUNT", es_count)
        colors = {}
        es_colors = {}

        #Create Colour maps
        for index, gt in enumerate(gt_count.keys()):
            colors[gt] = index+1
            
        for index, es in enumerate(es_count.keys()):
            es_colors[es] = index+1
        # cmaps = {}
        self.gt_map, self.es_map = self.calculate_identification_map()
        
        print("Loading data to plot")
        
        
        figs =[]
        cmap = []
        id_map_colors = []
        for frame in self.ground_truth_dict:
            # print(frame)
            ground_truths = self.get_ground_truths(frame)
            id_map = self.identification_map(frame)
            for gt in ground_truths:
                # ax.plot(gt['label'], frame)
                data.append(gt['label'])
                cmap.append(colors[gt['label']])
                frame_number.append(frame)
                if id_map is not None:
                    es_map = id_map['Estimate']
                    gt_map = id_map['Ground_Truth']

                    keys = [k for k, v in es_map.items() if v == gt['label']]
                    added = False
                    for key in keys:
                        if key == gt_map[gt['label']]:
                            id_map_colors.append(es_colors[gt_map[gt['label']]])
                            added = True
                            break
                    if not added:
                        id_map_colors.append(-20) 
                        
                else:
                    id_map_colors.append(-20)

        self.gt_map, self.es_map = self.calculate_identification_map()
    
        ax.scatter(frame_number, data, c=id_map_colors,  cmap="Spectral", marker="s", s=80)
        ax.scatter(frame_number, data, c=cmap,  cmap="inferno", marker=".", s=50)  # Plot some data on the axes.
        
        plt.show(block = False)

    def false_positive(self, es_config):
        '''
            |1 |2 |3 |4 |5  |
        ____|__|__|__|__|___|___
        GT  |a |b |b |- |d,e| 
                      ↑   ↑
                      FP  MO
        '''
        # An estimate exists that is not associated with a ground truth object
        # Tracker is on but not on a ground truth
        fp_count = 0
        for estimate in es_config.values():
            if None in estimate:
                fp_count += 1
        return fp_count

    def false_negative(self, gt_config):
        '''
        False Negative checks all estimates in a given list (of a given frame) and 
        checks if any of them exist on a ground truth within the f-measure threshold.

        Config map assigns a None for a value to a ground truth when no estimate exists 
        within it's threhsold
        estimate E

             ground truth GT
            |a |b  | c |d |e |
        ____|__|___|___|__|__|__
        E   |1 |2,3| - |5 |5 | 
                 ↑   ↑
                 MT  FN
        '''



        fn_count = 0
        for gt in gt_config.values():
            if None in gt:
                fn_count += 1
        return fn_count

    def multiple_trackers(self, gt_config, frame_number):
        '''
        Two or more estimates are associated with the same ground truth. A MT error is assigned for each excess estimate.
        Multiple trackers track a ground truth
        
        Look for more than one estimate for each ground truth

             ground truth GT
            |a |b  | c |d |e |
        ____|__|___|___|__|__|__
        E   |1 |2,3| - |5 |5 | 
                 ↑   ↑
                 MT  FN
        '''

        mt = 0
        for gt in gt_config:
            gts = self.get_ground_truths(frame_number, gt)
            occluded, _ = self.check_occlusion(gts[0], frame_number)

            es = gt_config[gt]
            if not occluded:
                mt += len(es) - 1

        return mt


    def multiple_objects(self, es_config, frame_number):
        '''
                estimate E
            |1 |2 |3 |4 |5  |
        ____|__|__|__|__|___|___
        GT  |a |b |b |- |d,e| 
                      ↑   ↑
                      FP  MO
        '''
        mo = 0
        # counts excess count of a 1:1 ground_truth:estimate
        for gt in es_config.values():
            
            occluded = False

            #Iterate through all the ground truths and check for occlusion.
            for ground_truth in gt:
                gts = self.get_ground_truths(frame_number, ground_truth)
                occluded, _ = self.check_occlusion(gts[0], frame_number)
                #If it is occluded we don't need to keep checking if it is
                if occluded:
                    break
            
            # # NOTE: This case represents the 'less correct' case mentioned in figure 3
            # #If it's not occluded, we count all excess objects
            # if not occluded:
            #     mo += len(gt) - 1
            
            # #This represents the 'more correct' case in figure 3
            if not occluded and len(gt) > 1:
                mo += len(gt)

        return mo
        
        # Still need to eliminate each excess ground truth
    def configuration_distance(self, frame_number):
        '''
        Checks the difference between the number of ground truths and the number of estimates of a given frame
        CD = #Estiamtes - #Ground Truths / max(#Ground Truths, 1)
        '''
        # collect all the data on given frame
        estimates = self.get_estimates(frame_number)

        if estimates is None:
            total_es = 0
        ground_truths = self.get_ground_truths(frame_number)
        # print(ground_truths)


        # Total ground truths and estimates
        total_es = estimates.shape[0]
        total_gt = len(ground_truths)
        # print("Total_GT", total_gt)
        # print("Total ES:", total_es, "Total GT:", total_gt)
        # Find the difference and set and normalization is non-zero total ground_truths.
        difference = total_es - total_gt
        normalization = max(total_gt,1)

        # CD is the difference over the total ground truths. 0 when es = gt. Negative gt > es, positive gt < es
        cd = difference/normalization
        # print("NormalizedCD", cd, "CD", difference)
        total_cd = difference
        
        # print("CD Normalization", normalization)

        # Don't normalize
        # print("Es:", total_es, " | GTs:",  total_gt, end=" - ")
        return cd, total_cd

    def ground_truth_exists(self, frame_num):
        if frame_num in self.ground_truth_dict.keys():
            return True
        return False

    def get_ground_truth_difference(self, frame_num_1, frame_num_2):
        '''
        Returns the sets of ground truths where ids are new, or leaving.
        '''
        gt_1 = self.get_ground_truths(frame_num_1)
        gt_2 = self.get_ground_truths(frame_num_2)

        set_1 = set()
        if gt_1:
            for gt in gt_1:
                set_1.add(gt['label'])
            
        set_2 = set()
        if gt_2:
            for gt in gt_2:
                set_2.add(gt['label'])
            
        new = set_2 - set_1
        leaving = set_1 - set_2
        return new, leaving

    ## Checks if a ground truth is occluded.
    def check_occlusion(self, gt,  frame_num):
        '''!
        Checks if a ground truth is occluded at a certain frame. 
        
        @param gt <b>(int, int, int, int)</b> a bounding box of the specified gt to check if they are occluded
        @param frame_num <b>int</b>

        @ref precision is calculated to generate percent covered.
        @ref threshold_to is used to determine occlusion theshold.

        @returns <b>True</b> if anyother ground truth is occluding the ground truth passed in
        @returns <b>False</b> if there exists no other ground truth that occludes 
        @returns <b>float</b> a percent covered (Precision)
        '''
        all_gt = self.get_ground_truths(frame_num)
        percent_covered = 0
        # Remove current shape from list so all other shapes are compared
        # Calculate intersection of other points onto ground truth
        for test in all_gt:
            if gt['label'] == test['label']:
                continue
            
            percent_covered = self.precision(gt['points'], test['points'])
            #Iterate through all other ground truths, returning true if any intersection exceeds threshold_to
            if percent_covered > self.threshold_to:
                return True, percent_covered

        # No occlusion if no value exceeds
        return False, percent_covered

    ## Returns a number of occluded ground truths occur at a frame
    def get_occlusion_count(self, frame_num):
        '''!
        Counts the number of True occurances of @ref check_occlusion happens on a frame.

        @returns int Count of number of occluded ground truths
        '''
        all_gt = self.get_ground_truths(frame_num)

        if not self.ground_truth_exists(frame_num):
            return None
            
        count = 0
        # Iterate over all ground truths and record how many others overlap with excess threshold_to
        for gt in all_gt:
            occluded, _ = self.check_occlusion(gt, frame_num)
            if occluded:
                count += 1
        
        return count

    
    def precision(self, estimate, ground_truth, correct_estimate=False):
        '''
        Precision measures how much of the E covers the GT and
        can take values between 0 (no overlap) and 1 (fully overlapped)
        '''

        if correct_estimate:
            measures = self.compute_iou(estimate, [ground_truth], ratios=(self.correction_ratio_x, self.correction_ratio_y))
        else:
            measures = self.compute_iou(estimate, [ground_truth])

        intersection = measures[1]

        gt = self.get_area(ground_truth)
        return abs(intersection[0])/abs(gt)
    
    def recall(self, estimate, ground_truth, correct_estimate=False):
        '''
        Recall measures how much of the GT covered by the
        E and can take values between 0 (no overlap) and 1 (fully overlapped)
        '''

        if correct_estimate:
            measures = self.compute_iou(estimate, [ground_truth], ratios=(self.correction_ratio_x, self.correction_ratio_y))
        else:
            measures = self.compute_iou(estimate, [ground_truth])
        intersection = measures[1]
        es = self.get_area(estimate)
        return abs(intersection[0])/abs(es)

    def fmeasure(self, estimate, ground_truth, correct_estimate=False):
        '''
        The F-measure(F = 2νρ/ν+ρ ) is suited to this task, as F is only high when both recall and precision are high
        '''
        # print("FMEASURE COMPARE ESTIMATE:", estimate, " | GT:", ground_truth)

        if correct_estimate is False:
            p = self.precision(estimate, ground_truth)
            r = self.recall(estimate, ground_truth)

        # Avoids division by zero
        if p == 0 and r == 0:
            return 0

        f = (2 * p * r) / (p + r)
        return f

    def configuration_map(self, frame_number):
        
        ground_truths = self.get_ground_truths(frame_number)
        estimates = self.get_estimates(frame_number)

        gt_config = {}
        es_config = {}
        for index, estimate in estimates.iterrows():
            
            es = self.estimate_to_point(estimate)
            name = estimate['name']
            for gt in ground_truths:
                f = self.fmeasure(es, gt['points'], correct_estimate=False)
                if f > self.threshold_tc:
                    if name not in es_config.keys():
                        es_config[name] = [gt['label']]
                    else:
                        es_config[name].append(gt['label'])
                    if gt['label'] not in gt_config.keys():
                        gt_config[gt['label']] = [name]
                    else:
                        gt_config[gt['label']].append(name)
                        
        
        for gt in ground_truths:
            if gt['label'] not in gt_config.keys():
                gt_config[gt['label']] = [None]
        
        for index, estimate in estimates.iterrows():
            if estimate['name'] not in es_config.keys():
                es_config[estimate['name']] = [None]

        # print("GT_CONFIG", gt_config, "ESCONFIG", es_config)
        
        return gt_config, es_config

    def calculate_identification_map(self, manual_gtmap=None, manual_emap=None):
        """! 
        @brief Calculates Identification map for an entire video

        Calculate_identification_map calculates the id map of every frame based on how one fits onto another. 
        Next is majority voting which counts the occurances of maps and grabs the ones which have the most.

        @note manual_gtmap and manual_emap is used when the identifications map 1:1 with eachother.
        For example, if Ground Truth is P1, then it's corresponding estimate is also P1, as well as Estimate P1 maps to Ground Truth P1.

        @param manual_gtmap input should be the unique set of ground truth label names
        @param manual_emap input should be the unique set of estimate label names

        @warning Required to use this when using labeled tracks (Not MaskRCNN) and it is required you DO NOT use this while using MaskRCNN predictions.



        """
        # if manual_gtmap is not None and manual_emap is not None:
        #     return self.list_labels(), self.list_estimates()
            # return manual_gtmap, manual_emap
        

        #Calculates ID Map for every frame. One frame can have multiple maps
        e_map = []
        g_map = []
        for frame in self.ground_truth_dict:
            maps = self.identification_map(frame)
                
            if maps is not None:
                e = maps["Estimate"]
                g = maps["Ground_Truth"]
                
                for key in e.keys():
                    e_map.append((key, e[key]))
                
                for key in g.keys():
                    g_map.append((key, g[key]))
            
        e_map = dict((x,e_map.count(x)) for x in set(e_map))
        g_map = dict((x,g_map.count(x)) for x in set(g_map))   

        print("EMAP",e_map)
        print("GMAP",g_map)

        # print("E_MAP",e_map, "\nG_MAP", g_map)

        #Majority voting

        # Iterate through labels and grab the label with the majority votes
        majority_vote_labels = {}
        for label in self.labels:
            # Majority = ((label,estimate), count)
            majority = None
            # print(label)
            for pair in g_map.keys():
                # Assign the majority to the first
                
                if label == pair[0]:
                    if majority is None:
                        majority = (pair, g_map[pair])

                    #Get the count and replace the majority if the count is 
                    elif g_map[pair] > majority[1]:
                        majority = (pair, g_map[pair])

                        # print(majority)
            # Majority_Vote_Labels = {label: estiamte}
            # print(majority)
            if majority is not None:
                majority_vote_labels[majority[0][0]] = majority[0][1]
            # else:
            #     # majority_vote_labels[]
        
        majority_vote_estimates = {}
        for es in self.list_estimates():
            # Majority = ((label,estimate), count)
            majority = None
            for pair in e_map.keys():
                # Assign the majority to the first
                if es == pair[0]:
                    
                    if majority is None:
                        majority = (pair, e_map[pair])
                        if es == "L2":
                            print("P2 MAjority!", majority)
                    #Get the count and replace the majority if the count is 
                    elif e_map[pair] > majority[1]:
                        majority = (pair, e_map[pair])
            # Majority_Vote_Labels = {label: estiamte}
            if majority is not None:
                majority_vote_estimates[majority[0][0]] = majority[0][1]

        
        return majority_vote_labels, majority_vote_estimates



    def identification_map(self, frame_number):
        """
        Identification map produces a map for a specific frame number
        """
        ground_truths = self.get_ground_truths(frame_number)
        estimates = self.get_estimates(frame_number)
        if estimates.empty:
            return None

        estimate_map = {}
        ground_truth_map = {}

        for index, estimate in estimates.iterrows():
            # print(estimate)
            
            es = self.estimate_to_point(estimate)
            fmeasures = []
            for gt in ground_truths:
                f = self.fmeasure(es, gt['points'], correct_estimate=False)
                # print("F", f)
                fmeasures.append(f)
            # Use the best estimate
            max_index = fmeasures.index(max(fmeasures))
            if fmeasures[max_index] == 0:
                estimate_map[estimate['name']] = None
            else:
                estimate_map[estimate['name']] = ground_truths[max_index]['label']

        for gt in ground_truths:
            fmeasures = []
            for index, estimate in estimates.iterrows():
                es = self.estimate_to_point(estimate)
                f = self.fmeasure(es, gt['points'], correct_estimate=False)
                fmeasures.append(f)
            
            max_index = fmeasures.index(max(fmeasures))
            if fmeasures[max_index] == 0:
                ground_truth_map[gt['label']] = None
            else:
                ground_truth_map[gt['label']] = estimates.iloc[max_index]['name']

        maps = {"Estimate": estimate_map, "Ground_Truth": ground_truth_map}
        # print(maps)
        return maps

    def falsly_identified_object(self, frame_number):
        """
        Calculates FIO for a single frame normalized by the number of estimates in that frame
        """
        #Calculates the id_map for the frame, both estimates and ground_truths
        id_map = self.identification_map(frame_number)

        if id_map is not None:
            es_map = id_map["Estimate"]
            gt_map = id_map["Ground_Truth"]


            scores = {}

            # Go through all the estimates in the frame
            for estimate in es_map.keys():
                # if the estimate matches the correct map, set that to true, otherwise that estimate is false.
                if es_map[estimate] == self.es_map[estimate]:
                    scores[estimate] = True
                elif es_map[estimate] == None:
                    scores[estimate] = None
                else:
                    scores[estimate] = False

            #Count the errors for that frame
            total_errors = 0
            for score in scores.values():
                if score is False:
                    total_errors += 1
        else:
            return 0, 0, {}
        
        normalized_error = total_errors/len(gt_map.keys())
        # print(normalized_error)

        return normalized_error, total_errors, scores


    def falsly_identified_tracker(self, frame_number):
        """
        Falsely Identified Object. A
        GT segment which passes the coverage test for E but
        is not the identified GT.
        
        Assign FIT and FIO errors for each frame by checking if E
        and GT segments which pass the coverage test match Es and 
        GTs indicated by the identity maps. Label segments as FIT,
        FIO, or correct
        """      
        id_map = self.identification_map(frame_number)
        

        if id_map is not None:
            es_map = id_map["Estimate"]
            gt_map = id_map["Ground_Truth"]
            scores = {}

            # Go through all the estimates in the frame
            for estimate in es_map.keys():
                # if the estimate matches the correct map, set that to true, otherwise that estimate is false.
                gt = es_map[estimate]
                if gt is not None:
                    if estimate == self.gt_map[gt]:
                        scores[estimate] = True
                    else:
                        scores[estimate] = False

            #Count the errors for that frame
            total_errors = 0
            for score in scores.values():
                if score is False:
                    total_errors += 1
        else:
            return 0, 0, {}
        
        normalized_error = total_errors/len(gt_map.keys())
        # print(normalized_error)

        return normalized_error, total_errors, scores

    def calculate_tracker_purity(self):
        '''
        Measure I-3: (TP) Tracker Purity. If the identity map
        w.r.t. E indicates Ei identifies GT j , the ratio of frames
        that Ei correctly identifies GT j ( niˆji ) to the total num-
        ber of frames Ei exists (ni) 
        '''
        e_count = self.get_estimate_count() #total number of frames Ei exists (ni)
        correct_dict = {}

        # Frames that Ei correctly identifies GTj ( niˆji )
        for frame in self.ground_truth_dict:
            id_map = self.identification_map(frame)
            # print(id_map)
            if id_map is not None:
                estimate_map = id_map['Estimate']
                estimates = self.get_estimates(frame)
                for name in estimates['name']:
                    if self.es_map[name] == estimate_map[name]:
                        if name in correct_dict.keys():
                            correct_dict[name] += 1
                        else:
                            correct_dict[name] = 1
                    else:
                        print("TRACKER PURITY FALSE", name, estimates['name'])
        purity = {}
        
        for name in e_count.keys():
            purity[name] = correct_dict[name]/e_count[name]
        
        tp = 0
        for value in purity.values():
            tp += value
            # print(value)
        total_correct = tp

        tp = tp/len(purity.values())
        print("TRACKER PURITY", tp, purity)
        return tp, total_correct, purity

    def calculate_object_purity(self):
        '''
        Measure I-4: (OP) - Object Purity. If the identity map
        w.r.t. GT indicates GTj is identified by Ei, the ratio of
        frames that GT j is correctly identified by Ei ( njˆij ) to
        the total number of frames GT j exists (nj ) 
        '''
        g_count = self.get_groundtruth_count() #total number of frames Ei exists (ni)
        correct_dict = {}

        # Frames that Ei correctly identifies GTj ( niˆji )

        #Iterate over the entire video's frames
        for frame in self.ground_truth_dict:
            id_map = self.identification_map(frame)

            #Check if ID map is valid
            if id_map is not None:
                
                #Grab ground truths and what the ground truths map to
                gt_map = id_map['Ground_Truth']
                ground_truths = self.get_ground_truths(frame)
                # estimates = self.get_estimates(frame)

                #Go through each ground truh and check if the current frame matches the id_map key
                for gt in ground_truths:
                    name = gt['label']
                    # if name not in list(correct_dict.keys()):
                    #     print("NOT IN, assigning", name)
                    #     correct_dict[name] = int(0)
                    #     print(correct_dict[name])
                    if self.gt_map[name] == gt_map[name]:
                        # print(self.gt_map[name], [name])

                        #If it does, we add to the number of times each id is correct
                        if name in correct_dict.keys():
                            correct_dict[name] += 1
                        else:
                            correct_dict[name] = 1

                        

        purity = {}
        for name in g_count.keys():
            if name not in correct_dict.keys():
                purity[name] = 1
            else:
                purity[name] = correct_dict[name]/g_count[name]
        
        op = 0
        for value in purity.values():
            op += value
        
        
        total_correct = op
        
        op = op/len(purity.values())
        print("Purity", purity)
        return op, total_correct, purity

    def calculate_frame_errors(self, frame_number, ratios=(1,1), y_offset=720):
        '''
        '''
        # print("Getting map...", end='/n')
        gt_map, es_map = self.configuration_map(frame_number)
        print(gt_map, es_map)
        # print("Getting CD...", end='/n')
        cd, total_cd = self.configuration_distance(frame_number)
        # print("Calculating Results")
        results = { "FP": self.false_positive(es_map), 
                    "FN": self.false_negative(gt_map),
                    "MT": self.multiple_trackers(gt_map, frame_number),
                    "MO": self.multiple_objects(es_map, frame_number),
                    "CD": cd,
                    "FIT":self.falsly_identified_tracker(frame_number), 
                    "FIO":self.falsly_identified_object(frame_number)
                    }
        # print("Results:", results)
        return results

    def calculate_identification_errors(self, frame_number):
        return {"TP":self.calculate_tracker_purity()[0], "OP":self.calulate, "FIT":self.falsly_identified_tracker(frame_number)[0], "FIO":self.falsly_identified_object(frame_number)[0]}
            


    def calculate_errors(self, printout=False, gt_maps_itself=True):
        """
        Iterates through every ground truth frame and compares a list of estimates. 

        Returns final video score, as well as normalized results through the video.
        Normalized results are results where the number of errors on a frame is normalized by the number of ground truths.

        We define GT maps itself to override majority voting to determine what the correct labels are. This accounts for the problem where
        an estimate incorrectly labels a ground truth forthe majority of the time
        """
        errors_dict = {}

        total_fp = 0
        total_fn = 0
        total_mt = 0
        total_mo = 0
        total_cd = 0

        total_fit = 0
        total_fio = 0

        self.gt_map, self.es_map = self.calculate_identification_map()

        if gt_maps_itself:
            self.gt_map, self.es_map = self.generate_manual_id_map()
            # self.es_map = self.gt_map
            # print("GT MAPPING ITSELF", self.gt_map, self.es_map)

        print("MAPS", self.gt_map, self.es_map)
        # input()
        
        # self.gt_map, self.es_map = self.calculate_identification_map()
        tp, total_tp, purity  = self.calculate_tracker_purity()
        op, total_op, o_purity = self.calculate_object_purity()

        count_fp = 0
        count_fn = 0
        count_mt = 0
        count_mo = 0
        count_cd = 0
        count_fit = 0
        count_fio = 0

        record_frames = []
        record_fp = []
        record_fn = []
        record_mt = []
        record_mo = []
        record_cd = []
        record_fit = []
        record_fio = []

        for frame in self.ground_truth_dict:
            
            # print("\nFRAME", frame)
            ngt = max(len(self.get_ground_truths(frame)),1)
            # nes = len(self.list_estimates())
            errors = self.calculate_frame_errors(frame)
            cd, cd_t = self.configuration_distance(frame)

            # print(errors)
            # print(count_cd)
            count_fp += errors['FP']
            count_fn += errors['FN']
            count_mt += errors['MT']
            count_mo += errors['MO']
            count_cd += cd_t
            

            total_fp += errors['FP']/ngt
            total_fn += errors['FN']/ngt
            total_mt += errors['MT']/ngt
            total_mo += errors['MO']/ngt
            total_cd += abs(cd)

            # print("Normalized CD", errors['CD'])
            
            fit, c_fit, fit_scores = self.falsly_identified_tracker(frame)
            fio, c_fio, fio_scores = self.falsly_identified_object(frame)

            total_fio += fio
            total_fit += fit
            count_fit += c_fit
            count_fio += c_fio 

            # Records normalized results per frame
            record_frames.append(frame)
            record_fp.append(errors['FP']/ngt)
            record_fn.append(errors['FN']/ngt)
            record_mt.append(errors['MT']/ngt)
            record_mo.append(errors['MO']/ngt)
            record_cd.append(abs(cd))
            record_fit.append(fit)
            record_fio.append(fio)
            
            # id_errors = self.calculate_identification_errors(frame)
            # total_tp += id_errors['TP']
            # total_op += id_errors['OP']
            # total_fit += id_errors['FIT']
            # total_fio += id_errors['FIO']
            # print(id_errors)

        # counts = {"FP": count_fp, 
        #             "FN": count_fn,
        #             "MT": count_mt,
        #             "MO": count_mo,
        #             "CD": count_cd,
        #             "TP": total_tp,
        #             "OP": total_op,
        #             "FIT": count_fit,
        #             "FIO": count_fio
        #             } 
        # print("\nCOUNTS", counts)


        # results = {"FP": total_fp, 
        #             "FN": total_fn,
        #             "MT": total_mt,
        #             "MO": total_mo,
        #             "CD": total_cd,
        #             "TP": total_tp,
        #             "OP": total_op,
        #             "FIT": total_fit,
        #             "FIO": total_fio
        #             } test
        
        # print(normalization)
        total_frames = self.total_images
        print("Total FP:", total_fp)
        print(self.total_images)
        # Normalization
        normalized_results = {
                    "FP": round((1/(total_frames)) * (total_fp),3),    #Good
                    "FN": round((1/(total_frames)) * (total_fn),3),    #Good 
                    "MT": round((1/(total_frames)) * (total_mt),3),    #Good
                    "MO": round((1/(total_frames)) * (total_mo),3),    #Good
                    "CD": round((1/(total_frames)) * total_cd,3),      #Good

                    "FIT": round((1/total_frames) * (total_fit),3),    #Good
                    "FIO": round((1/total_frames) * (total_fio),3),     #Good
                    "TP": round((tp), 3),                         #Good
                    "OP": round((op),3)     #Untested
                    }

        # Runs through history of the errors, normalized to compare errors over time, between videos 
        #NOTE tp and op are not recorded because those are performance across the entire video
        recorded_results = {
            "Frames":record_frames,
            "FP":record_fp,
            "FN":record_fn,
            "MT":record_mt,
            "MO":record_mo,
            "CD":record_cd,
            "FIT":record_fit,
            "FIO":record_fio
        }

        print("\nNORMALIZED", normalized_results)
        return normalized_results, recorded_results
        # return results, normalized_results

    def compute_iou(self, box, boxes, ratios=None, frame=None):
        """!
        Calculates IoU of the given box with the array of the given boxes.
        
        @param box: 1D vector [x1, y1, x2, y2]
        @param boxes: [boxes_count, (x1, y1, x2, y2)]
        @param box_area: float. the area of 'box'
        @param boxes_area: array of length boxes_count.
        @param ratio: ratio (width, height) to scale boxes from video resolution to analysis resolution

        @note: The areas are passed in rather than calculated here for efficiency. Calculate once in the caller to avoid duplicate work.
        """
        if ratios is None:
            ratios = self.correction_ratio_x, self.correction_ratio_y

        area = (box[0] - box[2]) * (box[1] - box[3]) # SOMETHING IS WRONG WITH BOX. GOOD NIGHT!
        ious = []
        intersections = []
        for index, preds in enumerate(boxes):
            
            # x1 = int(preds[0]*ratios[0])
            # y1 = int(preds[1]*ratios[1])
            # x2 = int(preds[2]*ratios[0])
            # y2 = int(preds[3]*ratios[1])

            # get the box coordinates (x1, y1, x2, y2)
            x1 = int(preds[0])
            y1 = int(preds[1])
            x2 = int(preds[2])
            y2 = int(preds[3])

            # if frame is not None:
            #     cv2.rectangle(frame, (x1,y1), (x2,y2), (150,150,0), 1)

            # calculating IOU
            xA = max(box[0], x1)
            yA = max(box[1], y1)
            xB = min(box[2], x2)
            yB = min(box[3], y2)

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            
            boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)

            iou = interArea / float(area + boxBArea - interArea)
            ious.append(iou)
            intersections.append(interArea)

        return ious, intersections, frame
    
    def get_frame_count(self):
        return self.ground_truth_dict.keys() 

    def get_area(self, box):
        return (box[0] - box[2]) * (box[1] - box[3])

    def get_height(self, box):
        return (box[1] - box[3])
    
    def get_width(self, box):
        return (box[0] - box[2])

    def get_ground_truths(self, frame_number, id=None):
        if frame_number in self.ground_truth_dict.keys():
            ground_truths = self.ground_truth_dict[frame_number]
            if id:
                for ground_truth in ground_truths:
                    if id in ground_truth['label']:
                        return [ground_truth]
            return ground_truths
        else:
            # print("frame out of range")
            return None
    
    def get_estimates(self, frame_number, id=None):
        '''
        Returns all estimates by frame number
        '''
        estimates = self.estimate_dict
        estimates = estimates.loc[estimates['frame'].astype(int) == frame_number]
        if id:
            estimates = estimates.loc[estimates['name'] == id]

        return estimates

    def estimate_to_point(self, estimate, ratios = None , invert_y=None):
        """
        compute_iou
        ratios is the x,y|width,height ratios of the estimate to the ground truth.compute_iou
        Example: Estimates are recorded at 480x720 while ground truths are recorded at 720x1280. The ratio inputed would be (720/480, 1280/720) or (1.5, 1.777778)
        

        invert_y is the height of the video. We record the data as if the origin is in the bottom left, but in other applications the origin is the top left.
        This means we subtract the height of the video to inverse this effect.
        If the height of the video is 720, invert_y=720. If the data being tested IS NOT inverted, leave it as 0 
        """
        if ratios is None:
            ratios = (self.correction_ratio_x, self.correction_ratio_y)

        if invert_y is None:
            invert_y = self.invert_y

        point = (float(estimate['x1'])*ratios[0],
        (invert_y-float(estimate['y2']))*ratios[1],
                float(estimate['x2'])*ratios[0],
        (invert_y-float(estimate['y1']))*ratios[1]
        )

        return point

    def get_number_estimates(self, frame_number):
        es = self.get_estimates(frame_number)
        return es.shape[0]

    def get_ground_truth_count(self, frame_number):
        gts = self.get_ground_truths(frame_number)
        # print(gts, frame_number)
        if gts == None:
            return None
        return len(gts)

    def list_labels(self):
        '''
        Returns a list of unique label names
        '''
        label_set = set()
        for frame in self.ground_truth_dict:
            for gt in self.ground_truth_dict[frame]:
                if gt['label'] not in label_set:
                    label_set.add(gt['label'])
        return list(label_set)

    def list_estimates(self):
        '''
        Returns a list of unique estimate names
        '''
        estimate_set = set()
        # print(self.ground_truth_dict)
        for frame in self.ground_truth_dict:
            # print(frame)
            estimates = self.get_estimates(frame)
            # print(estimates)
            if estimates.empty is False:
                for name in estimates['name']:
                    estimate_set.add(name)
        print(estimate_set)
        
        return list(sorted(estimate_set))
    
    def get_estimate_count(self):
        estimate_count = {}
        for frame in self.ground_truth_dict:
            estimates = self.get_estimates(frame)
            if estimates.empty is False:
                for name in estimates['name']:
                    if name in estimate_count.keys():
                        estimate_count[name] += 1
                    else:
                        estimate_count[name] = 1
        return estimate_count

    def get_groundtruth_count(self):
        gt_count = {}
        for frame in self.ground_truth_dict:
            for gt in self.ground_truth_dict[frame]:
                if gt['label'] in gt_count.keys():
                    gt_count[gt['label']] += 1
                else:
                    gt_count[gt['label']] = 1

        return gt_count
    
    def validate_and_correct_ground_tuths(self):
        '''
        This function removes ground truths where files imported are faulty.

        1: Labelme file is created but no ground truths are assigned.
        '''
        frames_removed = []
        for frame in self.ground_truth_dict:
            gt = self.ground_truth_dict[frame]
            # removes the frame from the dictionary
            if gt == []:
                frames_removed.append(frame)
                print("Removing frame ", frame, " because no ground truths found:", gt)
                # del self.ground_truth_dict[frame]

        for del_key in frames_removed:
            del self.ground_truth_dict[del_key]

        return frames_removed
        
    def generate_manual_id_map(self):
        '''
        Creates a manual mapping where every estimate maps to the same id as it's labelled (as well as ground truths)

        This is used in the case where ground truths and estimates are designed to have the same IDs
        '''
        manual_gt_map = {}
        for gt in self.list_labels():
            manual_gt_map[gt] = gt
        
        manual_es_map = {}
        for es in self.list_estimates():
            manual_es_map[es] = es

        return manual_gt_map, manual_gt_map

if __name__ == "__main__":
    # cv2.waitKey(0)

    te = tracker_evaluation()

    # te.load_json("K:/Github/PeopleTracker/src/testing/ID_Tests/", fps=1)
    # te.load_tracker_data("K:/Github/PeopleTracker/src/testing/ID_Tests/Test_ID.csv")
    te.load_tracker_data("F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GOPR4190/GOPR4190.csv")
    te.load_json("F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GOPR4190/", fps=60)
    
    te.identification_graph()
    errors, errors_dict = te.calculate_errors()
    print(errors)