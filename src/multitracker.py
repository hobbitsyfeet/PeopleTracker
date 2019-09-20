import cv2

class MultiTracker():
    def __init__(self, name="Person", colour=(255,255,255)):
        self.name = name
        self.colour = colour

        self.location_data = [] #(x, y)
        self.distance_data = [] #(CLOSE, MED, FAR) (estimated)
        self.time_data = [] #(frames tracked)

        self.record_state = False

        self.tracker = none


    def create(self):
        """
        The creation of the tracking square on a frame
        """
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[2]
    
        # if int(minor_ver) < 3:
            # tracker = cv2.Tracker_create(tracker_type)
        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
    # if int(minor_ver) < 3:
        # tracker = cv2.Tracker_create()


    def rename(self, name):
        """
        Renames the tracking object
        """
        self.name = name
        

    def set_colour(self, color=(255,255,255)):
        self.colour = color

    def reassign(self):
        pass

    def remove(self):
        pass

    def predict(self,):
        """
        Applies KNN taking into consideration a Hidden Markov Model
        """
        pass

    def record_data(self, frame, x, y, size):
        self.location_data.append((x,y)) #(x, y)
        self.time_data.append(frame) #(frames tracked)
        self.distance_data.append(self.estimate_distance(size)) #(CLOSE, MED, FAR) (estimated)

    def append_data(self, dataframe):
        """
        Appends the data into a given dataframe categoriezed with the name given
        """
        pass

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
    
    tracker = MultiTracker()
    
    time_tracked = [0,1,3,5,6,7,12,15,20,25,30,35,40,50,70,100,105,106,115,130]
    print("Testing Segment calculations")
    segments = tracker.part_time_to_segments(time_tracked, segment_size=20)
    print(segments)

    time = tracker.calculate_time(time_tracked[0],time_tracked[-1])
    print(str(time) + " Seconds")


    total_time = tracker.calculate_total_time(segments)

    print("Total Time given segments " + str(total_time) + " seconds")

    