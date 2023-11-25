import room_estimation as re
import cv2

# Manually calibrated gopro
calibration_list = ["K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5338.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5339.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5340.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5341.MP4", # Calibration we use
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide/GOPR5983.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver4Med/GOPR5985.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5342.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5343.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Black3Wide/GOPR5982.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Black3Med/GOPR5983.MP4",
                    "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide1080p/GOPR5987.MP4"
                    ]

# Undefined room
room = re.room_estimation()
Contemporary_videos = room.get_files_from_folder("K:/MonkeTracker/Gallery Videos/Contemporary/")

# CONTEMPORARY AGA Room 502
aga_502 = [(0,0,0),(0, 16465.55, 0),(13589, 16465.55,0), (13589,0,0)] # real world locations defined in mm
calibration = calibration_list[3]



    

if __name__ == "__main__":
    image = room.collect_frames("K:/MonkeTracker/Gallery Videos/Contemporary/GP014190.mp4",1,1,2)[0]
    room.image = image
    room.show_img = image
    cv2.waitKey(0)


    room.undistort_room(image)

    # Define room takes in a 13.5m x 16.4m room. We define a height of 1m since every object is assumed to be 1m in height.
    # Extend is the number of walls we want to "extend" into the regions not captured by the camera
    room.define_room(width=13589, length=16465.55, height=1000, stitch_videos = Contemporary_videos, calibration=calibration, extend=3)

