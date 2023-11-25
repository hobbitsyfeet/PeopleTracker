from cmath import inf, nan
import math
from multiprocessing.spawn import prepare
import numpy as np
import cv2
from copy import deepcopy
import pickle

from scipy import interpolate

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from sympy import limit

import pandas as pd
import os
# from sympy import symbols, Eq, solve
# from sympy.parsing.sympy_parser import parse_expr

# NOTE: To visualize extrinics: https://github.com/opencv/opencv/blob/9da9e8244b75a3754ff3a5d07ffdac46ee28ab6b/samples/python/camera_calibration_show_extrinsics.py#L132
"""
xdistorted=x(1+k1r2+k2r4+k3r6)
ydistorted=y(1+k1r2+k2r4+k3r6)


xdistorted=x+[2p1xy+p2(r2+2x2)]
ydistorted=y+[p1(r2+2y2)+2p2xy]

Distortioncoefficients=(k1k2p1p2k3)

CameraMatrix = [fx, 0, cx]
               [0, fy, cy]
               [0,  0,  1]

GOPRO_HERO3_BLACK_NARROW_MTX
"""      


'''
room 128, Edmond G. Odette Family Gallery, 32’1" x 37’7" (Floor Burger)
room 125, Richard Barry Fudger Memorial Gallery, 50’ x 31’5" (Historical)
room 502, Michael & Sonja Koerner Gallery, 55’11" x 45’10" (Contemporary)'''
# GOprop Hero 3 Black_WIDE

# Results from: http://argus.web.unc.edu/camera-calibration-database/ 
# MTX                                         f       w       h   cx     cy      a      k1    k2     t1
# GoPro Hero4 Silver	720p-120fps-narrow	1150	1280	720	640	    360	    1	-0.31	0.17	0	
# GoPro Hero3 Black	720p-120fps-narrow  	1101	1280	720	639.5	359.5	1	-0.359	0.279	0

# DST                                   FC   W       H          C                   D
# GoPro Hero3 Black	720p-60fps-wide	    4	1280	720	1.038962477337479	0.011039937655688	


# Hero4 Silver 720p 60fps (Justin's Calibration)
# Camera: [[509.22383538   0.         718.81835991]
#         [  0.         512.51171712 370.72882604]
#         [  0.           0.           1.        ]] 
# Distoriton: [[-0.11749638  0.01444089 -0.00093358 -0.00861135 -0.00078362]] Rotation: [array([[ 0.08662065],
#        [ 0.00767301],
#        [-3.13055965]]), array([[ 0.08935431],
#        [ 0.00756351],
#        [-3.12808664]]), array([[ 0.09481957],
#        [ 0.00987386],
#        [-3.12477895]]), array([[ 0.09588525],
#        [ 0.01172908],
#        [-3.12387205]]), array([[ 0.09157242],
#        [ 0.02002886],
#        [-3.12680096]]), array([[ 0.0932182 ],
#        [ 0.02060275],
#        [-3.128533  ]]), array([[ 0.08791136],
#        [ 0.0237016 ],
#        [-3.12286326]]), array([[ 0.07782842],
#        [ 0.02949126],
#        [-3.11788391]]), array([[-0.00131175],
#        [-0.01808636],
#        [ 0.02466551]])]

test = np.array([
                                    [857.48296979, 0, 968.06224829],
                                    [0 ,876.71824265, 556.37145899],
                                    [0,    0,   1],
                                ])

test_dist = np.array([-0.25761402, 0.0877086999, -0.0000256970803, -0.0000593390389])

GOPRO_HERO4_SILVER_NARROW_MTX = np.array([
                                    [1150, 0, 640],
                                    [0 ,1150, 360],
                                    [0,    0,   1],
                                ])

GOPRO_HERO4_SILVER_NARROW_DIST = np.array([-0.359, 0.279, 0, 0])
# GOPRO_HERO4_SILVER_NARROW_DST = np.array([-0.31,0.17, ,])


GOPRO_HERO3_BLACK_NARROW_MTX = np.array([
                       [1101, 0, 639.5],
                       [0 ,1101, 359.5],
                       [0,    0,   1 ],
                                ])


GOPRO_HERO3_SILVER ={"Resolution_x":720, "FOV":None}



JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_MTX = np.array([
                                         [509.22383538, 0.,             718.81835991],
                                         [0.,           512.51171712,   370.72882604],
                                         [0.,           0.,             1.          ]
                                         ])

JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_DIST = np.array([[-0.11749638,  0.01444089, -0.00093358, -0.00861135, -0.00078362]])

JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_RVEC = np.array([
                                                np.array([[0.08662065],[ 0.00767301],[-3.13055965]]),
                                                np.array([[ 0.08935431],[ 0.00756351],[-3.12808664]]),
                                                np.array([[ 0.09481957],[ 0.00987386],[-3.12477895]]), 
                                                np.array([[ 0.09588525],[ 0.01172908],[-3.12387205]]), 
                                                np.array([[ 0.09157242],[ 0.02002886],[-3.12680096]]), 
                                                np.array([[ 0.0932182 ],[ 0.02060275],[-3.128533  ]]), 
                                                np.array([[ 0.08791136],[ 0.0237016 ],[-3.12286326]]), 
                                                np.array([[ 0.07782842],[ 0.02949126],[-3.11788391]]),
                                                np.array([[-0.00131175],[-0.01808636],[ 0.02466551]])
                                            ])

# np.array(np.array([
#                             [0.08662065],[ 0.00767301],[-3.13055965]]),
#  array([[ 0.08935431],[ 0.00756351],[-3.12808664]]), 
# array([[ 0.09481957],[ 0.00987386],[-3.12477895]]), 
# array([[ 0.09588525],[ 0.01172908],[-3.12387205]]), 
# array([[ 0.09157242],[ 0.02002886],[-3.12680096]]), 
# array([[ 0.0932182 ],[ 0.02060275],[-3.128533  ]]), 
# array([[ 0.08791136],[ 0.0237016 ],[-3.12286326]]), 
# array([[ 0.07782842],[ 0.02949126],[-3.11788391]]), 
# array([[-0.00131175],[-0.01808636],[ 0.02466551]])])


def do_nothing(event,x,y,flags,params):
    pass

class room_estimation():
    def __init__(self, image=None, camera_matrix=JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_MTX, distortion_matrix=JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_DIST, rvecs=JUSTIN_HERO4_SILVER_720_MEDIUM_FOV_RVEC):

        
        self.window_name = "Room Estimation"
        cv2.namedWindow("Room Estimation")
        self.image = image
        self.show_img = self.img_copy()
        self.corners = []
        self.room_points = [] # The formatted real_locations that is used in processing data

        self.real_locations = [] # Raw input of points in 3D [(0,0,0), (1000,0 ,0), (1000, 2000, 0), (0, 2000,0)]

        self.camera_matrix = camera_matrix
        self.distioriton_matrix = distortion_matrix
        self.calibration_errors = None

        self.calibration_resolution = None

        self.rotation_vector = rvecs
        self.translation_vector = None

        # Size are in millimeters
        self.room_width = 3000  # X
        self.room_length = 3000 # Depth (Z)
        self.room_height = 3000 # Y

        self.sample_rate = 5   # estimates position every 1cm

        self.mapped_dictionary = {}
        self.mapped = {}

        self.points_data = None

        self.vector_depth = 0

        self.axis = self.set_axis(3000,1000,3000)
        self.fig = None
        self.stitch_homographies = None

        # img = self.draw_axis()
        # cv2.waitKey(1)

        # self.mapped_dictionary = self.map_2d_to_3d((self.room_width, self.room_length, self.room_height))
        
        self.show_3D_plot = True
        # while True:
        #     cv2.imshow(self.window_name, self.show_img)
        #     cv2.waitKey(1)
        # self.corner_points = []

    def set_image(self, image):
        self.image = image

    def display_room(self, image=None, axis=True, box=True, show_3d_plot=True,  with_points=False):
        while True:
            
            # self.draw_vector()
            if axis:
                self.draw_axis()
            
            if box:
                try:
                    self.draw_box()
                except:
                    pass

            # if show_3d_plot:
            #     self.show_3D_plot()
            if image is not None:
                print("Show img")
                cv2.imshow(self.window_name, image)
            else:
                cv2.imshow(self.window_name, self.show_img)
            key = cv2.waitKey(1)

            if key == ord('c'):
                break

    def set_axis(self, x, y, z):
        # axis = np.float32([[0,0,0], [0,3000,0], [3000,3000,0], [3000,0,0],
        #             [0,0,3000],[0,3000,3000],[3000,3000,3000],[3000,0,3000] ])
        self.axis = np.float32([[x,0,0], [0,z,0], [0,0,y]]).reshape(-1,3)
        return self.axis

    def set_corners(self, p1=None, p2=None, p3=None, p4=None):
        if p1:
            self.corner[0] = np.asarray([p1[0],p1[1]]).reshape(1,2,1)
        if p2:
            self.corner[1] = np.asarray([p2[0],p2[1]]).reshape(1,2,1)
        if p3:
            self.corner[2] = np.asarray([p3[0],p3[1]]).reshape(1,2,1)
        if p3:
            self.corner[3] = np.asarray([p4[0],p4[1]]).reshape(1,2,1)

    def img_copy(self):
        return deepcopy(self.image)

    def define_sides(self, sides, lengths):
        pass

    def define_room(self, width=None, length=None, height=None, offset=[0,0,0,0], corners=None, room_points=None, refine_corners=False, image=None, stitch_videos=None, calibration=None, extend=None):
        '''
        Takes variables and defines a room

        - width: and length are in mm

        - height: is the assumed height of all objects in the room

        - offset: is used when we extend the image to get all of the corners.
        NOTE: If arrows do not line up with lines you drew, the camera matrix or the distortion matrix are off

        - room_points: are the real-world defined points in mm (if not defined points are the height and width of the room)

        - refine_corners: is opencv's sub-pixel optimizer to better estimate corner locations. This is mostly used in checkerboard calibration but may be useful when manually selecting corners.

        - calibration: is the calibration saved from a video (enter a video to calibrate, if calibration has been done save results in a pickle file)

        '''


        if width is not None:
            self.room_width = width
        if length is not None:
            self.room_length = length
        if height is not None:
            self.room_height = height
        
        
        if calibration is not None:
            calibration_path = calibration[:-3] + "pickle"
            self.load_calibration(calibration_path)
        

        # Stitch images takes care of undistortion based on calibration
        if stitch_videos is not None:

            self.image, self.stitch_homographies = self.stitch_rooms(stitch_videos, key_index=3)
            self.show_img = deepcopy(self.image)

        
        # Undistort images
        self.undistort_room()

        # find extention before selecting points so you do not have to calculate offset
        # extend is the number of walls you want to extend. Each wall intersection/extention requires 2 walls, 4 estimates each.
        offsets = [0,0,0,0]
        if extend is not None:
            for i in range(extend):
                intersection = self.find_wall_intersection(degree=1)
                # offset = [top, left, bottom, right] in pixels
                self.image, offset = self.extend_image_to_corners(self.image, intersection)
                self.show_img =self.image
                # Offsets are the maximum offset values between all edges
                for index, value in enumerate(offset):
                    if offsets[index] < value:
                        offsets[index] = offset[index]


        self.image = self.superimpose_checker()
        self.show_img =self.image

        if stitch_videos is not None:
            self.stitch_trackers(stitch_videos, self.stitch_homographies, offsets)


        # creates a 4 corner room in real coordinates
        if room_points is not None:
            print("Choose your points for the defined corners (in order)")

            # Assign real world coordinates
            self.room_points = room_points

            # Assign a rectangular shape given room points (width and length)
            self.room_width, self.room_length = self.get_room_dimensions(room_points)
            
            # get 2D locations which will map to real world room points
            self.corners = self.get_room_corners(len(room_points))
            room_points = np.reshape(np.asfarray(room_points), (len(room_points),3,1))
            self.room_points = room_points

            
        # 
        elif corners is None:
            if width == None or length == None:
                room_points, self.corners = self.assign_non_square_corners()
            
            else:
                room_points = self.get_room_points(self.room_width, self.room_length)
                self.corners = self.get_room_corners()
        
        print(self.corners)

        # '''

        if refine_corners:
            # assert(image != None)
            if image is None:
                image = self.image
            # Refer to https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
            '''
            W FORSTNER. A fast operator for detection and precise location of distincs points, corners and center of circular features. In Proc. of the Intercommission Conference on Fast Processing of Photogrammetric Data, Interlaken, Switzerland, 1987, pages 281–305, 1987.
            '''
            # maxCorners = max(5000, 1)
            # # Parameters for Shi-Tomasi algorithm
            # qualityLevel = 0.01
            # minDistance = 10
            # blockSize = 3
            # gradientSize = 3
            # useHarrisDetector = False
            # k = 0.05

            src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply corner detection
            # corners = cv2.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
            #     blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
            # Draw corners detected
            # print('** Number of corners detected:', self.corners.shape[0])
            # radius = 4
            # for i in range(corners.shape[0]):
            #     cv2.circle(src_gray, (int(corners[i,0,0]), int(corners[i,0,1])), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv2.FILLED)
            
            # Set the needed parameters to find the refined corners
            winSize = (5, 5)
            zeroZone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
            # Calculate the refined corner locations
            src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.corners = cv2.cornerSubPix(src_gray, self.corners, winSize, zeroZone, criteria)

            cv2.imshow("Corner Detection", src_gray)
            cv2.waitKey(0)
            

        # print(room_points)
        # self.corners, room_points = self.assign_non_square_corners()
        # # Correct for offsets with exteneded image
        # offset = [top, left, bottom, right]
        # for index, corner in enumerate(self.corners):
        #     x,y =self.corners[index]
        #     self.corners[index] = ((x - offset[1]), (y - offset[0]))
            
        # print(room_points)
        # print(self.corners)
        key = None
        
        while True:
            test_image = deepcopy(self.image)
            print("Room Dimensions", self.room_width, self.room_length)
            room_points = self.get_room_points(self.room_width, self.room_length)
            #NOTE This is calculated after distortion is corrected
            # Because we correct for this, distortion should be "None" as we undistort already. 
            # retval, rvecs, tvecs = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)
            # retval, rvecs, tvecs = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None)
            retval, r1, t1 = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None, cv2.SOLVEPNP_IPPE, useExtrinsicGuess=False)
            retval, r2, t2 = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None, cv2.SOLVEPNP_EPNP, useExtrinsicGuess=False)
            retval, r3, t3 = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None, cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False)
            retval, r4, t4 = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None, cv2.SOLVEPNP_MAX_COUNT, useExtrinsicGuess=False)
            retval, r5, t5 = cv2.solvePnP(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None, cv2.SOLVEPNP_UPNP, useExtrinsicGuess=False)
            retval, r1, t1, inliers = cv2.solvePnPRansac(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None)
            img1, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r1, tvec=t1)
            cv2.imshow("Iterative Default", img1)
            cv2.waitKey(0)
            # Returns all solutions
            # retval, rall, tall, error = cv2.solvePnPGeneric(room_points.astype(np.float64), self.corners.astype(np.float64), self.camera_matrix, None)
            # for i in range(len(rall)):
                
            #     img1, cube, target_cube = self.draw_cuboid(image=test_image, rvec=rall[i], tvec=tall[i])
            #     cv2.imshow("Iterative Default", img1)
            #     cv2.waitKey(0)
            # # self.rotation_vector = rvecs
            # self.translation_vector = tvecs



            if key == ord('q'):
                break
            if key == ord('='):
                self.room_width += 100
            if key == ord('-'):
                self.room_width -= 100
            
            if key == ord('0'):
                self.room_length += 100
            if key == ord('9'):
                self.room_length -= 100

            point = 0

            if key == ord('1'):
                point = 1
            if key == ord('2'):
                point = 2
            if key == ord('3'):
                point = 3
            if key == ord('4'):
                point = 4
            

            self.set_axis(self.room_width, self.room_height, self.room_length)
            img1, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r1, tvec=t1)
            img2, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r2, tvec=t2)
            img3, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r3, tvec=t3)
            img4, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r4, tvec=t4)
            img5, cube, target_cube = self.draw_cuboid(image=test_image, rvec=r5, tvec=t5)

            cv2.imshow("IPPE", img1)
            cv2.imshow("EPNP", img2)
            cv2.imshow("ITERATIVE", img3)
            cv2.imshow("MAX_COUNT", img4)
            cv2.imshow("UPNP", img5)

            key = cv2.waitKey(1)

    def clear_corners(self):
        print("Clearing Corners!")
        self.corners = []
        return self.corners
    
    def get_room_corners(self, num_points=4):
        '''
        Assigns 2D Room corners which will map to the real world 3D.

        This function is a loop for the number of room points (real world) and lets you select the location they should exist in the video.

        Each point should be respective to the order that room points exist. For example, consistently clockwise starting from the bottom left.
        '''
        self.clear_corners()
        print("Assigning corners...")

        #Initializes feedback to a function
        cv2.setMouseCallback("Room Estimation", self.get_corners)
        print("Setting Callback")
        while len(self.corners) < num_points:

            # display the image and wait for a keypress
            cv2.imshow("Room Estimation", self.show_img)
            cv2.setMouseCallback("Room Estimation", self.get_corners)
            key = cv2.waitKey(1) & 0xFF
            self.connect_points(self.corners)

            # if len(self.corners) >= 4:
            #     self.show_img, cube, target_cube = self.draw_cuboid()

            if key == ord("c"):
                return self.connect_points(self.corners)
            
        self.connect_points(self.corners)
    
        self.corners = np.reshape(np.asfarray(self.corners), (num_points,2,1))
        # cv2.setMouseCallback("Room Estimation", self.draw_vector)
        return self.corners
        
    def edit_room_corner(self, index, new_pixel):
        self.corners[index][0] = new_pixel[0]
        self.corners[index][1] = new_pixel[1]
        return self.corners

    def connect_points(self, points):
        "Helper functio nthat visualizes corners"
        for index, point in enumerate(points):
            if self.room_points:
                p1 = (int(points[index][0]), int(points[index][1]))
                self.show_img = cv2.putText(self.show_img, str(self.room_points[index]), p1, fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(255,0,0), thickness=2)
            # if index+1 == len(points):
            #     # print("Last point")
            #     p1 = (int(points[index][0]), int(points[index][1]))
            #     p2 = (int(points[0][0]), int(points[0][1]))
            #     cv2.line(self.show_img, p1, p2, (255,0,0))
            if len(points) > 1 and index+1 < len(points):
                p1 = (int(points[index][0]), int(points[index][1]))
                p2 = (int(points[index+1][0]), int(points[index+1][1]))
                self.show_img=cv2.line(self.show_img, p1, p2, (255,0,0))
                

        cv2.imshow(self.window_name, self.show_img)

    def get_corners(self, event, x, y, flags, param):
        # grab references to the global variables
        # global refPt, cropping, corner_count, corner_np
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.append((x,y))
            cv2.circle(self.show_img, (x,y), 1,(0,0,255), 5, 0)
            print("CORNER !!!")

    def get_corner_with_definition(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.append((x,y))
            cv2.circle(self.show_img, (x,y), 1,(0,0,255), 5, 0)
            
            width = input("Width")
            length = input("Length")
            self.room_points.append((int(width), int(length)))
            self.connect_points(self.corners)

    def get_cube_side(self, cube, side):
        '''                      

        Returns the point set of each side of the cube in 
        Start at bottom left of the plane on the opened cube displayed, in clockwise order  (or, the perspective from the center)
        This function is to be paired before crop_plain() and the points returned from requested side should be passed on
        Example: Side 1 contains points 1,2,6,5

                                            |   5   |   
           5        6                       |_______|
           *--------*                       |       |
         / |       /|                       |   4   |
      1 *---------*2|                _______|_______|_______
        |  * 4    | *7              |       |       |       |
        | /       |/                |   1   |   2   |   3   |
      0 *---------*3                |_______|_______|_______|
        '''
        points = None

        if side == 1: # Left Wall
            points = [cube[0], cube[1], cube[5], cube[4]]
        elif side == 2: # Floor
            points = [cube[0], cube[4], cube[7], cube[3]]
        elif side == 3: # Right Wall
            points = [cube[7], cube[6], cube[2], cube[3]]
        elif side == 4: # Back wall
            points = [cube[4], cube[5], cube[6], cube[7]]
        elif side == 5: # Ceiling
            points = [cube[5], cube[1], cube[2], cube[6]]
        print(points)
        print(np.array(points))
        
        return points

    def crop_plain(self, points, plain_width, plain_length, image=None):
        '''
        Given 4 points the image will be cropped and reprojected as a rectangle.

        NOTE: Use length and height synonymously

        points = [(x1,y1), (x2,y2) ... (x4,y4)]
        '''
        if image == None:
            image = self.image
            
        h, w = image.shape[:2]

        # These are the physical dimensions of the wall and the ratio in order to calculate the perspective
        plane_ratio = plain_width/plain_length

        # Perspective points are the points labelled multiplied by the ratio to get the end values of the points
        # We start with the bottom left of the wall, which means we use max_y of the wall room

        '''
        2---3
        |   |
        1   4
        '''

        p1 = (plain_length,0)
        p2 = (0,0)
        p3 = (plain_length, plain_width)
        p4 = (0, plain_width)
        perspective_points = np.array([p1,p2,p3,p4])

        # print(points)
        # print(perspective_points)
        

        points = np.array(points, dtype=np.float32)
        perspective_points = np.array(perspective_points, dtype=np.float32)

        print(points.shape)
        print(perspective_points.shape)
        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        M = cv2.getPerspectiveTransform(points, perspective_points)

        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

        cv2.imshow("Wall Warped", warped)
        cv2.waitKey(0)

        return warped


    def project_points(self, points=[]):
        points_np = np.reshape(np.asfarray(points), (len(points),2,1))



    # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    def plot_points(event,x,y,flags,params):
        global mouseX,mouseY
        # print(params)
        mapped_points = params
        p_list = []
        x_list = []
        y_list = []
        z_list = []
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if (x,y) in mapped_points.keys():
                p_list = mapped_points[(x,y)]
                
                
                for p in p_list:
                    print(p)
                    x_list.append(p[0])
                    y_list.append(p[2])
                    z_list.append(p[1])



                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xs = x_list, ys = y_list, zs = z_list)

                ax.set_title("3D Point position in room")

                ax.set_xlabel("X (Width")
                ax.set_ylabel("Y (Height)")
                ax.set_zlabel("Z (Depth")

                plt.show()

    def get_3d_to_2d(self, point, show=True):
        '''
        Point = (X, Z, Y) in mm (NOT PIXELS)
        '''
        # np.ndarray((len(points), 3))
        # test = np.array(point[0], point[1], point[2], point[0], point[1], point[2], point[0], point[1], point[2])
        # points = np.float32(np.ndarray(points))
        point = np.float32([[point[0], point[1], point[2]]])
        # print(point.shape)
        mapped_point, jac = cv2.projectPoints(point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
        mapped_point = (int(mapped_point[0][0][0]),int(mapped_point[0][0][1]))

        if show is True:
            self.draw_axis()
            cv2.circle(self.show_img, mapped_point, 6, (0, 255, 255))
            cv2.imshow(self.window_name, self.show_img)
            cv2.waitKey(0)

        return mapped_point



    # def map_2d_to_3d(self, room_dimensions=(1000,1000,1000), step=10):
    #     '''
    #     DEPRECIATED!!! 
    #     Use get_room_3d instead.

    #     Maps the entire image to depth points with defined room
    #     '''
    #     print("Mapping points...")
    #     # Room dimensions is 1m (width) x 1m (height) x 1m(depth)
    #     w = room_dimensions[0]
    #     h = room_dimensions[1]
    #     d = room_dimensions[2]

    #     mapped = {}
    #     for i in range(0,w,self.sample_rate):
    #         print((i/step), "%")
    #         for j in range(0,1000,self.sample_rate):
    #             for k in range(0,d,self.sample_rate):
                    
    #                 point = np.float32([[i, k, j]])
                    
    #                 mapped_point, jac = cv2.projectPoints(point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
    #                 mapped_point = (int(mapped_point[0][0][0]),int(mapped_point[0][0][1]))

    #                 if mapped_point not in mapped.keys():
    #                     mapped[mapped_point] = list()
    #                     mapped[mapped_point].append((i,k,j))
    #                 else:
    #                     mapped[mapped_point].append((i,k,j))
                    
    #     # for point in mapped.keys():
    #     #     cv2.circle(self.show_img,point,1,color=(255,0,255))
    #     cv2.imshow(self.window_name, self.show_img)
    #     print("Finished mapping points")
    #     self.mapped_dictionary = mapped
    #     return mapped

    def get_3d_point(self, pixel, height=1500, camera_matrix=None, rotation_matrix=None, tvec=None, show_point=True):

        if not camera_matrix:
            camera_matrix = self.camera_matrix
        
        if not rotation_matrix:
            rotation_matrix = self.camera_matrix

        if not tvec:
            tvec = self.translation_vector
        
        if show_point:
            cv2.circle(self.show_img, pixel, 1, (255,255,0))
            cv2.imshow(self.window_name, self.show_img)
            cv2.waitKey(0)

        uv_point = np.array([pixel[0], pixel[1], 1])
        left_side = np.linalg.inv(rotation_matrix * np.identity(3) ) * np.linalg.inv(camera_matrix * np.identity(3) ) * uv_point
        right_side = np.linalg.inv(rotation_matrix * np.identity(3)) * tvec

        s = height + right_side[2][2]/left_side[2][2]

        point = np.linalg.inv(rotation_matrix * np.identity(3)) * (s*np.linalg.inv(camera_matrix* np.identity(3)) * uv_point - tvec)
        return point

    def display_3D_Plot(points, shown):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x_data = points[0]
        y_data = points[1]
        z_data = points[2]
        scatter = ax.scatter3D(x_data, y_data,z_data, cmap='Greens')
        
        if shown:
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            fig.show()


    def draw_axis(self):
        
        projected, jac = cv2.projectPoints(self.axis, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
        corner = tuple(self.corners[0].ravel().astype('int32'))
        corner2 = tuple(projected[0].ravel().astype('int32'))
        corner3 = tuple(projected[1].ravel().astype('int32'))
        corner4 = tuple(projected[2].ravel().astype('int32'))
        
        cv2.putText(self.show_img,"x",corner2,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(255,0,0))
        cv2.putText(self.show_img,"y",corner4,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
        cv2.putText(self.show_img,"z",corner3,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,0,255))

        self.show_img = cv2.arrowedLine(self.show_img, corner, corner2, (255,0,0), 2)
        self.show_img = cv2.arrowedLine(self.show_img, corner, corner4, (0,255,0), 2)
        self.show_img = cv2.arrowedLine(self.show_img, corner, corner3, (0,0,255), 2)

        return self.show_img

    def draw_cuboid(self, image=None, distortion=None, rvec=None, tvec=None):

        '''  
            Distortion is removed if image is undistorted.
            Rvec and Tvec are transformation vectors calculated by previous SolvePNP, so we use that if we have alerady done so.
            6        7
           *--------*
         / |       /|
      2 *---------*3|
        |  * 5    | *8
        | /       |/
      1 *---------*4

        '''

        cube = np.float32([ [0,0,0], #1
                            [0,0,self.room_height], #2 Y
                            [self.room_width,0,self.room_height], # 3
                            [self.room_width,0,0], #4 X

                            [0,self.room_length,0],
                            [0,self.room_length,self.room_height],
                            [self.room_width,self.room_length,self.room_height],
                            [self.room_width,self.room_length,0]
                            
                            ]).reshape(-1,3)

        # retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(cube.astype(np.float64), selected_2d.astype(np.float64), self.camera_matrix, distortion, rvec=rvec, tvec=tvec)
        # rvecs, tvecs = [None, None]


        # try:
        #     if distortion:
        #         retval, rvecs, tvecs = cv2.solvePnP(points.astype(np.float64), corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)
        #     else:
        #         retval, rvecs, tvecs =cv2.solvePnP(points.astype(np.float64), corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)
        # except:
        #     try:
        #         corners = np.reshape(np.asfarray(self.corners), (len(self.corners),2,1))
        #         retval, rvecs, tvecs = cv2.solvePnP(self.room_points.astype(np.float64), corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)
        #     except:
        #         corners = np.reshape(np.asfarray(self.corners), (len(self.corners),2,1))

        #         points = self.get_n_room_points(self.room_points[:corners.shape[0]])
        #         retval, rvecs, tvecs = cv2.solvePnP(points.astype(np.float64), corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)

                
        projected, jac = cv2.projectPoints(cube, rvec, tvec, self.camera_matrix, distortion)


        
        corner1 = tuple(projected[0].ravel().astype('int32'))
        corner2 = tuple(projected[1].ravel().astype('int32'))
        corner3 = tuple(projected[2].ravel().astype('int32'))
        corner4 = tuple(projected[3].ravel().astype('int32'))
        corner5 = tuple(projected[4].ravel().astype('int32'))
        corner6 = tuple(projected[5].ravel().astype('int32'))
        corner7 = tuple(projected[6].ravel().astype('int32'))
        corner8 = tuple(projected[7].ravel().astype('int32'))

        projected_cube = [corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8]
        
        if image is None:
            image = self.show_img

        cv2.putText(image,"x",corner4,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(255,25,200))
        cv2.putText(image,"y",corner2,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
        cv2.putText(image,"z",corner5,fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,0,255))

        image = cv2.line(image, corner1, corner2, (0,255,0), 2) # Y
        image = cv2.line(image, corner2, corner3, (255,0,0), 2)
        image = cv2.line(image, corner3, corner4, (255,0,0), 2)
        image = cv2.line(image, corner1, corner4, (255,25,200), 2) #X

        image = cv2.line(image, corner5, corner6, (255,0,0), 2)
        image = cv2.line(image, corner6, corner7, (255,0,0), 2)
        image = cv2.line(image, corner7, corner8, (255,0,0), 2)
        image = cv2.line(image, corner8, corner5, (255,0,0), 2)

        image = cv2.line(image, corner1, corner5, (0,0,255), 2) #Z
        image = cv2.line(image, corner2, corner6, (255,0,0), 2)
        image = cv2.line(image, corner3, corner7, (255,0,0), 2)
        image = cv2.line(image, corner4, corner8, (255,0,0), 2)

        return image, cube, projected_cube

    def draw_box(img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # draw pillars in blue color

        for i,j in zip(range(4),range(4,8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        return img

    def draw_point(self, img, front_pt, back_pt):
        img = cv2.circle(img, front_pt, 8,(125,0,125),2,0)
        img = cv2.drawMarker(img, back_pt, color=(255,255,0),markerType=1)
        img = cv2.line(img,front_pt, back_pt, (100,100,0), 2)
        img = cv2.circle(img, front_pt, 8,(255,255,0),2,0)
        return img

    def estimate_plane(img, corners, points, calibration_matrix, distortion_matrix):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        
        axis = np.array([[3,0,0], [0,3,0], [0,0,-3]],dtype=np.float32).reshape(-1,3)
        
        corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnPRansac(objp, corners2, calibration_matrix, distortion_matrix)
        

        print("CalibRVEC:", rvecs, "CALIBTVEC", tvecs)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, calibration_matrix, distortion_matrix)

        return imgpts, jac

    def draw_vector(self,event,x,y,flags,params):
        # self.show_img = self.img_copy()
        values = None
        x_list = []
        y_list = []
        z_list = []

        if event == 10:
            #sign of the flag shows direction of mousewheel
            if flags > 0:
                self.vector_depth += 10
            else:
                self.vector_depth -= 10


        front_point = np.float32([[x,0,y]])
        
        front, jac = cv2.projectPoints(front_point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)

        back_point = np.float32([[x,self.room_length,y]])
        back, jac = cv2.projectPoints(back_point,self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)

        floater_point = np.float32([[x, self.vector_depth, y]])
        floater, jac = cv2.projectPoints(floater_point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)

        # print(x,y)

        front_point = (front[0][0][0],front[0][0][1])
        back_point = (back[0][0][0],back[0][0][1])
        floater_point = (floater[0][0][0],floater[0][0][1])


        self.show_img = cv2.circle(self.show_img, front_point, 8,(125,0,125),2,0)
        self.show_img = cv2.drawMarker(self.show_img, back_point, color=(255,255,0),markerType=1)
        self.show_img = cv2.line(self.show_img,front_point, back_point, (100,100,0), 2)
        self.show_img = cv2.circle(self.show_img, front_point, 8,(255,255,0),2,0)
        self.show_img = cv2.circle(self.show_img, floater_point, 8,(255,0,0),2,0)

        ifp = (int(front_point[0]), int(front_point[1]))
        if (x,y) in self.mapped_dictionary.keys():
            values = self.mapped_dictionary[(x,y)]
            for v in values:
                self.show_img = cv2.drawMarker(self.show_img, (x,y), color=(0,0,255),markerType=3)
                x_list = []
                y_list = []
                z_list = []
                for v in values:
                    x_list.append(v[0])
                    y_list.append(v[2])
                    z_list.append(v[1])

        cv2.imshow(self.window_name,self.show_img)
        cv2.waitKey(1)
        if self.show_3D_plot == True:
            plt.ion


            if self.fig is None:
                self.fig = plt.figure(1)
                self.ax = ax = plt.axes(projection='3d')
                self.front_plt = ax.scatter3D(front_point[0], front_point[1], 0)
                self.back_plt = ax.scatter3D(back_point[0],back_point[1], self.vector_depth)
                self.float_plt = ax.scatter3D(floater_point[0],floater_point[1],self.vector_depth)
                self.line_plt = ax.plot3D([front_point[0],back_point[0]], [front_point[1],back_point[1]], [0,self.room_length], color='teal')
                
                # plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
                self.ax.set_xlim(-self.room_width,self.room_width)
                self.ax.set_ylim(-self.room_height,self.room_width)
                self.ax.set_zlim(-self.room_length,self.room_width)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.fig.show()
            else:

                
                
                self.ax.cla()
                if values:
                    self.points_plt = self.ax.scatter3D(x_list,y_list,z_list, marker="D", color="red")
                # plt.quiver((0,0,0), (0,1,0), , color=['r','b','g'], scale=21)
                self.ax.set_xlim(-self.room_width,self.room_width)
                self.ax.set_ylim(-self.room_height,self.room_width)
                self.ax.set_zlim(-self.room_length,self.room_width)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
                self.float_plt = self.ax.scatter3D(floater_point[0],floater_point[1],self.vector_depth, marker="o", color="blue")
                self.front_plt = self.ax.scatter3D(front_point[0], front_point[1], 0, marker=("o"), color="cyan")
                self.back_plt = self.ax.scatter3D(back_point[0],back_point[1], self.room_length, marker="x", color="cyan")
                self.line_plt = self.ax.plot3D([front_point[0],back_point[0]], [front_point[1],back_point[1]], [0,self.room_length], color='teal')
                self.quiver = self.ax.quiver([0,0,0], [0,0,0], [0,0,0], [self.room_width,0,0], [0,self.room_height,0], [0,0,self.room_length], length=0.1, normalize=False)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

        return self.show_img


    # def get_room_3d(self, width_start=-1000, width_stop=3000, width_step=2, length_start=-100, length_stop=5000, length_step=2, height=1000):
    def get_room_3d(self, limits, height, step, save_filename=None):
        min, max = limits
        
        points = np.mgrid[min[0]:max[0]:step, min[1]:max[1]:step, height:height+1:1].reshape(3,-1).T.astype(np.float) # all 40 inches
        print("Projecting points with size", points.shape)
        mapped_pixels, jac = cv2.projectPoints(points, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)

        print("Mapping...")
        mapped = {}
        for index, pixel in enumerate(mapped_pixels):
            
            point = points[index]
            pixel = (int(pixel[0][0]), int(pixel[0][1]))
            # image = cv2.circle(image, pixel, 1, (1,1,1))
            if pixel in mapped.keys():
                mapped[pixel].append(point)
            else:
                mapped[pixel] = [point]
        self.mapped = mapped
        print("Mapping Complete.")
        return mapped

    def save_room_3d(self, path, filename, mapped):
        print("Saving mapped points...")
        if path is not None:
            with open(filename, 'wb') as f:
                pickle.dump(mapped, f)
                f.close()
        print("Save Complete.")
    
    def load_room_3d(self, filename):
        with open(filename, 'rb') as f:
            mapped = pickle.load(f)
            f.close()
        return mapped

    def get_room_points(self, width, length, height=0):
        print("Creating room points")
        """
        Returns a matrix for real world points.
        These points start on the bottom left and are assigned as a rectangle in a clockwise direction
        
        (width)
        1------2 
        |      |
        |      | (length)
        0      3

        We may include Height, but this is not included.

        Format of points (X,Z,Y)
        """

        objectPoints = np.array(
            [
                [[0.0],[0.0],[float(height)]],
                [[0.0],[float(length)],[float(height)]],
                [[float(width)],[float(length)],[float(height)]],
                [[float(width)],[0.0],[float(height)]],
            ]
        ) 
        print(objectPoints.shape)
        return objectPoints
    
    def get_n_room_points(self, n_locations=None, height=0):
        '''
        N_locations are known locations along the floor (Non-square rooms)
            5000mm
        --------------  
               3-----4  
               |     |
               |     |
        1------2     | 5000mm  (Length)
        |            |
        |            |
        0------------5
            (Width)
        Where (width,length) in mm
        0 = (0,0)
        1 = (0, 2500)
        2 = (2500, 2500)
        3 = (2500, 5000)
        4 = (5000, 5000)
        5 = (5000, 0)
        '''
        if n_locations == None:
            n_locations = self.room_points
        # objectPoints = np.reshape(np.asfarray(n_locations), (len(n_locations),3,1))
        objectPoints = np.ndarray((len(n_locations), 3, 1))
        for index, location in enumerate(n_locations):
            # for index2, dim in enumerate(location):
            print(location)
            objectPoints[index][0][0] = location[0] # X 
            objectPoints[index][1][0] = location[1] # Z
            objectPoints[index][2][0] = height      # Y
            # print(objectPoints[index][:])
        print(objectPoints)
        return objectPoints

    def assign_non_square_corners(self, num_points=inf):
        '''
        Assigns points around the room with any number of known coordinates.

        Assign point
        Set width and length of known coordinate.
        Origin should be 0,0
        Sets the dimentions of the room to the largest dimensions
        '''

        print("Assigning corners...")
        #Initializes feedback to a function
        cv2.setMouseCallback("Room Estimation", self.get_corners)
        while len(self.corners) < num_points:
            
            # if len(self.corners) >= 4:
            #     print("SHOWING ROOM")
            #     self.show_img, cube, target_cube = self.draw_cuboid()
            # print(len(self.corners))

            # display the image and wait for a keypress
            cv2.setMouseCallback("Room Estimation", self.get_corner_with_definition)



            cv2.imshow("Room Estimation", self.show_img)
            key = cv2.waitKey(1) & 0xFF
            self.connect_points(self.corners)



            if key == ord("c"):
                if len(self.corners) < 4:
                    print("You need at least 4 points")
                else:
                    # break
                    self.corners = np.reshape(np.asfarray(self.corners), (len(self.corners),2,1))
                    self.room_points = self.get_n_room_points()

                    # set measurments to maximum sizes
                    self.room_width = 0
                    self.room_length = 0
                    for point in self.room_points:
                        width = int(point[0][0])
                        length = int(point[1][0])
                        if width > self.room_width:
                            self.room_width = width
                        if length > self.room_length:
                            self.room_length = length

                    return self.room_points, self.corners

        self.connect_points(self.corners)

        # set measurments to maximum sizes 
        for point in self.room_points:
            width = int(point[0][0])
            length = int(point[1][0])
            if width > self.room_width:
                self.room_width = width
            if length > self.room_length:
                self.room_length = length
        
        # Set measurments to numpy arrays
        self.corners = np.reshape(np.asfarray(self.corners), (len(self.corners),2,1))
        self.room_points = self.get_n_room_points()
        # cv2.setMouseCallback("Room Estimation", self.draw_vector)
        return self.room_points, self.corners

    # def define_walls(self, width, length, height=0):
    #     print("Creating room points")
    #     """
    #     Returns a matrix for real world points.
    #     These points start on the bottom left and are assigned as a rectangle in a clockwise direction
        
    #     (width)
    #     1------2 
    #     |      |
    #     |      | (length)
    #     0      3

    #     We may include Height, but this is not included.

    #     """

    #     objectPoints = np.array(
    #         [
    #             [[0.0],[0.0],[float(height)]],
    #             [[0.0],[float(length)],[float(height)]],
    #             [[float(width)],[float(length)],[float(height)]],
    #             [[float(width)],[0.0],[float(height)]],
    #         ]
    #     )
    #     return objectPoints

    def find_3d_limits(self, height=1000):
        print("Finding Limits...")

        edge_buffer = 5

        x_limit = self.show_img.shape[1] - edge_buffer
        y_limit = self.show_img.shape[0] - edge_buffer
        x_min = 5
        y_min = 5
        
        bottom_left = (0,0,height)
        top_left = (0,self.room_length,height)
        top_right = (self.room_width,self.room_length,height)
        bottom_right = (self.room_width,0,height)

        test_step = 100 # 10cm
        max_distance_test = 12000 # 12m max
        '''
        Get top left
        '''
        #test left limits
        pixel_y = inf
        pixel_x = inf

        # test_top left
        x_lim = top_left[0]
        y_lim = top_left[1]
        z_lim = 0


        
        while abs(x_lim) <= float(max_distance_test) and abs(y_lim) <= float(max_distance_test) and abs(z_lim):
            print(x_lim, y_lim, z_lim, max_distance_test)
            # grab x and y of 
            x_lim = top_left[0]
            y_lim = top_left[1]
            z_lim = top_left[2]


            point_lim = np.float32([x_lim, y_lim , height])

            mapped_point, jac = cv2.projectPoints(point_lim, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
            pixel_y = mapped_point[0][0][1]
            pixel_x = mapped_point[0][0][0]
            if pixel_x >= edge_buffer:
                top_left = (top_left[0] - test_step, top_left[1], top_left[2])
            else:
                top_left = (top_left[0] + test_step/2, top_left[1], top_left[2])
            # if pixel_x < 10:
            #     print("X reached")
                
            
            if pixel_y >= edge_buffer:
                top_left = (top_left[0], top_left[1] + test_step, top_left[2])
            else:
                top_left = (top_left[0], top_left[1] - test_step/2, top_left[2])
            # if pixel_y < 10:
            #     print("Y reached")
            try:
                self.show_img = cv2.circle(self.show_img, (mapped_point[0][0][0], mapped_point[0][0][1]), 1,(255,0,0),1,0)
                cv2.imshow('Points', self.show_img)
                cv2.waitKey(1)
            except:
                pass
            # print(pixel_x, pixel_y,top_left)

            if pixel_x < edge_buffer and pixel_y < edge_buffer:
                break
        print("Top Left:", top_left)

        '''
        Get top right
        '''
        pixel_y = inf
        pixel_x = -inf
        x_lim = top_right[0]
        y_lim = top_right[1]
        z_lim = top_right[2]
        # top_right = top_left
        while abs(x_lim) <= max_distance_test and abs(y_lim) <= max_distance_test and abs(z_lim) <= max_distance_test:
            print(x_lim, y_lim)
            # grab x and y of 
            x_lim = top_right[0]
            y_lim = top_right[1]
            z_lim = top_right[2]

            if x_lim <= max_distance_test or y_lim <= max_distance_test or abs(z_lim) <= max_distance_test:

                point_lim = np.float32([x_lim, y_lim , height])

                mapped_point, jac = cv2.projectPoints(point_lim, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
                pixel_y = mapped_point[0][0][1]
                pixel_x = mapped_point[0][0][0]
                if pixel_x <= x_limit:
                    top_right = (top_right[0] + test_step, top_right[1], top_right[2])
                else:
                    top_right = (top_right[0] - test_step/2, top_right[1], top_right[2])

                if pixel_y >= edge_buffer:
                    top_right = (top_right[0], top_right[1] + test_step, top_right[2])
                else:
                    top_right = (top_right[0], top_right[1] - test_step/2, top_right[2])
                # if pixel_y > y_limit:
                #     print("Y reached")

            try:
                self.show_img = cv2.circle(self.show_img, (mapped_point[0][0][0], mapped_point[0][0][1]), 1,(255,0,0),1,0)
                cv2.imshow('Points', self.show_img)
                cv2.waitKey(1)
            except:
                pass
                
                if pixel_x >= x_limit and pixel_y < edge_buffer:
                    break
        print("Top Right", top_right)

        '''
        Get bottom left
        '''
        pixel_y = -inf
        pixel_x = -inf
        x_lim = bottom_left[0]
        y_lim = bottom_left[1]
        z_lim = bottom_left[2]
        # bottom_left = top_left
        while abs(x_lim) <= max_distance_test and abs(y_lim) <= max_distance_test and abs(z_lim) <= max_distance_test:
            # grab x and y of 
            x_lim = bottom_left[0]
            y_lim = bottom_left[1]

            point_lim = np.float32([x_lim, y_lim , height])

            mapped_point, jac = cv2.projectPoints(point_lim, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
            pixel_y = mapped_point[0][0][1]
            pixel_x = mapped_point[0][0][0]
            if pixel_x >= edge_buffer:
                bottom_left = (bottom_left[0] - test_step, bottom_left[1], bottom_left[2])
            else:
                bottom_left = (bottom_left[0] + test_step/2, bottom_left[1], bottom_left[2])


            
            if pixel_y <= y_limit:
                bottom_left = (bottom_left[0], bottom_left[1] - test_step, bottom_left[2])
            else:
                bottom_left = (bottom_left[0], bottom_left[1] + test_step/2, bottom_left[2])
            # if pixel_y > y_limit:
            #     print("Y reached")

            try:
                self.show_img = cv2.circle(self.show_img, (mapped_point[0][0][0], mapped_point[0][0][1]), 1,(255,0,0),1,0)
                cv2.imshow('Points', self.show_img)
                cv2.waitKey(1)
            except:
                pass
            # print(pixel_x, pixel_y,bottom_left)

            if pixel_x < edge_buffer and pixel_y > y_limit:
                break
        print("Bottom Left:", bottom_left)

        '''
        Get bottom right
        '''
        pixel_y = -inf
        pixel_x = -inf
        # bottom_right = bottom_left
        x_lim = bottom_right[0]
        y_lim = bottom_right[1]
        while abs(x_lim) <= max_distance_test and abs(y_lim) <= max_distance_test and abs(z_lim) <= max_distance_test:
            # grab x and y of 
            x_lim = bottom_right[0]
            y_lim = bottom_right[1]

            point_lim = np.float32([x_lim, y_lim , height])

            mapped_point, jac = cv2.projectPoints(point_lim, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
            pixel_y = mapped_point[0][0][1]
            pixel_x = mapped_point[0][0][0]
            if pixel_x <= x_limit:
                bottom_right = (bottom_right[0] + test_step, bottom_right[1], bottom_right[2])
            else:
                bottom_right = (bottom_right[0] - test_step/2, bottom_right[1], bottom_right[2])
   
            if pixel_y <= y_limit:
                bottom_right = (bottom_right[0], bottom_right[1] - test_step, bottom_right[2])
            else:
                bottom_right = (bottom_right[0], bottom_right[1] + test_step/2, bottom_right[2])
            # if pixel_y > y_limit:
            #     print("Y reached")

            try:
                self.show_img = cv2.circle(self.show_img, (mapped_point[0][0][0], mapped_point[0][0][1]), 1,(255,0,0),1,0)
                cv2.imshow('Points', self.show_img)
                cv2.waitKey(1)
            except:
                pass
            # print(pixel_x, pixel_y,bottom_right)

            if pixel_x >= x_limit and pixel_y >= y_limit:
                break
        print("Bottom Right:", bottom_right)
        min = np.array((top_left, bottom_left, top_right, bottom_right))
        min = np.amin(min, 0)

        max = np.array((top_left, bottom_left, top_right, bottom_right))
        max = np.amax(max, 0)

        print(min, max)
        return (min, max), (top_left, bottom_left, top_right, bottom_right)


    def project_point(self, event,x,y,flags,params):
        pass
    
    def get_depth(self, event, x, y, p1, p2):
        """
        get z position given x and y.
        """
        # if event == cv2.EVENT_LBUTTONDOWN:
        # resets the image so display does not accumulate
        self.show_img = self.img_copy()
        min = np.amin(np.array(self.mapped[(x,y)]), 0)
        max = np.amax(np.array(self.mapped[(x,y)]), 0)
        self.show_img = cv2.putText(self.show_img, str((min, max)), (x,y), fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(255,0,0), thickness=2)
        
        print(x,y, "->", min[:2], max[:2])

    def estimate(self):
        '''
        Depreciated...

        NOTE: define room first
        
        This defines room limits and projects 3D points onto the screen
        '''
        print("Finding Limits...")
        limits, _ = self.find_3d_limits()
        print("Limits found...")

        print("Estimating 3D space within limits...")
        self.get_room_3d(limits, height=self.room_height, step=2)
        print("Estimation complete..")

        cv2.imshow("Points", self.show_img)
        cv2.setMouseCallback('Points', self.get_depth)
        cv2.waitKey(0)
    
    def get_pixel_location(self, pixel):
        '''
        Returns the pixel location in 3D
        '''
        return self.mapped[pixel]


    def poly_fit_wall(self, samples, pixels_past_extent=100,horizontal_extent=None, degree=2):
        '''
        Samples exist in format np.array([x,y,z], [x,y,z])

        Used to find intersection points.

        Utilized to define points in the grid.
        '''

        samples = np.reshape(samples, (samples.shape[0], 2))
        if horizontal_extent is None:
            min = np.amin(samples, 0)[0]
            max = np.amax(samples, 0)[0]
            horizontal_extent = np.linspace(min  - pixels_past_extent, max + pixels_past_extent)


        # Polyfit a model on the samples x,y then create a line with the exten of the horizontal extent we provide
        fit = np.polyfit(samples[:,0], samples[:,1], degree)
        
        polynomial_equation = np.poly1d(fit)
        print("Equation", polynomial_equation)
        
        pred_line = polynomial_equation(horizontal_extent)
        pred_line = np.reshape(pred_line, (len(pred_line),1))
        horizontal_extent = np.reshape(horizontal_extent, (len(horizontal_extent),1))
        # print(pred_line)
        return np.hstack((horizontal_extent, pred_line)), polynomial_equation

    def find_intersection(self, line_equation1, line_equation2):
        """
        Returns the (x,y) location of the intersection of 2 lines given m and b of both lines from slope intercept form (m*x + b)
        """
        m1 = line_equation1.coef[0]
        b1 = line_equation1.coef[1]

        m2 = line_equation2.coef[0]
        b2 = line_equation2.coef[1]

        # x intersection
        xi = (b1-b2) / (m2-m1)

        #y intersection
        yi = m1 * xi + b1

        print("INTERSECTION", xi, yi)
        return (xi, yi)

    def make_interpolater(left_min, left_max, right_min, right_max): 
        '''
        Depreciated.
        Supposed to be used to divide a grid space given pixels.
        Interpolates values in between min and max.
        '''
        # Figure out how 'wide' each range is  
        leftSpan = left_max - left_min  
        rightSpan = right_max - right_min  

        # Compute the scale factor between left and right values 
        scaleFactor = float(rightSpan) / float(leftSpan) 

        # create interpolation function using pre-calculated scaleFactor
        def interp_fn(value):
            return right_min + (value-left_min)*scaleFactor

        return interp_fn

    # def maprange(a, b, s):
    #     (a1, a2), (b1, b2) = a, b
    #     return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

    def superimpose_checker(self, size=21):
        '''
        This draws the grid from "define_perspective_grid" onto an image. 

        This grid is given a "size" that is the number of lines present. 
        This function utilizes draw_grid which is a 1080x1080 resolution image with cv2.lines drawn on it.

        This grid image is then registered onto defined points from "define_perspective_grid" given the homograpy.
        '''
        
        temp = deepcopy(self.image)
        # grid = cv2.imread("K:/Github/PeopleTracker/Grid.jpg")
        print("\n \n Creating Grid...")
        corners = self.get_room_corners()

        grid = self.draw_grid()
        h, w, _ = grid.shape
        src = np.array(((0,0), (w-1,0), (w-1,h-1), (0,h-1)))

        h, status = cv2.findHomography(src, corners)
        self.clear_corners()
        print("+/- to change grid, Q to quit.")
        while True:
            grid = self.draw_grid(grid_shape=(size,size))
            registered = cv2.warpPerspective(grid, h, (temp.shape[1], temp.shape[0]))
            registered = cv2.addWeighted(temp,1,registered,0.3,0)
            cv2.imshow("Registered", registered)
            k = cv2.waitKey(1)
            if k == ord("="):
                size += 1
            if k == ord('-'):
                size -= 1
            if k == ord('q'):
                break

            

        return registered

    def define_perspective_grid(self, division=10, extend = 500):
        '''
        Assign 4 corners on an image where a grid will be projected into that space. This only selects the points and draws the edges. This should be called before superimpose_checker.

        See superimpose_checker for definition of the grid and how it works.
        '''
        wall = self.get_room_corners()

        # Describe a line with the two end points
        fit1, equation1 = self.poly_fit_wall(wall, pixels_past_extent=1000, degree=1)
        line_1 = (fit1[0], fit1[-1])
        self.clear_corners()

        wall = self.get_room_corners()
        fit2, equation2 = self.poly_fit_wall(wall, pixels_past_extent=1000, degree=1)
        line_2 = (fit2[0], fit2[-1])

        intersection = self.find_intersection(equation1, equation2)
        self.show_img = cv2.circle(self.show_img, (int(intersection[0]), int(intersection[1])), 1,(0,255,0), 5, 0)
        print(intersection)
        
        # self.make_interpolater()
        # x1i = interpolate.interp1d(line_1[0],line_1[1])
        # y1i = interpolate.interp1d(line_2[0],line_2[1])

        # x2i = interpolate.interp1d(line_1[0],line_2[0])
        # y2i = interpolate.interp1d(line_1[1],line_2[1])

        edge, step1 = np.linspace(line_1[0] - extend, line_2[0] + extend, division, endpoint=True, retstep=True)
        edge2, step2 = np.linspace(line_1[1] - extend, line_2[1] + extend, division, endpoint=True, retstep=True)
        
        for i in range(len(edge)):
            start = (int(edge[i][0]),int(edge[i][1]))
            end = (int(intersection[0]), int(intersection[1]))
            self.show_img = cv2.line(self.show_img, start, end, (255,0,0))


            start = (int(edge2[i][0]),int(edge2[i][1]))
            end = (int(intersection[0]), int(intersection[1]))

            self.show_img = cv2.line(self.show_img, start, end, (255,0,0))

            if i % 10 == 0:
                self.show_img = cv2.line(self.show_img, start, end, (0,100,0), thickness=2)
            cv2.imshow("test_interp", self.show_img)
            cv2.waitKey(1)

        self.clear_corners()


    def find_wall_intersection(self, number_of_walls=2, samples_per_wall=4, extent=10000, intersection_pairs=[], degree=1):
        '''
        Using samples along visible parts of walls, it is possible to fit the curve and extend the visible location of the wall


        intersection_pairs is a list of pairs of lines which are to be checked for intersections
        Because we only check the current, previous and beginning and last, we leave out all other instances so a specification of which pairs are desired is optional
        [(0,3), (1,3)] 
        '''
        fit_list = []
        corners = []
        # while True:
        for wall in range(number_of_walls):
            print("Press C when done, 4 points per wall")
            self.clear_corners()
            print("Finding Wall intersection")


            wall = self.get_room_corners()

            fit, equation = self.poly_fit_wall(wall, pixels_past_extent=extent, degree=degree)
            fit_list.append(equation)
            if len(fit_list) >= 2:
                for index, f1 in enumerate(fit_list):

                    # test only when multiple lines exist
                    if index > 0:

                        #look for intersection between current and previous
                        intersection = self.find_intersection(fit_list[index],fit_list[index-1])
                        if not math.isnan(intersection[0]):
                            corners.append(intersection)
                            cv2.circle(self.show_img, (int(intersection[0]), int(intersection[1])), 1,(0,255,0), 5, 0)

                    # test between the beginning and end
                    if index+1 == len(fit_list):
                        intersection = self.find_intersection(fit_list[index],fit_list[0])
                        if not math.isnan(intersection[0]):
                            corners.append(intersection)
                            cv2.circle(self.show_img, (int(intersection[0]), int(intersection[1])), 1,(0,255,0), 5, 0)

            end_1 = fit[0]
            end_2 = fit[-1]
            print(end_1, end_2)
            # print(fit.tolist())
            
            self.connect_points(list(fit.astype(np.int).tolist()))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                break
        
        # Add all of intersection_pairs that are specified
        for intersection in intersection_pairs:
            intersection = self.find_intersection(fit_list[index],fit_list[0])
            corners.append(intersection)
            cv2.circle(self.show_img, (int(intersection[0]), int(intersection[1])), 1,(0,255,0), 5, 0)

        # self.display_room()
        return corners
        # self.connect_points(fit)

    def get_checkerboard_shape(self, checkerboard_images):
        print("detecting checkerboard shape...")
        found = False
        for i in range(4,10):
            
            if found:
                break
            for j in range(4,10):
                print(i,j)
                try:
                    retval, corners = cv2.findChessboardCorners(checkerboard_images[0], (i,j))
                    found = retval
                except:
                    found=False
                if found:
                    checkerboard_grid = (i,j)
                    # break
        print("Checkerboard shape:", checkerboard_grid)
        return checkerboard_grid

    def fisheye_calibrate(self, checkerboard_images, checkerboard_grid=None, out=None):
        # Checkboard dimensions
        if checkerboard_grid is None:
            CHECKERBOARD = self.get_checkerboard_shape(checkerboard_images)
        else:
            CHECKERBOARD = checkerboard_grid

        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        imgshape = None
        for image in checkerboard_images:

            height, width = image.shape[:2]

            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            cv2.imshow('img',image)
            cv2.waitKey(1)
            imgshape = image.shape
            retval, corners = cv2.findChessboardCorners(image, CHECKERBOARD)
            # cv2.cornerSubPix(checkerboard_image, corners)

            # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize)
            objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
            prev_img_shape = None
	 


            if retval == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(image, corners, (11,11),(-1,-1), subpix_criteria)  
                imgpoints.append(corners2)
                # Draw and display the corners

                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, retval)
                cv2.imshow("Calibration", image)
                cv2.waitKey(1)

        mean_error = 0
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
        mean_error = total_error/len(objpoints)

        print(mean_error)
        cv2.destroyAllWindows()

        # calculate K & D
        N_imm = len(checkerboard_images)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, imgshape[::-1], K, D, flags=calibration_flags, criteria=criteria)
        self.camera_matrix = K
        self.distioriton_matrix = D
        self.rotation_vector = rvecs
        self.translation_vector = tvecs

        if out :
            with open(out, 'wb') as f:
                pickle.dump(K, f)
                pickle.dump(D, f)
                pickle.dump(rvecs, f)
                pickle.dump(tvecs, f)
                pickle.dump(error, f)
                f.close()

                

    def checkerboard_calibrate(self, checkerboard_images, checkerboard_grid=None, out=None):
        # Will look at what this means after
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW + cv2.

        calibration_flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 

        imgshape = None
    
        checkerboard_grid = self.get_checkerboard_shape(checkerboard_images)
        height, width = checkerboard_images[0].shape[:2]

        for image in checkerboard_images:

            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            cv2.imshow('img',image)
            cv2.waitKey(1)
            imgshape = image.shape
            retval, corners = cv2.findChessboardCorners(image, checkerboard_grid)
            # cv2.cornerSubPix(checkerboard_image, corners)

            # retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize)
            objp = np.zeros((1, checkerboard_grid[0] * checkerboard_grid[1], 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:checkerboard_grid[0], 0:checkerboard_grid[1]].T.reshape(-1, 2)
            prev_img_shape = None
	 


            if retval == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(image, corners, (11,11),(-1,-1), criteria)  
                imgpoints.append(corners2)
                # Draw and display the corners

                image = cv2.drawChessboardCorners(image, checkerboard_grid, corners2, retval)
                cv2.imshow("Calibration", image)
                cv2.waitKey(1)
        cv2.destroyAllWindows()



        # cv2.calibrateCameraExtended()
        print("Calibrating... ", len(objpoints), " images")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape[::-1], None, None, flags=calibration_flags, criteria=criteria)
        
        mean_error = 0
        total_error = 0
        error_list = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
            error_list.append(error)
        mean_error = total_error/len(objpoints)

        print(mean_error)
        self.camera_matrix = mtx
        self.distioriton_matrix = dist
        self.rotation_vector = rvecs
        self.translation_vector = tvecs
        self.error_list = error_list
        self.calibration_resolution = (width, height)
        self.save_calibration(out, mtx, dist, rvecs, tvecs, error_list, self.calibration_resolution)
        print("Camera:" ,mtx, "Distoriton:", dist)
        return mtx, dist, rvecs, tvecs, mean_error

    def save_calibration(self, file, mtx, distortion, rvecs, tvecs, error_list, resolution):
        if file :
            with open(file, 'wb') as f:
                pickle.dump(mtx, f)
                pickle.dump(distortion, f)
                pickle.dump(rvecs, f)
                pickle.dump(tvecs, f)
                pickle.dump(error_list, f)
                pickle.dump(resolution, f)
                f.close()

    def load_calibration(self, file):
        """
        Loads in order
        1 Camera matrix
        2 Distortion Matrix
        3 Rvecs
        4 Tvecs
        """

        with open(file, 'rb') as f:
            mtx = pickle.load(f)
            dist =pickle.load(f)
            rvecs =pickle.load(f)
            tvecs =pickle.load(f)
            error_list = pickle.load(f)
            calibration_resolution = pickle.load(f)
            f.close()

        print(mtx)
        self.camera_matrix = mtx
        self.distioriton_matrix = dist
        self.rotation_vector = rvecs
        self.translation_vector = tvecs
        self.error_list = error_list
        self.calibration_resolution
        return mtx, dist, rvecs, tvecs, error_list, calibration_resolution

    def collect_frames(self, video_source, start_frame=1, skip=15, total_frames=None):
        """
        Collects the images for stitching
        """
        print(video_source)
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        print("Collecting Frames...")
        if total_frames == None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #setup cv2 capture from video
        cap = cv2.VideoCapture(video_source)
        frames = []
        #set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # print("Starting at frame: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        last = 0
        while (cap.get(cv2.CAP_PROP_POS_FRAMES)+ skip-1) < total_frames:
            print(cap.get(cv2.CAP_PROP_POS_FRAMES), "/", total_frames)
            last = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print(frames)
            #read the image from that skipped frame
            ret, frame = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip-1)

            if last == cap.get(cv2.CAP_PROP_POS_FRAMES):
                # print("NEXT")
                next=10
                last = cap.get(cv2.CAP_PROP_POS_FRAMES)
                while last == cap.get(cv2.CAP_PROP_POS_FRAMES):
                    next += 10
                    print(next)
                    ret, frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + next + skip-1)
            #set current frame to the next n-skipped frames
            # print(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if ret:
                # print(ret)
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #     break
                # cv2.imshow('frame', frame)
                #append the frames to be processed
                frames.append(frame)

                cv2.imshow("FrameCollection", frame)
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        return frames

    def get_calibration_ratio(self, video_resolution, calibration_resolution):
        w_ratio = video_resolution[0]/calibration_resolution[0]
        h_ratio = video_resolution[1]/calibration_resolution[1]
        return w_ratio, h_ratio

    def undistort_room(self, image=None, points_data=None, camera_matrix=None, distortion=None, show=False, fisheye=False, calibration_resolution=None):
        '''
        '''
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        if distortion is None:
            distortion = self.distioriton_matrix

        if image is None: 
            image = self.image
        
        if calibration_resolution is None:
            calibration_resolution = self.calibration_resolution
        
        width = image.shape[1]
        height = image.shape[0]

        
        if fisheye:
            img_dim = image.shape[:2][::-1]  
            scaled_K = camera_matrix * img_dim[0] / width
            scaled_K[2][2] = 1.0  

            '''
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, distortion,
                                                                        img_dim, np.eye(3), balance=0, fov_scale=1)

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, distortion, np.eye(3),
                                                                    new_K, img_dim, cv2.CV_16SC2)
            undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            '''

            '''
            # New Camera matrix is 
            newcameramatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, distortion, img_dim, np.eye(3), balance=0, fov_scale=1)
            undistorted_image = cv2.fisheye.undistortImage(image, camera_matrix, distortion, None, newcameramatrix)         
            '''             
            # '''
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion, np.eye(3), camera_matrix, (width,height), cv2.CV_16SC2)
            undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # '''
        else:
            # self.image.shape[1]
            # h1, w1, _ = image.shape
            # #Account for resolution differences between calibration and image input
            # w_ratio, h_ratio = self.get_calibration_ratio((w1,h1), calibration_resolution)
            # print(w_ratio, h_ratio)

            # camera_matrix[0][0] = camera_matrix[0][0] * w_ratio
            # camera_matrix[1][1] = camera_matrix[1][1] * h_ratio
            # camera_matrix[0][2] = camera_matrix[0][2] * w_ratio
            # camera_matrix[1][2] = camera_matrix[1][2] * h_ratio

            # for index, d in enumerate(distortion[0]):
            #     distortion[0][index] = [0][index] * w_ratio
            
            newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (width, height), 1, (width, height))
            if points_data is not None:


                #NOTE Solution to this being wrong, we should just calibrate the camera in the same resolution as our videos
                x = data['Pixel_Loc_x'].to_numpy()
                y = data['Pixel_Loc_y'].to_numpy()
                points = np.vstack((y, x)).T

                # d2 = distortion
                # for index, d in enumerate(distortion[0]):
                # d2 = distortion[0] * w_ratio

                pts_r_norm = cv2.undistortPoints(points, camera_matrix, distortion, None, newcameramatrix)

                new_x = pts_r_norm.T[1]
                new_y = pts_r_norm.T[0]

                data['undistorted_x'] = new_x[0].tolist()
                data['undistorted_y'] = new_y[0].tolist()
                self.points_data = data
            undistorted_image = cv2.undistort(image, camera_matrix, distortion, None, newcameramatrix)




            # if show:
            #     cv2.imshow("undistorted", undistorted_image)
            #     cv2.waitKey(0)

        if show:
            cv2.imshow("undistorted", undistorted_image)
            cv2.waitKey(0)
        return undistorted_image

    def undistort_tracker_data(self,data, image=None, camera_matrix=None, distortion=None):
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        if distortion is None:
            distortion = self.distioriton_matrix
        
        if image is None:
            image = self.image

        #Image dimensions
        width = image.shape[1]
        height = image.shape[0]


        x = data['Pixel_Loc_x'].to_numpy()
        y = data['Pixel_Loc_y'].to_numpy()

        print((width, height), self.calibration_resolution)


        points = np.vstack((y, x)).T
        
        # h1, w1, _ = image.shape
        # # Account for resolution differences between calibration and image input
        # w_ratio, h_ratio = self.get_calibration_ratio((w1,h1), self.calibration_resolution)
        # print(w_ratio, h_ratio)

        # camera_matrix[0][0] = camera_matrix[0][0] * w_ratio
        # camera_matrix[1][1] = camera_matrix[1][1] * h_ratio
        # camera_matrix[0][2] = camera_matrix[0][2] * w_ratio
        # camera_matrix[1][2] = camera_matrix[1][2] * h_ratio

        newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (width, height), 1, (width, height))
        # numpy_points = np.reshape(np.asfarray(points), (len(points),2,1))



        pts_r_norm = cv2.undistortPoints(points, camera_matrix, distortion, None, newcameramatrix)

        new_x = pts_r_norm.T[1]
        new_y = pts_r_norm.T[0]


        data['undistorted_x'] = new_x[0].tolist()
        data['undistorted_y'] = new_y[0].tolist()

        # for i in range(new_x.shape[0]):
        #     # print('showing')
        #     p = (int(new_x[0][i]),int(new_y[0][i]))
        #     cv2.circle(self.show_img,p,1,color=(0,255,255), thickness=4)
        # cv2.imshow("Projected", self.show_img)
        # cv2.waitKey(0)

        return data


    def load_tracker(self, csv_file):
        '''
        Loads the tracker data of a csv file
        '''

        df = pd.read_csv(csv_file)
        fps = int(round(df.iloc[0]['FrameRate']))
        self.invert_y = int(df.iloc[1]['Max_Pixel_y'])

        df = df[['Frame_Num','Pixel_Loc_x','Pixel_Loc_y', 'Name', 'ID','Max_Pixel_x', 'Max_Pixel_y', 'Width(px)', 'Height(px)']]
        df = self.correct_tracker_points(df)
        print(df)
        return df


    def correct_tracker_points(self, df):
        """
        ratios is the x,y|width,height ratios of the estimate to the ground truth.
        Example: Estimates are recorded at 480x720 while ground truths are recorded at 720x1280. The ratio inputed would be (720/480, 1280/720) or (1.5, 1.777778)
        

        invert_y is the height of the video. We record the data as if the origin is in the bottom left, but in other applications the origin is the top left.
        This means we subtract the height of the video to inverse this effect.
        If the height of the video is 720, invert_y=720. If the data being tested IS NOT inverted, leave it as 0 
        """
        width = int(df.iloc[0]['Width(px)'])
        height = int(df.iloc[0]['Height(px)'])

        df = df.iloc[1: , :] # Drop first row
        
        df['Pixel_Loc_x'] = df['Pixel_Loc_x'].astype(int) * (width/df['Max_Pixel_x'].astype(int))
        df['Pixel_Loc_y'] = (df['Max_Pixel_y'].astype(int) - df['Pixel_Loc_y'].astype(int)) * (height/df['Max_Pixel_y'].astype(int))
        # point = (float(point['Pixel_Loc_x'])*ratios[0],
        # (invert_y-float(point['y2']))*ratios[1],
        #         float(point['x2'])*ratios[0],
        # (invert_y-float(point['y1']))*ratios[1]
        # )
        print(df)
        return df

    def update_tracker_3D(self,index, df, mapped_min, mapped_max):
        '''
        Updates the already created plot with new data
        '''

    def show_tracker_3D(self, df, mapped_min, mapped_max):
        # https://stackoverflow.com/questions/38118598/3d-animation-using-matplotlib
        # Convert df points numpy array
        x_df = df['Pixel_Loc_x']
        y_df = df['Pixel_Loc_y']



        # N = 100
        # data = np.array(list(gen(N))).T
        # line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])



        x_list1 = []
        y_list1 = []
        z_list1 = []

        x_list2 = []
        y_list2 = []
        z_list2 = []
        for index in range(len(x_df)):
            point = (int(x_df.loc[index+1]), int(y_df.loc[index+1]))

            # Interactive mode

            if point in mapped_min.keys():
# 
                min_point = mapped_min[point]
                # for p in min_point:
                    # x,y,z = p
                x,z,y = min_point[0]
                x_list1.append(x)
                y_list1.append(y)
                z_list1.append(z)
                # scatter = ax.scatter3D(x, y,z, cmap='Greens')
                # self.front_plt = ax.scatter3D(x, y, z)

            if point in mapped_max.keys():
                max_point = mapped_max[point]

                # for p in max_point:
                x,z,y = max_point[0]
                x_list2.append(x)
                y_list2.append(y)
                z_list2.append(z)
        plt.ion
        if self.fig is None:
            self.fig = plt.figure(1)
            self.ax = plt.axes(projection='3d')


            self.back_plt = self.ax.scatter3D(x_list1, y_list1, z_list1, cmap='Greens')
            self.back_plt = self.ax.scatter3D(x_list2, y_list2, z_list2, cmap='Blues')

            # self.line_plt = ax.plot3D([front_point[0],back_point[0]], [front_point[1],back_point[1]], [0,self.room_length], color='teal')
            
            # plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
            self.ax.set_xlim(-self.room_width,self.room_width)
            self.ax.set_ylim(-self.room_height,self.room_height)
            self.ax.set_zlim(-self.room_length,self.room_length)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.show()

            # else:
                # self.ax.cla()
                # if values:
                #     self.points_plt = self.ax.scatter3D(x_list,y_list,z_list, marker="D", color="red")
                # # plt.quiver((0,0,0), (0,1,0), , color=['r','b','g'], scale=21)
                # self.ax.set_xlim(-self.room_width,self.room_width)
                # self.ax.set_ylim(-self.room_height,self.room_width)
                # self.ax.set_zlim(-self.room_length,self.room_width)
                # self.ax.set_xlabel("X")
                # self.ax.set_ylabel("Y")
                # self.ax.set_zlabel("Z")
                
                # self.float_plt = self.ax.scatter3D(floater_point[0],floater_point[1],self.vector_depth, marker="o", color="blue")
                # self.front_plt = self.ax.scatter3D(front_point[0], front_point[1], 0, marker=("o"), color="cyan")
                # self.back_plt = self.ax.scatter3D(back_point[0],back_point[1], self.room_length, marker="x", color="cyan")
                # self.line_plt = self.ax.plot3D([front_point[0],back_point[0]], [front_point[1],back_point[1]], [0,self.room_length], color='teal')
                # self.quiver = self.ax.quiver([0,0,0], [0,0,0], [0,0,0], [self.room_width,0,0], [0,self.room_height,0], [0,0,self.room_length], length=0.1, normalize=False)
                # self.fig.canvas.draw()
                # self.fig.canvas.flush_events()
            # image = cv2.circle(image, point, 2, (255,255,0), 2)
            # cv2.imshow("Tracker2D", image)
            # cv2.waitKey(1)

            # if point in mapped_min.keys():
            #     min_point = mapped_min[point]

            #     for p in min_point:
            #         x,y,z = p
            #         scatter = ax.scatter3D(x, y,z, cmap='Greens')
            #         # ax.plot_surface(x, y, z, color=('r'))

            # if point in mapped_max.keys():
            #     max_point = mapped_max[point]

            #     for p in max_point:
            #         x,y,z = p
            #         scatter = ax.scatter3D(x, y,z, cmap='Blues')

            
            
        return self.show_img


    def show_tracker_2D(self, df, image, data_columns =['Pixel_Loc_x', 'Pixel_Loc_y'], window="Tracker2D", colour = (255,255,0)):
        x = df[data_columns[0]].tolist()
        y = df[data_columns[1]].tolist()

        for index in range(len(x)):
            image = cv2.circle(image, (int(x[index]), int(y[index])), 2, colour, 2)
            cv2.imshow(window, image)
            cv2.waitKey(1)


    def extend_image_to_corners(self, image, corners):
        '''
        extends the image with black until corners are visible.

        Returns new image as well as the amount extended and on which sides. This is useful for transforming datapoints.

        offset = [top, left, bottom, right] in pixels
        '''
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        for corner in corners:
            if corner[0] < min_x:
                min_x = corner[0]
            if corner[1] < min_y:
                min_y = corner[1]
            if corner[0] > max_x:
                max_x = corner[0]
            if corner[1] > max_y:
                max_y = corner[1]

        if max_y < image.shape[0]:
            max_y = image.shape[0]
        if max_x < image.shape[1]:
            max_x = image.shape[1]

        offset = (abs(int(min_y)), abs(int(min_x)), abs(int(max_y - image.shape[0])), abs(int(max_x - image.shape[1])))
        image = cv2.copyMakeBorder(image, top=offset[0], left=offset[1], bottom=offset[2], right=offset[3], borderType=0)
        

        for corner_1 in corners:
            p1 = (int(corner_1[0] + offset[1]), int(corner_1[1] + offset[0]))
            image = image = cv2.circle(image, p1, 2, (255,255,0), 2)
            for corner_2 in corners:
                p1 = (int(corner_1[0] + offset[1]), int(corner_1[1] + offset[0]))
                p2 = (int(corner_2[0] + offset[1]), int(corner_2[1] + offset[0]))
                image = cv2.line(image, p1, p2, (255,0,0))
        
        #Resets corners for future assignment
        self.clear_corners()
        return image, offset

    def get_room_dimensions(self, room_points):
        '''
        Given room points returns the width and length of the room given the farthest points.
        '''

        max_width = 0
        max_length = 0
        
        for point in room_points:
            x, z, y = point
            if x > max_width:
                max_width = x
            if z > max_length:
                max_length = z

        return max_width, max_length

    def evaluate_calibration(self):
        pass

    def visualize_distortion(self, camera_matrix, distiortion, image=None):
        '''
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20projectPoints(InputArray%20objectPoints,%20InputArray%20rvec,%20InputArray%20tvec,%20InputArray%20cameraMatrix,%20InputArray%20distCoeffs,%20OutputArray%20imagePoints,%20OutputArray%20jacobian,%20double%20aspectRatio)

        opencv 3 here: 
        https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        [x] =  [X]
        [y] = R[Y] + t
        [z] =  [Z]

        x' = x/z
        y' = y/z

        NOTE: k1-6 are radial distortion coefficients, p1 and p2 are tangential distortion
                if any 

        x'' = x' (1 + k1 * r^2+k2 + k3*r^4 + k3*r^6)/ + 2*p1*x'*y' + p2(r^2 + (2x')^2)
                 (1 + k4 * r^2+k5 + k3*r^4 + k6*r^6)

        y'' = y' (1 + k1 * r^2+k2 + k3*r^4 + k3*r^6)/ + p1(r^2 + (2y')^2) + 2*p2 * x' * y'
                 (1 + k4 * r^2+k5 + k3*r^4 + k6*r^6)

        r^2 = x'^2 + y'^2
        u = fx * x'' + cx
        v = fy * y'' + cy
        '''
        if image is None:
            image = self.image
        
        height, width, _ = image.shape
        nstep = 20
        d = list(distiortion[0])
        # print(d)
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy = list(distiortion[0])
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        cx = camera_matrix[0,2]
        cy = camera_matrix[1,2]

        # Create a grid from the size of the image
        uv = np.meshgrid(np.linspace(-width,width-1,nstep), np.linspace(-height,height-1,nstep))
        # print(uv)
        # NOTE: p stands for prime (')
        u = uv[0]
        v = uv[1]
        z = np.ones(u.shape)
        xp = u / z # z=1
        yp = v / z # z=1

        # print(xp, yp)

        r2 = xp ** 2 + yp ** 2
        r4 = r2 ** 2 
        r6 = r2 ** 3
        coef = (1 + k1*r2 + k2*r4 + k3*r6)/(1 + k4*r2 + k5*r4 + k6*r6)
        
        # print(coef)

        xpp = xp * coef + 2*p1*(xp*yp) + p2*(r2 + 2*xp**2) + s1*r2 + s2*r4
        ypp = yp * coef + p1*(r2 + 2*yp**2) + 2 * p2 * (xp * yp) + s1*r2 + s2*r4

        x_radial_distortion = xp * (1 + k1*r2 + k2*r4 + k3*r6) # xp
        y_radial_distortion = yp * (1 + k1*r2 + k2*r4 + k3*r6) # yp

        x_tangential_distortion = xp + (2*p1*xp*yp +p2*(r2 + 2*xp**2)) # xpp
        y_tangential_distortion = yp + (p1*(r2 + 2*yp**2) + 2 * p2 * xp * yp) # ypp


        u2 = fx * xpp + cx
        v2 = fy * ypp + cy

        difference_u = u2[:] - u[:]
        difference_v = v2[:] - v[:]

        # du, dv = np.meshgrid(difference_u, difference_v)
        # dr = np.reshape(np.hypot(difference_u,difference_v), xp.shape)
        dr = np.reshape(np.hypot(difference_u,difference_v), u.shape)
        # print(dr)
        center = (width/2, height/2)
        principal_point = (cx, cy)

        fix, ax = plt.subplots()
        ax.scatter(center[0], center[1], marker="o")
        ax.scatter(principal_point[0], principal_point[1], marker="X")
        CS = ax.contour(u[1,:]+1, v[:,1]+1, dr)
        # ax.quiver(u[:]+1, v[:]+1, du, dv)
        ax.clabel(CS, fontsize=9, inline=True)
        plt.show()
        # dr = reshape(hypot(du,dv), size(u));

    # coef = (1 + D(1)*r2 + D(2)*r4 + D(5)*r6) ./ (1 + D(6)*r2 + D(7)*r4 + D(8)*r6);
        # % plot
        # quiver(u(:)+1, v(:)+1, du, dv)
        # hold on
        # plot(opts.imageSize(1)/2, opts.imageSize(2)/2, 'x', M(1,3), M(2,3), 'o')
        # [C, hC] = contour(u(1,:)+1, v(:,1)+1, dr, 'k');
        # clabel(C, hC)

        # plot(opts.imageSize(1)/2, opts.imageSize(2)/2, 'x', M(1,3), M(2,3), 'o')

    def save_room(self, filename):
        '''
        Saves all values which define a room.
        '''
        
        if filename :

            with open(filename, 'wb') as f:
                pickle.dump(self.room_points, f)
                pickle.dump(self.corners, f)

                # Room Dimensions
                pickle.dump(self.room_width, f)
                pickle.dump(self.room_length, f)
                pickle.dump(self.room_height, f)
                
                # Mapped coordinates from 2D to 3D. Useful for loading up tracker and getting 3D points
                pickle.dump(self.mapped, f)

                # Save calibration.
                pickle.dump(self.camera_matrix, f)
                pickle.dump(self.distioriton_matrix, f)
                pickle.dump(self.rotation_vector, f)
                pickle.dump(self.translation_vector, f)

                pickle.dump(self.calibration_resolution, f)
                f.close()

            # Saves camera calibration
            # self.save_calibration(filename)

    def load_room(self, filename):
        if filename :

            with open(filename, 'wb') as f:
                self.room_points = pickle.load(f)
                self.corners = pickle.load(f)
                self.room_width = pickle.load(f)
                self.room_length = pickle.load(f)
                self.room_height = pickle.load(f)
                self.mapped = pickle.load(f)
                self.camera_matrix = pickle.load(f)
                self.distioriton_matrix = pickle.load(f)
                self.rotation_vector = pickle.load(f)
                self.translation_vector = pickle.load(f)
                self.calibration_resolution = pickle.load(f)


                f.close()

    def load_and_undistort_calibration(self, video_path, visualize_distortion=False, points_data=None):
        calibration_path = video_path[:-3] + "pickle"
        checkerboard_image = self.collect_frames(video_path,1,1,2)[0]

        self.calibration_resolution = (checkerboard_image.shape[1], checkerboard_image.shape[0])

        c, d, r, t, error, resolution = self.load_calibration(calibration_path)
        checkerboard = self.undistort_room(image=checkerboard_image, points_data=points_data)
        visroom = self.undistort_room()
        cv2.imshow(video_path, visroom)
        cv2.imshow("checkerboard", checkerboard)

        if visualize_distortion:
            self.visualize_distortion(c, d)

        cv2.waitKey(0)
        return visroom, checkerboard, c, d ,r ,t, error, resolution

    # def warp_point(self, M, x: int, y: int):
    #     # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    #     # M - 3×3 transformation matrix (inverted).

    #     return (
    #         int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d), # x
    #         int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d), # y
    #     )


    def stitch_rooms(self, video_list, undistort=True, key_index=None, show=False):
        '''
        Stitches rooms with custom keypoints with SIFT descriptors

        undistort uses current room undistortion parameters to undistort all the videos in video_list

        key_index is the index for which video is the source that all others will use as alignment reference.

        returns a stitched image that is overlapped by all videos and a list of homography transformation matrices for every video index
        '''

        # Undistort all the rooms
        room_images = []
        room_tracks = []
        for video in video_list:
            print(video)
            frame = self.collect_frames(video,1,1,2)[0]
            if undistort:
                frame = self.undistort_room(frame)
            room_images.append(frame)
        
        #
        image_dict = {}
        orb = cv2.ORB_create()
        sift = cv2.SIFT_create()

        # Self defined keypoints with corner detection. 
        for index, img in enumerate(room_images):
            maxCorners = max(0, 200)
            # Parameters for Shi-Tomasi algorithm
            qualityLevel = 0.01
            minDistance = 10
            blockSize = 3
            gradientSize = 3
            useHarrisDetector = False
            k = 0.03
            # Copy the source image
            copy = np.copy(img)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Apply corner detection
            corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, None, blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
            # Draw corners detected
            print('** Number of corners detected:', corners.shape[0])
            radius = 4
            for i in range(corners.shape[0]):
                cv2.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), radius, (0, 255, 0), cv2.FILLED)
            # Show what you got
            if show:
                cv2.namedWindow("source_window")
                cv2.imshow("source_window", copy)
                cv2.waitKey(600)
            # Set the needed parameters to find the refined corners
            winSize = (5, 5)
            zeroZone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
            # Calculate the refined corner locations
            corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

            # Convert subpixel corners to keypoints and create SIFT descriptors
            kp_corners = cv2.KeyPoint_convert(corners)
            kp_corners, kp_des = sift.compute(img, kp_corners)
            img=cv2.drawKeypoints(img,kp_corners,img)

            # Store for later use
            image_dict[index] = (kp_corners, kp_des, img, corners)



        matcher = cv2.BFMatcher()
        # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
        # matcher = flann.knnMatch(des1,des2,k=2)
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 8)
        # search_params = dict(checks = 100)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        print("MATCHING")
        if key_index is None:
            source_images = image_dict.keys()
        else:
            source_images = [key_index]

        accumulate = image_dict[source_images[0]][2]
        homographies = []
        for img_key_1 in source_images:
            accumulate = image_dict[img_key_1][2]
            for img_key_2 in image_dict.keys():

                if img_key_1 == img_key_2:
                    continue

                print(image_dict[img_key_1][3].shape)
                d1 = image_dict[img_key_1][1]
                d2 = image_dict[img_key_2][1]

                kp1 = image_dict[img_key_1][0]
                kp2 = image_dict[img_key_2][0]
                img = image_dict[img_key_2][2]
                matches = matcher.knnMatch(d1,d2, k=2)
                # matches = flann.knnMatch(d1,d2, k=2)
                for m in matches:
                    print(m[0].distance)

                good = []
                for m,n in matches:
                    if m.distance < 0.4*n.distance:
                        good.append(m)
                

                dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                
                
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x[0].distance)

                # print(matches)
                # img3 = cv2.drawMatchesKnn(image_dict[img_key_1][2], image_dict[img_key_1][0],
                #                         image_dict[img_key_2][2], image_dict[img_key_2][0],
                #                         matches[:10],
                #                         None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # cv2.imshow("homography", img3)
                


                # RHO = [Bazargani15]. (weighted RANSAC modification, faster in the case of many outliers)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 10.0)
                homographies.append((M,mask))
                h,w, _ = img.shape
                result = cv2.warpPerspective(image_dict[img_key_2][2], M,(w, h))
                print(h)

                accumulate = cv2.addWeighted(accumulate,0.8,result,0.4,0)

        # cv2.imshow("warped", accumulate)
        # cv2.waitKey(0)
        return accumulate, homographies

    def stitch_trackers(self, video_list, homography_list, use_csv=True, offsets=[0,0,0,0]):
        '''
        returns a list of trackers
        # offset = [top, left, bottom, right] in pixels

        We correct offsets if top and left edges are extended
        '''

        dataframe_list = []
        for index, video in enumerate(video_list):
            data_path = str(video[:-3] + 'csv')
            image = self.collect_frames(video,1,1,2)[0]
            # Load data
            try:
                data = self.load_tracker(data_path)
            except Exception as e:
                print(e)
                print("Skipping Data")
                continue

            offset_top = offsets[0]
            offset_left = offsets[1]
            
            # Undistort data
            data = self.undistort_tracker_data(data, image)
            x = data['undistorted_x'].to_numpy() + offset_left
            y = data['undistorted_y'].to_numpy() + offset_top
            z = [1] * len(x)
            points = np.vstack((x, y)).T
            points = points.reshape((-1, 1, 2))
            # points = points.reshape((1,3, len(x)))

            # No transformation if we have the same image
            if index == 0:
                print("NO DISTORTION")
                Matrix = np.identity(3)
                Maks = None
            else:
                Matrix, Mask = homography_list[index-1]
            # a_transformed = np.dot(Matrix, points)
            # print(a_transformed)
            # Note homography_lost[index] = (TransformationMatrix, Mask)
            # Project points with stitched homographies
            projected_points = cv2.perspectiveTransform(points, Matrix).T


            new_x = projected_points[0][0]
            new_y = projected_points[1][0]

            data['stitched_x'] = new_x
            data['stitched_y'] = new_y

            print(data)

            dataframe_list.append(data)
            

        return dataframe_list


    def draw_grid(self, grid_shape=(20,20), color=(255, 255, 255), thickness=2):
        h = 1080
        w = 1080
        # h, w, _ = img.shape
        img = np.zeros((h, w, 3), dtype = "uint8")
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols


        # draw vertical lines
        for index, x in enumerate(np.linspace(start=dx, stop=w-dx, num=cols-1)):
            if index == (cols-1)/2:
                x = int(round(x))
                img = cv2.line(img, (x, 0), (x, h), color=(255,0,255), thickness=7)
            else:
                x = int(round(x))
                img = cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for index, y in enumerate(np.linspace(start=dy, stop=h-dy, num=rows-1)):
            if index == (rows-1)/2:
                y = int(round(y))
                img = cv2.line(img, (0, y), (w, y), color=(255,0,255), thickness=7)
            else:
                y = int(round(y))
                img = cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        return img


    def get_files_from_folder(self, folder, extension="MP4"):
        
        files = []
        for file in os.listdir(folder):
            # check only text files
            if file.endswith(('.' + extension)):
                files.append((folder + file))
        return files

if __name__ == "__main__":
    room = room_estimation()
    calibration_list = ["K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5338.MP4",
                        "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5339.MP4",
                        "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5340.MP4",
                        "K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5341.MP4",
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


    '''
    # CONTEMPORARY AGA Room 502
    Contemporary_videos = room.get_files_from_folder("K:/MonkeTracker/Gallery Videos/Contemporary/")
    aga_502 = [(0,0,0),(0, 16465.55, 0),(13589, 16465.55,0), (13589,0,0)]
    # room_points = [(0,0,0),(0, 17043.4, 0),(13970, 17043.4,0), (13970,0,0)]
    # room_points = [(0,0,0),(0, 17043.4, 0),(17043.4,17043.4,0), (17043.4,0,0)]
    calibration = calibration_list[3]
    room.define_room(width=13589, length=16465.55, height=1000, stitch_videos=Contemporary_videos, room_points=aga_502, calibration=calibration, extend=3)
    '''
    
    
    # 
    # room = room_estimation(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4",1,1,2)[0])

    # room = room_estimation(room.collect_frames("K:/MonkeTracker/Gallery Videos/Contemporary/GP014190.MP4",1,1,2)[0])
    # room.load_and_undistort_calibration("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5341.MP4")
    # # room = room_estimation(room.collect_frames("F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GOPR4190.MP4",1,1,2)[0])
    # room.image = room.undistort_room()
    # room.show_img = room.image
    # room.define_perspective_grid()
    # room.define_perspective_grid()

    # room.superimpose_checker()
    # room.define_perspective_grid()
    # room.define_perspective_grid()
    

    

    image = room.collect_frames("K:/MonkeTracker/Gallery Videos/Contemporary/GP014190.mp4",1,1,2)[0]
    # cv2.imshow(image)
    test_image = room.undistort_room(image)
    room.show_image = test_image
    # cv2.imshow()


    data = room.load_tracker("K:/MonkeTracker/Gallery Videos/Contemporary/GP014190.csv")
    data_undist = room.undistort_tracker_data(data, image)
    print(data_undist)
    room.show_tracker_2D(data, image, window="original")
    room.show_tracker_2D(data_undist, test_image, data_columns=['undistorted_x', 'undistorted_y'], window="undistorted")
    cv2.waitKey(0)
    # exit()
    
    # # Stitch multiple videos together

    # stitched, h_list = room.stitch_rooms(videos, key_index=0)
    # room.set_image(stitched)
    # room.show_img = stitched
    # dataframe_list = room.stitch_trackers(videos, h_list)

    # room.show_tracker_2D(dataframe_list[3], image, data_columns=['Pixel_Loc_x', 'Pixel_Loc_y'], window="original_image")
    # room.show_tracker_2D(dataframe_list[3], test_image , data_columns=['undistorted_x', 'undistorted_y'], window="undistort")
    # for df in dataframe_list:
    #     # room.show_tracker_2D(df, room.collect_frames("F:/MONKE_Ground_Truth/Gallery Videos/ChrisCran/GOPR3814.MP4",1,1,2)[0], data_columns=['stitched_x', 'stitched_y'])
    #     room.show_tracker_2D(df, stitched, data_columns=['stitched_x', 'stitched_y'])

    for i in range(3):
        print(" Draw 2 lines to find where the walls meet...")
        corners = room.find_wall_intersection(degree=1)
        room.show_img, offsets = room.extend_image_to_corners(room.show_img, corners)
    room.clear_corners()

    # print("Draw 2 parallel lines to define perspective")
    # room.define_perspective_grid()
    # room.define_perspective_grid()

    room.show_img = room.superimpose_checker()
    room.image = room.show_img

    aga_502 = [(0,0,0),(0, 16465.55, 0),(13589, 16465.55,0), (13589,0,0)]
    
    # points=[(0, 3741.67, 0), (0,11225.21,0), (2355.85, 11225.21,0), (7213.6, 11225.21,0), (7213.6,0,0), (4876.8, 0,0)]
    print("Define room, draw corners that match", aga_502)
    # room.define_room(height=1000, room_points=aga_502, refine_corners=False)

    room.show_img, cube, target_cube = room.draw_cuboid()


    limits, _ = room.find_3d_limits()
    mapped_max = room.get_room_3d(limits, height=1200, step=5)
    room.save_room_3d("K:/MonkeTracker/Gallery Videos/Contemporary/", "GP014190_1200.pickle", mapped_max)
    mapped_min = room.get_room_3d(limits, height=1000, step=5)
    room.save_room_3d("K:/MonkeTracker/Gallery Videos/Contemporary/", "GP014190_1000.pickle", mapped_min)

    for data in dataframe_list:
        room.show_tracker_3D(data, mapped_min, mapped_max)

    room.save_room("K:/MonkeTracker/Gallery Videos/Contemporary/GP014190.pickle")


    # # # corners = room.find_wall_intersection(degree=2)
    # # room.clear_corners()

    # # # c, d, r, t = room.load_calibration("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.pickle")

    # # # room.define_room(10364, 10414, refine_corners=False)

    # points=[(0, 3741.67, 0), (0,11225.21,0), (2355.85, 11225.21,0), (7213.6, 11225.21,0), (7213.6,0,0), (4876.8, 0,0)]
    # room.define_room(height=1000, room_points=points, refine_corners=False)


    
    # # # room.draw_axis()
    # cv2.imshow("Room Estimation",room.show_img)
    # cv2.waitKey(0)

    # # '''
    # # room.show_img = room.img_copy()
    # image, cube, target_cube = room.draw_cuboid()

    # cv2.imshow("Cube", image)

    # limits, _ = room.find_3d_limits()
    # mapped_max = room.get_room_3d(limits, height=1200, step=5)
    # mapped_min = room.get_room_3d(limits, height=1000, step=5)

    # room.show_tracker_3D(data, mapped_min, mapped_max)
    # cv2.setMouseCallback("Room Estimation", room.draw_vector)
    


    # This crops the image for 3D display
    # left_wall = room.get_cube_side(target_cube, 1)
    # floor = room.get_cube_side(target_cube, 2)
    # right_wall = room.get_cube_side(target_cube, 3)
    # back_wall = room.get_cube_side(target_cube, 4)

    # left_wall_img = room.crop_plain(left_wall, 1000, 1000)
    # cv2.imshow("Left Wall", left_wall_img)
    # cv2.waitKey(0)
    # floor_img = room.crop_plain(floor, 1000, 1000)
    # cv2.imshow("Floor", floor_img)
    # cv2.waitKey(0)
    # right_wall_img = room.crop_plain(right_wall, 1000, 1000)
    # cv2.imshow("Right Wall", right_wall_img)
    # cv2.waitKey(0)
    # back_wall_img = room.crop_plain(back_wall, 1000, 1000)
    # cv2.imshow("Back Wall", back_wall_img)
    # cv2.waitKey(0)
    
    # limits, _ = room.find_3d_limits()
    # room.get_room_3d(limits, height=1200, step=5)
    # limits, _ = room.find_3d_limits()
    # print("Limits", limits)
    # room.get_room_3d(limits, height=1000, step=10)
    # '''





    '''
    checkerboard calibration
    '''
    # room = room_estimation()





    #Silver3Wide
    
    # # "K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4"
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5338.MP4", skip=5), checkerboard_grid=(7,9), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5338.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5339.MP4", skip=5), checkerboard_grid=(7,9), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5339.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5340.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5340.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5341.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5341.pickle")

    # # Silver4Wide
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide/GOPR5982.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide/GOPR5982.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide/GOPR5983.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide/GOPR5983.pickle")
    # # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4", skip=5), checkerboard_grid=None, out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.pickle")
    # # room.fisheye_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4"), checkerboard_grid=None, out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.MP4", skip=5), checkerboard_grid=None, out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Wide-Justin/GOPR0076.pickle")
    # # #Silver4Med
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver4Med/GOPR5985.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver4Med/GOPR5985.pickle")

    # # #Silver3Med
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5342.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5342.pickle")
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5343.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Med/GOPR5343.pickle")

    #1080p
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide1080p/GOPR5987.MP4",skip=5), out="K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide1080p/GOPR5987.pickle")

    # # #Black3Wide
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Black3Wide/GOPR5982.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Black3Wide/GOPR5982.pickle")

    # # # Black3Med
    # room.checkerboard_calibrate(room.collect_frames("K:/Github/PeopleTracker/GorpoCalibration/Black3Med/GOPR5983.MP4", skip=5), checkerboard_grid=(8,5), out="K:/Github/PeopleTracker/GorpoCalibration/Black3Med/GOPR5983.pickle")
    

    
    # room = room_estimation(cv2.imread("C:/Users/legom/Pictures/Screenshot 2022-05-06 084849.png"))
    # room = room_estimation(cv2.imread("L:/.shortcut-targets-by-id/1MP4p63J_OlME1O2ysxy_aSATSfgtn850/Gallery Videos/JohnScott COMPLETE/3-Nov20-GP030008/img_001.jpg"))
    
    # c, d, r, t, error = room.load_calibration("K:/Github/PeopleTracker/GorpoCalibration/Silver3Wide/GOPR5338.pickle")
    # room.visualize_distortion(c, d)
    # img = room.undistort_room(show=True)

    # '''
    # NOTE: this is where we draw lines like a ruler
    '''
    corners = room.find_wall_intersection(degree=2)
    room.show_img, offsets = room.extend_image_to_corners(room.show_img, corners)
    '''
    # room.show_img = room.img_copy()

    # for corner_1 in corners:
    #     for corner_2 in corners:
    #         p1 = (int(corner_1[0] + offsets[1]), int(corner_1[1] + offsets[0]))
    #         p2 = (int(corner_2[0] + offsets[1]), int(corner_2[1] + offsets[0]))
    #         cv2.line(room.show_img, p1, p2, (255,0,0))
    # room.show_img = cv2.resize(room.show_img, (1281,722))
    # room.image = deepcopy(room.show_img)
    # cv2.imshow("TEST",room.show_img)
    # cv2.waitKey(0)
    # room.clear_corners()
    # '''

    # offsets = [0,0,0,0]
    room.corners  = room.get_room_corners(4)
    room.define_room(10364, 10414)
    # room.estimate_plane(room.show_img,room.corners, room.room_points, room.camera_matrix, room.distioriton_matrix)
    room.draw_axis()
    # cv2.imshow("Room Estimation",room.show_img)
    # cv2.waitKey(0)

    room.show_img = room.img_copy()
    room.draw_cuboid()
    # limits, _ = room.find_3d_limits()
    # room.get_room_3d(limits, height=1200, step=5)
    limits, _ = room.find_3d_limits()
    # print("Limits", limits)
    room.get_room_3d(limits, height=1000, step=10)
    
    # cv2.waitKey(0)

    # room.show_img = room.img_copy()
    room.image = deepcopy(room.show_img)
    cv2.imshow("Points", room.show_img)
    cv2.setMouseCallback('Points', room.get_depth)
    # while True:
    #     cv2.imshow("Points", room.show_img)
    #     cv2.setMouseCallback('Points', room.get_depth)
    #     cv2.waitKey(1)
    # '''

    '''
    
    room.room_height = 1753 # My height
    # room_width = 2260.6 # Width
    # room_length = 1028.7 # Depth
    room_width = 2260.6 # Width
    room_length = 1620# Depth from bottom left stud

    target_height = height=1016

    

    room.define_room(room_width, room_length)


    dist_from_back = room_length - 457.2 # 18 inches from back wall
    # dist_from_back = room_length # 18 inches from back wall
    dist_from_left = 457.2

    # Ground Truth Location point
    pixel_2d = (269,105)
    point = room.get_3d_point((269,105), height=1016) # 40 inches off the ground

    point = (point[0][0], point[1][1], point[2][2])
    point = (point[0], point[1], point[2])
    point = (point[0], point[1], point[2])

    mapped = room.get_3d_to_2d(point)

    real = room.get_3d_to_2d((dist_from_left, dist_from_back, 1016))

    # xy = room.get_3d_to_2d((dist_from_left, 1016, 0))
    # xz = room.get_3d_to_2d((dist_from_left, 0 , dist_from_back))

    # xy = room.get_3d_to_2d((dist_from_left, 0, 0))
    xz = room.get_3d_to_2d((dist_from_left, dist_from_back , 0))

    # room.show_img = room.draw_point(room.show_img, xy,real)
    room.show_img = room.draw_point(room.show_img, xz,real)
    room.show_img = room.draw_point(room.show_img, xz,real)
    cv2.imshow(room.window_name, room.show_img)
    cv2.waitKey(0)
    print(point)
    print(mapped)

    limits, _ = room.find_3d_limits()
    room.get_room_3d(limits, height=1016, step=2)

    # points = np.mgrid[0:1000:10, 0:1000:10, 0:1000:10].reshape(3,-1).T
    print("Projecting")
    
    cv2.imshow("Points", room.show_img)
    cv2.setMouseCallback('Points', room.get_depth)
    cv2.waitKey(0)
    print("Projected done.")
    # points2 = np.mgrid[0:1000:10, 0:1000:10, 500:501:1].reshape(3,-1).T # all 40 inches

    # img = room.draw_axis()
    # room.get_room_points(1,2)
    # room.show_room()
    # room.draw_point()
    
    
    # room.draw_box()

    # while True:
    #     key = cv2.waitKey(50) 
    #     if key == ord('w'):
    #         height += 10
    #     elif key == ord('s'):
    #         height -= 10
    #     elif key == ord('a'):
    #         width -= 10
    #     elif key == ord('d'):
    #         width += 10
    #     elif key == ord('x'):
    #         depth += 10
    #     elif key == ord('z'):
    #         depth -= 10
        

    #     axis = np.float32([[0,0,0], [0,3000,0], [3000,3000,0], [3000,0,0],
    #                 [0,0,3000],[0,3000,3000],[3000,3000,3000],[3000,0,3000] ])

    #     imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, calib.GOPRO_HERO3_BLACK_NARROW_MTX, dist)
    #     init = deepcopy(image)

    #     # mapped = get_3d_point((500,500), calib.GOPRO_HERO3_BLACK_NARROW_MTX, rvecs, tvecs)
    #     floater_point = np.float32([[width, depth, height]])
    #     floater, jac = cv2.projectPoints(floater_point, rvecs, tvecs, calib.GOPRO_HERO3_BLACK_NARROW_MTX, dist)
        
    #     X = np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*width],[-1*height],[-1]])))).dot((-depth*Lcam[:,2]-Lcam[:,3]))


    #     back_point = np.float32([[width,height,0]])

    #     cv2.circle(img,front_point,5,color=(255,255,255))
    #     cv2.circle(img,back_point,10,color=(0,255,255))
        
    #     cv2.circle(img, (100,100),5,color=(255,255,255))
    #     img = draw_point(init, front_point, back_point)
    #     cv2.circle(img,floater_point,4,color=(0,0,255), thickness=4)

    #     test = (X)
    #     # cv2.putText()
    #     # cv2.imshow('img', img) 
    #     cv2.waitKey(0) 


    '''