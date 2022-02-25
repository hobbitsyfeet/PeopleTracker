import numpy as np
import cv2
from copy import deepcopy


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
xdistorted=x(1+k1r2+k2r4+k3r6)
ydistorted=y(1+k1r2+k2r4+k3r6)


xdistorted=x+[2p1xy+p2(r2+2x2)]
ydistorted=y+[p1(r2+2y2)+2p2xy]

Distortioncoefficients=(k1k2p1p2k3)

CameraMatrix = [fx, 0, cx]
               [0, fy, cy]
               [0,  0,  1]

Distortion Coefficients = (k1, k2, p1, p2, k3)
"""      
# Results from: http://argus.web.unc.edu/camera-calibration-database/ 
# MTX                                         f       w       h   cx     cy      a      k1    k2     t1
# GoPro Hero4 Silver	720p-120fps-narrow	1150	1280	720	640	    360	    1	-0.31	0.17	0	
# GoPro Hero3 Black	720p-120fps-narrow  	1101	1280	720	639.5	359.5	1	-0.359	0.279	0

# DST                                   FC   W       H          C                   D
# GoPro Hero3 Black	720p-60fps-wide	    4	1280	720	1.038962477337479	0.011039937655688	

GOPRO_HERO4_SILVER_NARROW_MTX = np.array([
                                    [1150, 0, 640],
                                    [0 ,1150, 360],
                                    [0,    0,   1],
                                ])
# GOPRO_HERO4_SILVER_NARROW_DST = np.array([-0.31,0.17, ,])


GOPRO_HERO3_BLACK_NARROW_MTX = np.array([
                       [1101, 0, 639.5],
                       [0 ,1150, 359.5],
                       [0,    0,   1],
                                ])


GOPRO_HERO3_SILVER ={"Resolution_x":720, "FOV":None}

def do_nothing(event,x,y,flags,params):
    pass

class room_estimation():
    def __init__(self, image=None, camera_matrix=GOPRO_HERO3_BLACK_NARROW_MTX):

        
        self.window_name = "Room Estimation"
        cv2.namedWindow("Room Estimation")
        self.image = image
        self.show_img = self.img_copy()
        self.corners = []
        self.camera_matrix = camera_matrix
        self.distioriton_matrix = None

        self.rotation_vector = None
        self.translation_vector = None

        # Size are in millimeters
        self.room_width = 3000  # X
        self.room_length = 3000 # Depth (Z)
        self.room_height = 3000 # Y

        self.sample_rate = 5   # estimates position every 1cm

        self.mapped_dictionary = {}

        self.vector_depth = 0

        self.axis = self.set_axis(3000,1000,3000)
        self.fig = None

        # self.define_room()
        # img = self.draw_axis()
        # cv2.waitKey(1)

        # self.mapped_dictionary = self.map_2d_to_3d((self.room_width, self.room_length, self.room_height))
        
        self.show_3D_plot = True
        # while True:
        #     cv2.imshow(self.window_name, self.show_img)
        #     cv2.waitKey(1)
        # self.corner_points = []

    def show_room(self, axis=True, show_3d_plot=True,  with_points=False):
        while True:
            
            # self.draw_vector()
            if axis:
                self.draw_axis()
            
            # if show_3d_plot:
            #     self.show_3D_plot()
            cv2.imshow(self.window_name, self.show_img)
            cv2.waitKey(1)


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

    def define_room(self, width, length):
        '''
        Takes variables and defines a room

        width and length are in mm
        '''
        self.room_width = width
        self.room_length = length

        # creates a 4 corner room in real coordinates
        room_points = self.get_room_points(self.room_width, self.room_length)
        self.corners = self.get_room_corners()
        retval, rvecs, tvecs = cv2.solvePnP(room_points, self.corners.astype(np.float64), self.camera_matrix, self.distioriton_matrix)
        self.rotation_vector = rvecs
        self.translation_vector = tvecs
        self.set_axis(self.room_width, self.room_height, self.room_length)

    
    def get_room_corners(self):
        print("Assigning corners...")
        #Initializes feedback to a function
        cv2.setMouseCallback("Room Estimation", self.get_corners)
        print("Setting Callback")
        while len(self.corners) < 4:
            # display the image and wait for a keypress
            cv2.imshow("Room Estimation", self.show_img)
            key = cv2.waitKey(1) & 0xFF
            self.connect_corners()
            if key == ord("c"):
                break
        self.connect_corners()
        corner_np = np.zeros((4,2,1))
        self.corners = np.reshape(np.asfarray(self.corners), (4,2,1))
        # cv2.setMouseCallback("Room Estimation", self.draw_vector)
        return self.corners
        

    def connect_corners(self):
        "Helper functio nthat visualizes corners"
        for index, point in enumerate(self.corners):
            if index == 4:
                print("Last point")
                cv2.line(self.show_img, self.corners[-1], self.corners[0], (255,0,0))
            if len(self.corners) > 1 and index+1 < len(self.corners):
                cv2.line(self.show_img, self.corners[index], self.corners[index+1], (255,0,0))

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
        point = np.float32([[point[0], point[1], point[2]]])
        mapped_point, jac = cv2.projectPoints(point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
        mapped_point = (int(mapped_point[0][0][0]),int(mapped_point[0][0][1]))

        if show is True:
            self.draw_axis()
            cv2.circle(self.show_img, mapped_point, 6, (0, 255, 255))
            cv2.imshow(self.window_name, self.show_img)
            cv2.waitKey(0)

        return mapped_point


    def map_2d_to_3d(self, room_dimensions=(1000,1000,1000), step=10):
        '''
        Maps the entire image to depth points with defined room
        '''
        print("Mapping points...")
        # Room dimensions is 1m (width) x 1m (height) x 1m(depth)
        w = room_dimensions[0]
        h = room_dimensions[1]
        d = room_dimensions[2]

        mapped = {}
        for i in range(0,w,self.sample_rate):
            print((i/step), "%")
            for j in range(0,1000,self.sample_rate):
                for k in range(0,d,self.sample_rate):
                    
                    point = np.float32([[i, k, j]])
                    
                    mapped_point, jac = cv2.projectPoints(point, self.rotation_vector, self.translation_vector, self.camera_matrix, self.distioriton_matrix)
                    mapped_point = (int(mapped_point[0][0][0]),int(mapped_point[0][0][1]))

                    if mapped_point not in mapped.keys():
                        mapped[mapped_point] = list()
                        mapped[mapped_point].append((i,k,j))
                    else:
                        mapped[mapped_point].append((i,k,j))
                    
        # for point in mapped.keys():
        #     cv2.circle(self.show_img,point,1,color=(255,0,255))
        cv2.imshow(self.window_name, self.show_img)
        print("Finished mapping points")
        self.mapped_dictionary = mapped
        return mapped

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

    # def get_calibration(img):
    #     # Define the dimensions of checkerboard 
    #     CHECKERBOARD = (6, 9) 
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    #     objp = np.zeros((6*9,3), np.float32)
    #     objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    #     axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    #     # Arrays to store object points and image points from all the images.
    #     objpoints = [] # 3d point in real world space
    #     imgpoints = [] # 2d points in image plane.

    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # Find the chess board corners
    #     ret, corners = cv2.findChessboardCorners( 
    #                 gray, CHECKERBOARD,  
    #                 cv2.CALIB_CB_ADAPTIVE_THRESH  
    #                 + cv2.CALIB_CB_FAST_CHECK + 
    #                 cv2.CALIB_CB_NORMALIZE_IMAGE)
    #     # If found, add object points, image points (after refining them)
    #     if ret == True:
    #         objpoints.append(objp)
    #         corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #         imgpoints.append(corners)
    #     #     # Draw and display the corners
    #     #     cv2.drawChessboardCorners(img, (6,9), corners2, ret)
    #     #     cv2.imshow('img', img)
    #     #     cv2.waitKey(500)
    #     # cv2.destroyAllWindows()
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #     print("Mtx",mtx)
    #     print("dist",dist)

    #     ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    #     imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    #     img = draw(img,corners2,imgpts)
    #     cv2.imshow('img',img)

    #     print("CalibRVEC:", rvecs, "CALIBTVEC", tvecs)
    #     # print(mtx,dist,rvecs,tvecs)
    #     return ret, mtx, dist, rvecs, tvecs

    # def draw_normals(img, corners, imgpts):
    #     # corner = tuple(corners[0].ravel())
    #     # print(corners[1][1])
    #     corn = (int(corners[0][0]), int(corners[0][1]))
    #     # print(imgpts)

    #     # print(corn)
    #     # print(imgpts[0].ravel())
    #     # print(tuple(imgpts[0].ravel()))
    #     # print(tuple(imgpts[1].ravel()))
    #     # print(tuple(imgpts[2].ravel()))
    #     # img = cv2.line(img, corn, tuple(imgpts[0].ravel()), (255,0,0), 5)
    #     # img = cv2.line(img, corn, tuple(imgpts[1].ravel()), (0,255,0), 5)
    #     # img = cv2.line(img, corn, tuple(imgpts[2].ravel()), (0,0,255), 5)

    #     img = cv2.arrowedLine(img, corn, tuple(imgpts[0].ravel()), (255,0,0), 2)
    #     img = cv2.arrowedLine(img, corn, tuple(imgpts[1].ravel()), (0,255,0), 2)
    #     img = cv2.arrowedLine(img, corn, tuple(imgpts[2].ravel()), (0,0,255), 2)

    #     return img

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
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, calibration_matrix, distortion_matrix)

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

        """

        objectPoints = np.array(
            [
                [[0.0],[0.0],[float(height)]],
                [[0.0],[float(length)],[float(height)]],
                [[float(width)],[float(length)],[float(height)]],
                [[float(width)],[0.0],[float(height)]],
            ]
        )

        return objectPoints

    def project_point(self, event,x,y,flags,params):
        pass
    



if __name__ == "__main__":

    # room = room_estimation(cv2.imread("C:/Users/legom/Pictures/testing.jpg"))
    room = room_estimation(cv2.imread("C:/Users/legom/Pictures/Screenshot 2022-02-09 122217.png"))
    room.room_height = 1753 # My height
    room_width = 2260.6 # Width
    room_length = 1028.7 # Depth

    room.define_room(2260.6, 1028.7)

    point = room.get_3d_point((650,290), height=1016) # 40 inches off the ground
    point = (point[0][0], point[1][1], point[2][2])
    point = (point[0]*100, point[1]*100, point[2])
    point = (point[0]*100, point[1]*100, point[2])
    mapped = room.get_3d_to_2d(point)
    real = room.get_3d_to_2d((700, 700, 1016))

    xy = room.get_3d_to_2d((700, 700, 0))
    xz = room.get_3d_to_2d((0, 700, 1016))

    room.show_img = room.draw_point(room.show_img, xy,real)
    room.show_img = room.draw_point(room.show_img, xz,real)
    cv2.imshow(room.window_name, room.show_img)
    cv2.waitKey(0)
    print(point)
    print(mapped)
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


