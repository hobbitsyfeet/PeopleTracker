"on the input of the video"
from __future__ import print_function
import cv2 as cv2
import numpy as np
import sys
import room_estimation as re
# from random import randrange
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QPlainTextEdit, QSlider, QStyle, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QCheckBox, QMenuBar, QSpinBox, QErrorMessage, QProgressDialog, QFormLayout, QDoubleSpinBox, QSplashScreen

from PyQt5.QtGui import QIcon, QIntValidator, QPixmap, QImage, QPainter, QPen, QKeySequence
from PyQt5.QtCore import Qt, QRect, QCoreApplication, QTimer
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from numpy.lib.utils import source

class CameraPosition(QWidget):
    """
    Steps to stitching the images
    1) Find Keypoints between images (SIFT/SURF/ORB?)
     - SIFT and SURF are patented and if you want to use it in a real-world application, you need to pay a licensing fee.
     - ORB is not.
    2) Compute the distances between every descriptor
    3) Select the best matches to align (Knn - 2 best matches for each descriptor {k=2})
    - finds two best matches for each feature and leaves the best one only if the ratio between descriptor distances is greater than the threshold
    4) Estimate Homography (Homography Estimator)
    5) Warp the images (translate and rotate) to align
        - NOTE this is where we will extract the top left of each frame for reference on the stitched image
        locations of each frame (hopefully)
    6) Stitch the images (Stitch and blend)
    """
    def __init__(self, source_vid=None, reference_vid=None):
        super().__init__()
        # self.openFileNameDialog()
        self.title = 'Camera Position'
        self.left = 250
        self.top = 250
        self.width = 480
        self.height = 240
        self.stitched_img = None
        self.initUI()
        self.init_menubar()

        self.reference_imgs = []

        self.room_estimation = None

        if source_vid is not None:
            self.source_img = self.collect_frames(source_vid,0,1,1)[0]
            cv2.imshow("source", self.source_img)
            cv2.waitKey(0)
        else:
            self.source_img = None


        # if reference_vid is not None:
        #     self.add_reference(reference_vid)
        # else:
        self.reference_imgs = []

        self.stitch_list = []
        # self.stitched_img = None
        # self.source_img = None
        # if self.source_img is not None and self.reference_imgs:
        #     status, self.stitched_img = self.stitch()
        #     cv2.imshow("stitched", self.stitched_img)
        #     cv2.waitKey(0)
        # else:
        #     

        # self.show_estimation()
    def openFileNameDialog(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","MP4 (*.mp4)")
    
        return fileName



    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.show()


    def init_menubar(self):
            bar = QMenuBar(self)
            file = bar.addMenu("File")
            file.addAction("New Source")
            file.addAction("New Reference")
            file.addAction("Save Stitched Image")
            file.addAction("Stitch Images")


            edit = bar.addMenu("Edit")
            edit.addAction("Estimate Room in 3D")
            

            view = bar.addMenu("View")
            view.addAction("View Camera Estimation")

            file.triggered[QAction].connect(self.processtrigger)
            edit.triggered[QAction].connect(self.processtrigger)
            view.triggered[QAction].connect(self.processtrigger)
            self.layout.addWidget(bar)

    def set_view_roomestimation(self):
        self.room_widget = QWidget()
        self.layout.addWidget(self.room_widget)
        self.room_layout = QVBoxLayout()
        self.room_widget.setLayout(self.room_layout)

        self.bottom_left_pt_x = QSpinBox()
        self.bottom_left_pt_y = QSpinBox()
        self.bottom_left_pt_x.valueChanged.connect(lambda:self.room_estimation.set_corners(self.bottom_left_pt_x.value()))
        self.bottom_left_pt_y.valueChanged.connect(lambda:self.room_estimation.set_corners(self.bottom_left_pt_x.value()))

        self.top_left_pt_x = QSpinBox()
        self.top_left_pt_y = QSpinBox()
        self.top_left_pt_x.valueChanged.connect(lambda:self.room_estimation.set_corners(self.top_left_pt_x.value()))
        self.top_left_pt_y.valueChanged.connect(lambda:self.room_estimation.set_corners(self.top_left_pt_y.value()))

        self.top_right_pt_x = QSpinBox()
        self.top_right_pt_y = QSpinBox()
        self.top_right_pt_x.valueChanged.connect(lambda:self.room_estimation.set_corners(self.top_right_pt_x.value()))
        self.top_right_pt_y.valueChanged.connect(lambda:self.room_estimation.set_corners(self.top_right_pt_y.value()))

        self.bottom_right_pt_x = QSpinBox()
        self.bottom_right_pt_y = QSpinBox()
        self.bottom_right_pt_x.valueChanged.connect(lambda:self.room_estimation.set_corners(self.bottom_right_pt_x.value()))
        self.bottom_right_pt_y.valueChanged.connect(lambda:self.room_estimation.set_corners(self.bottom_right_pt_x.value()))

        self.room_layout.addWidget(self.bottom_left_pt_x)   
        self.room_layout.addWidget(self.bottom_left_pt_y)
        self.room_layout.addWidget(self.top_left_pt_x) 
        self.room_layout.addWidget(self.top_left_pt_y)
        self.room_layout.addWidget(self.top_right_pt_x) 
        self.room_layout.addWidget(self.top_right_pt_y)
        self.room_layout.addWidget(self.bottom_right_pt_x)
        self.room_layout.addWidget(self.bottom_right_pt_y)
        # self.room_estimation.set_corners()

    def processtrigger(self, trigger):
        text = trigger.text()

        if text == "New Source":
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","MP4 (*.mp4)")
            if fileName:
                self.set_source(fileName)
        elif text == "New Reference":
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","MP4 (*.mp4)")
            if fileName:
                self.add_reference(fileName, 10, 5, 100)
        elif text == "Stitch Images":
            success, stitched = self.stitch(self.reference_imgs)
            if success:
                print(stitched)
                cv2.imshow("Stitched image", stitched)
        elif text == "Save Stitched Image":
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","JPEG (*.jpg);;All Files (*);;Text Files (*.txt)", options=options)
            if fileName:
                cv2.imwrite(fileName, self.stitched_img)
        elif text == "Estimate Room in 3D":
            
            img = None
            # room = None
            self.room_estimation = re.room_estimation()
            if self.stitched_img is not None:
                print("Setting estimation to stitched image")
                self.room_estimation.set_image(self.stitched_img)
            elif self.source_img is not None:
                print("Setting estimation to stitched image")
                self.room_estimation.set_image(self.source_img)

            limits, _ = self.room_estimation.find_3d_limits()
            self.room_estimation.get_room_3d(limits, height=1000, step=2)
            
            self.set_view_roomestimation()

        elif text == "View Camera Estimation":
            self.show_estimation()

    def stitch(self, image_list, out_path=None):
        print("Stitching")
        # Test images
        # Initiate ORB detector
        image_dict = {}
        orb = cv2.ORB_create()
        for index, img in enumerate(image_list):
            kp, des = orb.detectAndCompute(img,None)
            image_dict[index] = (kp, des)

        bf = cv2.BFMatcher()
        for img_key_1 in image_dict.keys():
            for img_key_2 in image_dict.keys():
                matches = bf.knnMatch(image_dict[img_key_1][1],image_dict[img_key_2][1], k=10)
                for m in matches:
                    print(m[0].distance)

        # print("Stitching images")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        # img_list = [self.source_img] + self.reference_imgs
        status, self.stitched_img = stitcher.stitch(image_list)

        if status == cv2.Stitcher_OK:
            print("Stitching Successful.")

        if out_path is None and self.stitched_img is not None:
            cv2.imwrite("./stitched.jpg", self.stitched_img)

        return status, self.stitched_img

    def set_source(self, video_path, frame=0):
        self.source_img = self.collect_frames(video_path, frame,1,1)[0]
        self.room_estimation = re.room_estimation(self.source_img)
        # if self.reference_imgs and self.source_img is not None:
        #     status, self.stitched_img = self.stitch()
        cv2.imshow(video_path, self.source_img)
        cv2.waitKey(0)

    def add_reference(self, video_path, frame_start, total_frames, step):
        frames = self.collect_frames(video_path,frame_start,step,total_frames)
        self.reference_imgs = frames
        # if self.reference_imgs:
        #     status, self.stitched_img = self.stitch()
        # cv2.imshow(video_path, frames)
        # cv2.waitKey(0)

    def find_reference(self, query_img, train_img):

        #find the common features between the two images
        kp1, kp2, des1_umat, des2_umat = self.find_features(query_img, train_img)
        
        #match the features
        matches = self.match_features(des1_umat, des2_umat, mode=1)
        '''
        #draw matches
        try:
            match_img = cv2.drawMatchesKnn(query_img, kp1, train_img, kp2, matches, None, flags=2)
        except:
            print("draw matches (non-knn)")
            match_img = cv2.drawMatches(query_img, kp1, train_img, kp2, matches, None, flags=2)
        '''
        #find the homography matrix
        dst, M = self.find_homography(query_img, train_img, kp1, kp2, matches)

        #return the pixel values of the lines
        return [np.int32(dst)], M

    def find_features(self, img1, img2):
        """
        This function finds features between 2 images using ORB
        Desribes keypoints using "steer" BRIEF
        """
        #use GPU
        img1, img2 = cv2.UMat(img1), cv2.UMat(img2)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        return kp1, kp2, des1, des2

    def match_features(self, des1, des2, mode=0):
        # 0 = BRUTE FORCE
        # 1 = BRUTE FORCE HAMMING
        if mode == 0:
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matches = good

        elif mode == 1:
            # create BFMatcher object with NORM_HAMMING
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x:x.distance)
            
            matches = matches[:20]
            
        return matches

    def find_homography(self, img1, img2, kp1, kp2, matches):
        ## extract the matched keypoints
        # 
        # feature_match = cv2.drawMatches(img1,kp1,img2,kp2,matches,None)

        # plt.imshow(feature_match, 'gray'),plt.show()

        src_pts  = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts  = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        ## find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        h,w = img1.shape[:2]
        pts = np.array([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ], dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # dst = cv2.getAffineTransform(pts, M)
        return dst, M

    def show_estimation(self):
        
        src_points, matrix = self.find_reference(self.source_img, self.stitched_img)
        cover_img = cv2.polylines(self.stitched_img, src_points, True, (255, 0, 0),5, cv2.LINE_AA)

        for ref in self.reference_imgs:
            ref_points, matrix = self.find_reference(ref, self.stitched_img)
            cover_img = cv2.polylines(self.stitched_img, ref_points, True, (0, 0, 255),5, cv2.LINE_AA)
        
        
        cv2.imshow("Camera Estimation", cover_img)
        return cover_img

    def collect_frames(self, video_source, start_frame, skip, total_frames):
        """
        Collects the images for stitching
        """
        print(video_source)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        print("Collecting Frames...")

        #setup cv2 capture from video
        cap = cv2.VideoCapture(video_source)
        frames = []
        #set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # print("Starting at frame: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        while len(frames) < total_frames:
            # print(frames)
            #read the image from that skipped frame
            ret, frame = cap.read()

            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip-1)
            #set current frame to the next n-skipped frames
            # print(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if ret:
                # if cv2.waitKey(30) & 0xFF == ord('q'):
                #     break
                # cv2.imshow('frame', frame)
                #append the frames to be processed
                frames.append(frame)
        cv2.destroyAllWindows()
        return frames

    cv2.destroyAllWindows()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    ex = CameraPosition()
    ex.set_source("C:/Users/legom/Videos/GOPR0072.MP4", 200)
    # ex.room_estimation.define_room(2260, 2872)
    # ex.room_estimation.estimate()




    # ex.add_reference("F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GP014189/GP014189.MP4")
    # ex.add_reference("C:/Users/legom/Videos/GOPR0072.MP4", 150, 5, 10)
    # print(ex.source_img)
    # print(ex.reference_imgs)

    # img_list = [ex.source_img] + ex.reference_imgs
    # print(img_list)
    # success, img = ex.stitch(img_list)
    # print(success)

    # cv2.imshow("Stitched", img)
    # cv2.waitKey(0)
    # ex.show_estimation()

    sys.exit(app.exec_())

    # cv2.imwrite("./output/cover_img2.jpg", cover_img)
    # cv2.imwrite("./output/query_img.jpg", frames[4])
