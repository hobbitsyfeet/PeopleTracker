# import the necessary packages
from threading import Thread
import sys
import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.frame_number = 1
        self.next = self.frame_number + 1
        self.skip_value = 10
        self.reset = False
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def set_next_frame(self, frame_num):
        self.stream.set(1, frame_num)

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                continue
            # print(self.next, self.frame_number + 1)
            if self.reset is True:
                
                # self.next += 1
                self.stream.set(1, self.next)
                self.frame_number = self.next
                self.reset = False
                self.Q.queue.clear()

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                for i in range(self.skip_value):
                    self.stream.grab()
                (grabbed, frame) = self.stream.read()
                

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stream.set(1,0)
                    self.frame_number = 0
                    
                    # self.stopped = True
                    
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
                
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        self.frame_number += self.skip_value
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
        del self.thread


# import numpy as np
# from multiprocessing import Process
# from subprocess import Popen, PIPE
# import cv2
# import time
# import datetime

# class ffmpeg_videocapture:
#     def __init__(self, stream, width=640, height=360, scale=1, fps=15):
#         # time.strftime('%H:%M:%S', time.gmtime(x))

#         self.command = 'ffmpeg -hwaccel cuda -ss 0 -i {i} -pix_fmt bgr24 -s {w}x{h} -vcodec rawvideo -an -sn -r {fps} -f image2pipe pipe:1'# -h encoder=hevc_videotoolbox'

#         self.stream = stream
#         self.width = width
#         self.height = height
#         self.scale = scale
#         self.fps = fps

#         self.errors = []
#         self.start()

#     def start(self):
#         width = int(self.width * self.scale)
#         height = int(self.height * self.scale)
#         command = self.command.format(i=self.stream, w=width, h=height, fps=self.fps)
#         self.capture = Popen(command.split(' '), stdout=PIPE, stderr=PIPE, bufsize=10 ** 8)

#     def read(self, frame):
#         width = int(self.width * self.scale)
#         height = int(self.height * self.scale)

#         print(time.gmtime(frame))
#         # hhmmss = time.strftime('%H:%M:%S.%f'[:-3], time.gmtime(frame/1000.0))
#         hhmmss = datetime.datetime(2000,1,1,0,0,frame%60).strftime('%H:%M:%S.%f')[:-3]

#         print(hhmmss)
#         self.command = 'ffmpeg -hwaccel cuda -ss {hhmmss} -i {i} -pix_fmt bgr24 -s {w}x{h} -vcodec rawvideo -an -sn -r {fps} -f image2pipe pipe:1'# -h encoder=hevc_videotoolbox'
#         command = self.command.format(i=self.stream, w=width, h=height, fps=self.fps, hhmmss = hhmmss)
#         self.capture = Popen(command.split(' '), stdout=PIPE, stderr=PIPE, bufsize=10 ** 8)
#         print(self.capture)
#         width = int(self.width * self.scale)
#         height = int(self.height * self.scale)
#         raw_image = self.capture.stdout.read(width * height * 3)
#         # print(raw_image)
#         frame = np.fromstring(raw_image, dtype='uint8')
#         # print(frame)
#         frame = frame.reshape((height, width, 3))
#         self.capture.stdout.flush()
#         self.capture.stdout.close()

#         return frame is not None, frame

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.capture.terminate()

# def single_camera(rtsp_stream):
#     cap = ffmpeg_videocapture(rtsp_stream)

        
# if __name__ == "__main__":
#     cap = ffmpeg_videocapture("C:/Users/legom/Desktop/Lots_of_people/GP034322.MP4")
#     # bw_image = np.zeros((512,512), dtype="unit8")
#     ret, frame = cap.read(0)
#     blank_image = np.zeros((720,480,3), np.uint8)
#     frame_num = 0
#     while(True):
#         ret, frame = cap.read(frame_num)
#         cv2.imshow("frame", frame)
#         cv2.waitKey(1)
#         frame_num += 1

# import cv2
# import sys
# from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
# from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
# from PyQt5.QtGui import QImage, QPixmap

# class Thread(QThread):

#         # self.height = height
#         # self.width = width
#     changePixmap = pyqtSignal(QImage)

#     def display(self):
#         # https://stackoverflow.com/a/55468544/6622587
#         rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgbImage.shape
#         bytesPerLine = ch * w
#         convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
#         p = convertToQtFormat.scaled(720, 480, Qt.KeepAspectRatio)
#         self.changePixmap.emit(p)

#     def run(self):
#         self.pause = False
#         self.cap = cv2.VideoCapture("C:/Users/legom/Desktop/Lots_of_people/test.avi")
#         width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
#         height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
#         while not self.pause:
#             ret, self.frame = self.cap.read()
#             if ret:
#                 self.display()

#     def draw(self, x, y):
#         print("drawing")
#         cv2.circle(self.frame, (int(x+240),int(y+240)),2,(0,0,255),-1)
#         self.display()
#         self.pause = True


        


# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.title = 'PyQt5 Video'
#         self.left = 100
#         self.top = 100
#         self.width = 720
#         self.height = 360
#         self.initUI()
#         self.cap = None


#     @pyqtSlot(QImage)
#     def setImage(self, image):
#         self.label.setPixmap(QPixmap.fromImage(image))

#     def initUI(self):
#         self.setWindowTitle("Hello")
#         self.setGeometry(self.left, self.top, self.width, self.height)
#         self.width = self.frameGeometry().width()
#         self.height = self.frameGeometry().height()
#         # self.resize(1800, 1200)
#         # create a label
#         self.label = QLabel(self)
#         self.label.move(0, -38)
#         self.label.resize(720, 360)
#         self.th = Thread(self)
#         self.th.changePixmap.connect(self.setImage)
#         self.th.start()
#         self.show()
        

#     def keyPressEvent(self, e):
#         if e.key() == Qt.Key_Escape:
#             print("CLOSING")
#             self.close()

#     def mouseMoveEvent(self, e):
#         x = e.x()
#         y = e.y()
#         self.th.draw(x,y)

#         text = f'x: {x},  y: {y}'
#         # self.label.setText(text)
        
#         print(text)
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = App()
#     sys.exit(app.exec_())