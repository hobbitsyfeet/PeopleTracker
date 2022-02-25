import cv2
from matplotlib.pyplot import show
import numpy as np
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img


def calculate_optical_flow(video, show=False):
    flow_dict = {}
    previous_frame = None
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if previous_frame is None:
            previous_frame = frame
            continue
        # print(previous_frame.shape, frame.shape)
        # cv2.imshow("Previous",previous_frame)
        
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray_prev = cv2.resize(gray_prev, (0,0), fx=0.25, fy=0.25) 
        gray_current = cv2.resize(gray_current, (0,0), fx=0.25, fy=0.25)

        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current, flow=None, pyr_scale=0.5, levels=6, winsize=50, iterations=3, poly_n=5, poly_sigma=1.2, flags=None)
        # flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_dict[cap.get(cv2.CAP_PROP_POS_FRAMES)] = np.mean(flow)

        mean = np.mean(flow)
        median = np.median(flow)

        # print(abs(mean), median)
        print(abs(mean))

        if show:
            temp = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) 
            temp = draw_flow(temp, flow)
            # Display the resulting frame
            cv2.imshow('frame', temp)
            if cv2.waitKey(1) == ord('q'):
                break

        previous_frame = frame
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return flow_dict


if __name__ == "__main__":
    # video = "F:/MONKE_Ground_Truth/Gallery Videos/ChrisCran/GOPR3841/GOPR3841.MP4"
    video = "F:/Videos/MonkeyVideos/2017-06-26 095735.mp4"
    flow_dict = calculate_optical_flow(video, show=True)
    
    mean = np.mean(np.array(flow_dict.values()))
    print(mean)
    # 


