'''
Model from: https://github.com/Superlee506/Mask_RCNN_Humanpose
'''

import cv2
import pandas as pd
from pixellib.instance import custom_segmentation
from PyQt5 import QtCore
import numpy as np



import timeit
import time

def load_predicted(pred_file):
    """
    Loads predicted data into a variable for later use.
    Pred_dict[frame_number] = Lost of boxes

    A box is described by:
    x = box[0],
    y = box[1],
    width = box[2],
    height = box[3]
    """
    preds = pd.read_csv(pred_file)
    # print(preds)

    frames = preds.iloc[:,0].tolist()
    rois = preds.iloc[:,1].tolist()

    pred_dict = {}
    ## Convert ROIs to proper lists
    for i, roi in enumerate(rois):
        roi_list = roi.split("\n")
        box_list = []
        box_areas = []

        #Parse the string saved in csv into a list
        for box in roi_list:
            box = box.replace("[", "")
            box = box.replace("]", "")
            box = box.split()
            box = tuple(map(int, box))

            area = 0
            if box:
            
                y1 = box[0]
                x1 = box[1]
                y2 = box[2]
                x2 = box[3]

                box = (x1,y1,x2,y2)
                area = (box[0] - box[2]) * (box[1] - box[3])

            box_areas.append(area)
            box_list.append(box)

        pred_dict[frames[i]] = (box_list, box_areas)

    return pred_dict

def display_preds(frame, frame_num, pred_dict, ratios):
    """
    Displays the prediction
    """

    if frame_num in pred_dict.keys():
    
        boxes = pred_dict[frame_num][0]
        # print("RATIOS", ratios)
        for box in boxes:
            if box:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]

                p1 = (int(x1*ratios[0]), int(y1*ratios[1]))
                p2 = (int(x2*ratios[0]), int(y2*ratios[1]))

                # print(p1,p2)
                frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 1)
                frame = cv2.putText(frame,(str(int(x1*ratios[0]))) + "," + str(int(y1*ratios[1])) , p1, 5, 0.5, (0,0,0))

    return frame

def predict(filename, model="mask_rcnn_coco_person.h5", class_names=["BG", "person"], step=10, display=False, progress=None, logger=None):
    '''
    Uses MaskRCNN COCO models and uses them to predict items on the image.
    Steps indicate how frequently the model should predict on the video (default every 10 frames)
    Display shows predictions visually
    Progress is a QProgressDialog for the application
    Logger is a logger which both displays info and records it for crashlogger

    Exports results into filename_predict.csv
    '''
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #loads models
    custom_model = custom_segmentation()
    custom_model.inferConfig(num_classes= 1, class_names=class_names)
    custom_model.load_model(model)

    #loads videos
    cap = cv2.VideoCapture(filename)
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #init lists for temp storage
    roi_list = []
    frame_list = []
    score_list = []
    
    time_queue = []
    print(vid_length)
    #iterate through the video by said steps
    for frame_num in range(0, vid_length, step):
        start = timeit.default_timer()
        
        #set frame number to i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read() 

        if ret:

            if progress is not None:
                if frame_num == 0:
                    progress.setLabelText("Predict people location progress")
                    progress.setRange(0,vid_length)
                progress.setValue(frame_num)
                QtCore.QCoreApplication.processEvents()

            #Segment the image
            segmask, frame = custom_model.segmentFrame(frame, True)

            #Place data in lists for easy export
            roi_list.append(segmask['rois'])
            score_list.append(segmask['scores'])
            frame_list.append(frame_num)

            if display:
                cv2.imshow("Predictions", frame)
                key = cv2.waitKey(1) & 0xFF

            stop = timeit.default_timer()

            if len(time_queue) == 60:
                time_queue.pop()
            
            fps = 1/(stop - start)

            time_queue.append(fps)
            print('FPS:', fps)
            fps = sum(time_queue) / len(time_queue)

            if logger is not None:
                eta = ((vid_length-frame_num)/(step*fps))
                printstr = "Predicting frame " + str(frame_num) + "/" + str(vid_length) + " \npeople: "+ str(len(segmask['rois'])) + " \nETA: " + time.strftime('%H:%M:%S', time.gmtime(eta))
                logger(printstr)
                progress.setLabelText("Predict people location progress \n\n" + printstr)
                if frame_num + step == vid_length:
                    logger("Predicting Complete.")
    
            else:
                print("Predicting frame " + str(frame_num) + "/" + str(vid_length) + " people:"+ str(len(segmask['rois'])))
                eta = ((vid_length-frame_num)/(step*fps)) 
                print("ETA:", time.strftime('%H:%M:%S', time.gmtime(eta)))

        
    #create dataframe
    data = {
        "Frame_Num":frame_list,
        "Region_of_interest":roi_list,
        "Scores":score_list
    }

    #export dataframe
    df = pd.DataFrame(data)
    export_csv = df.to_csv((filename[:-4] + "_predict.csv"), index = None, header=True) #Don't forget to add '.csv' at the end of the path
    return frame_list, roi_list, score_list

def compute_iou(box, boxes, boxes_area, ratios=(1,1), frame=None):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, x2, y2]
    boxes: [boxes_count, (x1, y1, x2, y2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    ratio: ratio (width, height) to scale boxes from video resolution to analysis resolution

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """

    area = (box[0] - box[2]) * (box[1] - box[3]) # SOMETHING IS WRONG WITH BOX. GOOD NIGHT!

    ious = []
    for index, preds in enumerate(boxes):
        

        x1 = int(preds[0]*ratios[0])
        y1 = int(preds[1]*ratios[1])
        x2 = int(preds[2]*ratios[0])
        y2 = int(preds[3]*ratios[1])

        if frame is not None:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (150,150,0), 1)

        xA = max(box[0], x1)
        yA = max(box[1], y1)
        xB = min(box[2], x2)
        yB = min(box[3], y2)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        boxBArea = (x2 - x1 + 1) * (y2 - y1 + 1)
        # print(box_area, boxBArea)
        iou = interArea / float(area + boxBArea - interArea)
        ious.append(iou)

    return ious, frame


if __name__ == "__main__":
    # predict("./videos/(Simple) GP014125.MP4",display=True,step=100)

    box_1 = (485, 461, 714, 588)
    area_1 = (box_1[0] - box_1[2]) * (box_1[1] - box_1[3])

    box_2 = (485, 460, 717, 599)
    area_2 = (box_2[0] - box_2[2]) * (box_2[1] - box_2[3])
    iou = compute_iou(box_1, area_1, [box_2,box_2], [area_2,area_2], (1,1) )
    print(iou)
