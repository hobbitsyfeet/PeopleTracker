# -*- coding: utf-8 -*-
import maskrcnn
import PyQt5
import qt_dialog
import evaluate
import os
import pandas as pd

def video_predictions(models, videos, step=2):
    for model_index, gallery in enumerate (videos):
        for video in gallery:
            print("Loading video:", video, " and training on model: people_models[model_index]")

            # Load model and Predict video
            frame, rois, scores = maskrcnn.predict(video, model=models[model_index], step=step, display=True, progress=True, class_names=['BG', 'Person'])  
            
            predict_file = str(video[:-4] + "_predict.csv")
            pred_dict = maskrcnn.load_predicted(predict_file)

            # Track predictions and save
            export_filename = str(video[:-4] + "Predictions_Ids.csv")
            prediction_dict = maskrcnn.track_predictions(pred_dict, video, preview=True)
            prediction_dict.to_csv(export_filename)

def video_evaluation(videos, folders, maskrcnn=False):

    for gallery_index, gallery_vids in enumerate(videos):
        for vid_index, video in enumerate(gallery_vids):

            # try:
            out_filename = video[:-4]
            if maskrcnn:
                out_filename += "_predicted"
            out_filename+="_Evaluation_Results.txt"

            print(out_filename)
            if os.path.exists((out_filename+"_Evaluation_Results.txt")):
                print("already evaluated, skipping")
                continue
            print("Loading ", video)
            te = evaluate.tracker_evaluation()

            if maskrcnn:
                video_tracked_csv = video[:-4] + "Predictions_Ids.csv"
            else:
                video_tracked_csv = video[:-4] + ".csv"

            te.load_tracker_data(video_tracked_csv)
            te.load_json(folders[gallery_index][vid_index])
            frames_removed = te.validate_and_correct_ground_tuths()

            print("Calculating Errors...")
            errors, error_dict = te.calculate_errors(gt_maps_itself = not maskrcnn)
            print("Done Calculating Errors..")
            print(errors)

            # info_box = PyQt5.QtWidgets.QMessageBox(self)
            
            # info_box.setWindowTitle("Evaluation Results")
            message =   ("False Positive (FP):\t\t" + str(errors['FP']) +
                            "\nFalse Negative (FN):\t\t" + str(errors['FN']) +
                            "\nMultiple Trackers (MT):\t\t" + str(errors['MT']) +
                            "\nMultiple Objects (MO):\t\t" + str(errors['MO']) +
                            "\nConfiguration Distance (CD):\t" + str(errors['CD']) +
                            "\nFalsely Identified Tracker (FIT):\t" + str(errors['FIT']) +
                            "\nFalsely Identified Object (FIO):\t" + str(errors['FIO']) +
                            "\nTracker Purity (TP):\t\t" + str(errors['TP']) +
                            "\nObject Purity (OP):\t\t" + str(errors['OP'])
                        )

            print(message)

            f = open((out_filename+"_Evaluation_Results.txt"), "w")
            f.write(message)
            f.close()

            errors_hist = pd.DataFrame(error_dict)
            errors_hist.to_csv(out_filename+"_Evaluation_History.csv")
            # except Exception as e:
            #     print(e)
            #     print("No identification map")
        # info_box.setText(message)
        # info_box.show()
        # ret = info_box.question(self, 'Graph Option', "Would you like to show a graph?", PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.No | PyQt5.QtWidgets.QMessageBox.Cancel, PyQt5.QtWidgets.QMessageBox.Cancel)
        # if ret == PyQt5.QtWidgets.QMessageBox.Yes:
        # te.identification_graph()



if __name__ == "__main__":

    # Respective_Model_List
    # 0: John Scott
    # 1: Camera Obscura
    # 2: Chris Cran
    # 3: Contemporary
    # 4: Historical

    people_models = [
        "K:/Github/PeopleTracker/Evaluation/People/TestTrain/0/mask_rcnn_model.049-0.530556.h5", # Test John Scott
        "K:/Github/PeopleTracker/Evaluation/People/TestTrain/1/mask_rcnn_model.051-0.627259.h5", # Test Camera Obscura
        "K:/Github/PeopleTracker/Evaluation/People/TestTrain/2/mask_rcnn_model.088-1.185947.h5", # Test Chris Cran
        "K:/Github/PeopleTracker/Evaluation/People/TestTrain/3/mask_rcnn_model.069-0.236787.h5", # Test Contemporary
        "K:/Github/PeopleTracker/Evaluation/People/TestTrain/4/mask_rcnn_model.065-0.916503.h5"  # Test Historical
        ]
    
    people_gt_folders = [
        # John Scott (0)
        [
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/1-Nov17-GP020002 (done)/",
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/2-Nov20-GOPR0008 (FIXED)/",
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/3-Nov20-GP030008 (FIXED)/"
        ],
        
        # Camera Obscura (1)
        [   "K:/Github/PeopleTracker/Evaluation/People/Camera Obscura/Sept 12 (Good)/", # GOPR4386.MP4
            "K:/Github/PeopleTracker/Evaluation/People/Camera Obscura/Sept 13 (Good)/", # GP014394.MP4
            "K:/Github/PeopleTracker/Evaluation/People/Camera Obscura/Sept 28 (Good)/"  # GP044475.MP4
        ],
        
        # Chris Cran (2)
        [   "K:/Github/PeopleTracker/Evaluation/People/Chris Cran/GOPR3814 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Chris Cran/GOPR3820 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Chris Cran/GOPR3841 (IR done)/"
        ],

        # Contemporary (3)
        [   "K:/Github/PeopleTracker/Evaluation/People/Contemporary/GP014189 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Contemporary/GP014188 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Contemporary/GOPR4190 (IR done)/"
        ],

        # Historical (4)
        [   "K:/Github/PeopleTracker/Evaluation/People/Historical/GP044104 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Historical/GP054105 (IR done)/",
            "K:/Github/PeopleTracker/Evaluation/People/Historical/GP054106 (IR done)/"
        ]
    ]
    
    people_evaluate_videos = [
        # John Scott (0)
        [
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/GP020002.MP4",
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/GOPR0008.MP4",
            "K:/Github/PeopleTracker/Evaluation/People/John Scott/GP030008.MP4"
        ],

        # Camera Obscura (1)
        [
            "F:/MONKE_Ground_Truth/Gallery Videos/CameraObscura/GOPR4386.MP4", # Sept 12 (Good)
            "F:/MONKE_Ground_Truth/Gallery Videos/CameraObscura/GP014394.MP4", # Sept 13 (Good)
            "F:/MONKE_Ground_Truth/Gallery Videos/CameraObscura/GP044475.MP4" # Sept 28 (Good) # NOTE COULD NOT EVALUATE
        ],
        
        # Chris Cran (2)
        [
            "F:/MONKE_Ground_Truth/Gallery Videos/ChrisCran/GOPR3814.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/ChrisCran/GOPR3820.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/ChrisCran/GOPR3841.MP4"
        ],

        # Contemporary (3)
        [
            "F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GP014189.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GP014188.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/Contemporary/GOPR4190.MP4"
        ],
    
        # Historical (4)
        [   
            "F:/MONKE_Ground_Truth/Gallery Videos/Historical/GP044104.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/Historical/GP054105.MP4",
            "F:/MONKE_Ground_Truth/Gallery Videos/Historical/GP054106.MP4"
        ]
    ]

    justin_tracked = [
        # John Scott (0)
        ["K:/Github/PeopleTracker/Evaluation/People/Justin_People_Tracked/GOPR0008_Tracked/GOPR0008.csv"],
        # Camera Obscura (1)
        ["K:/Github/PeopleTracker/Evaluation/People/Justin_People_Tracked/GP01439_Tracked/GP014394.csv"],
        # Chris Cran (2)
        ["K:/Github/PeopleTracker/Evaluation/People/Justin_People_Tracked/GOPR3814_Tracked/GOPR3814_compiled.csv"],
        # Contemporary (3)
        ["K:/Github/PeopleTracker/Evaluation/People/Justin_People_Tracked/GP014189_Tracked/GP014189.csv"],
        # Historical (4)
        ["K:/Github/PeopleTracker/Evaluation/People/Justin_People_Tracked/GP044104_Tracked/GP044104_COMPILED.csv"]
    ]

    justin_gt = [
        # John Scott (0)
        ["K:/Github/PeopleTracker/Evaluation/People/John Scott/2-Nov20-GOPR0008 (FIXED)/"],
        # Camera Obscura (1)
        ["K:/Github/PeopleTracker/Evaluation/People/Camera Obscura/Sept 13 (Good)/"],
        # Chris Cran (2)
        ["K:/Github/PeopleTracker/Evaluation/People/Chris Cran/GOPR3814 (IR done)/"],
        # Contemporary (3)
        ["K:/Github/PeopleTracker/Evaluation/People/Contemporary/GP014189 (IR done)/"],
        # Historical (4)
        ["K:/Github/PeopleTracker/Evaluation/People/Historical/GP044104 (IR done)/"]
        ]

    
    monkey_models = [
        "K:/Github/PeopleTracker/Evaluation/Monkeys/TestTrain/0/mask_rcnn_model.019-1.093175.h5", # Test Aggression
        "K:/Github/PeopleTracker/Evaluation/Monkeys/TestTrain/1/mask_rcnn_model.016-0.848864.h5", # Test Grooming
        "K:/Github/PeopleTracker/Evaluation/Monkeys/TestTrain/2/mask_rcnn_model.065-1.512491.h5", # Test Play
        ]

    monkey_gt_folders = [
    # Displays Aggression (0)
    [
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2889/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2975/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_3029/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_3064/"
    ],
    
    # Grooming (1)
    [   "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/IMG_3910/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/IMG_4258/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/MVI_1693/"
    ],
    
    # Play (2)
    [   "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/DSC_0553/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/MVI_1077/",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/MVI_2408/"
    ],
]

monkey_videos = [
    [   "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2889.MOV",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_2975.MOV",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_3029.MOV",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Displays Aggression/MVI_3064.MOV"
    ],

    [   "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/IMG_3910.m4v",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/IMG_4258.m4v",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Grooming/MVI_1693.m4v"
    ],

    [   "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/DSC_0553.MOV",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/MVI_1077.MOV",
        "K:/Github/PeopleTracker/Evaluation/Monkeys/Play/MVI_2408.MOV"
    ]
]

video_evaluation(justin_tracked, justin_gt, maskrcnn=False)
# video_predictions(monkey_models, monkey_videos, step=2)
            # # Evaluate predictions 
            # dialog = qt_dialog()
            # self.evaluate_errors(csv, True)