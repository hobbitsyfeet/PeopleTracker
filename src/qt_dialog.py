import sys
import os
import glob

import PyQt5
import webbrowser
import crashlogger
import traceback
# from TrackerTab import person_tab
import TrackerTab
import evaluate

# from multiple_inputs import InputDialog
import multiple_inputs
import cv2
import numpy as np
import math
import pandas as pd




class App(PyQt5.QtWidgets.QWidget):
# TODO ADD A CURRENT STATE INFO TAG WHICH LETS THE USER KNOW WHAT TO DO AT A CERTAIN TIME!
    def __init__(self, start_file=None):
        self.flashSplash()

        super().__init__()
        self.filename = start_file
        self.title = 'Person'
        self.left = 250
        self.top = 250
        self.width = 480
        self.height = 240
        self.initUI(start_file)
        self.record_live = False
        self.nothing_loaded = True
        
        self.snap_to_frame_skip = True 

        self.pause_to_play = False
        self.play_to_pause = False
        self.play_state = False
        self.export_state = False
        self.region_state = False
        self.del_region_state = False
        self.export_all_state = False
        self.scrollbar_changed = False
        self.resolution_x = 720
        self.resolution_y = 480
        self.original_resolution = (None, None)
        self.vid_fps = 30
        self.snap_state = None
        self.set_tracker_state = False
        self.retain_region = False
        self.quit_State = False
        self.image = None

        self.predict_state = False
        self.load_predictions_state = False
        self.load_tracked_video_state = False
        self.was_loaded = False
        self.track_preds_state = False
        self.export_charactoristics = False
        self.export_activity = False
        self.evaluate_occlusion = False

        self.del_frame = False
        self.del_active_frame = False
        self.data_version_updated = False

        self.newest_version_path, self.newest_version = self.get_new_data_version()

        self.exported_data = False
        self.exported_activity = False
        
    
        # self.videoWindow = VideoWindow()
        # self.videoWindow.show()

    def flashSplash(self):
        self.splash = PyQt5.QtWidgets.QSplashScreen(PyQt5.QtGui.QPixmap('CursedSplash.png'))
        self.splash.show()

    def keyPressEvent(self, event):
        self.test_method()
        if int(event.modifiers()) == (PyQt5.QtCore.Qt.ControlModifier+PyQt5.QtCore.Qt.AltModifier):
            self.log("Setting Tracker")
            self.set_tracker_state = True

    def mousePressEvent(self, event):
        if event.button() == PyQt5.QtCore.Qt.MidButton:
            self.set_tracker_state = True
        elif event.button() == PyQt5.QtCore.Qt.RightButton:
            self.log("Play Toggled.")
            self.mediaStateChanged()

            
    def test_method(self):
        print('key pressed')

    def evaluate_errors(self, filename, is_mrcnn):
        # filename = str(self.filename[:-4]) + ".csv"
        # print(filename)
        te = evaluate.tracker_evaluation()

        te.load_tracker_data(filename)

        ground_truth_folder = os.path.dirname(self.filename) + "/" + os.path.basename(self.filename)[:-4] + "/"
        
        if os.path.exists(ground_truth_folder):
            te.load_json(ground_truth_folder)
        else:
            ground_truth_folder = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Ground Truth folder')
            ground_truth_folder += "/"
            te.load_json(ground_truth_folder)

        print(te.ground_truth_dict)
        if bool(te.ground_truth_dict) is False:
            self.log("Failed To load Json Files in Folder...")
            return

        frames_removed = te.validate_and_correct_ground_tuths()

        self.log("Calculating Errors...")
        errors, error_dict = te.calculate_errors()
        self.log("Done Calculating Errors..")
        print(errors)

        info_box = PyQt5.QtWidgets.QMessageBox(self)
        
        info_box.setWindowTitle("Evaluation Results")
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
        out_filename = self.filename[:-4]
        if is_mrcnn:
            out_filename += "_predicted"

        f = open((out_filename+"_Evaluation_Results.txt"), "w")
        f.write(message)
        f.close()

        errors_hist = pd.DataFrame(error_dict)
        errors_hist.to_csv(out_filename+"_Evaluation_History.csv")
        
        info_box.setText(message)
        info_box.show()
        ret = info_box.question(self, 'Graph Option', "Would you like to show a graph?", PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.No | PyQt5.QtWidgets.QMessageBox.Cancel, PyQt5.QtWidgets.QMessageBox.Cancel)
        if ret == PyQt5.QtWidgets.QMessageBox.Yes:
            te.identification_graph()

    def train_model(self):
        from train import train
        
        inputs = ["Classes", "Batch Size", "Number of Epochs", "Pretrained Model", "Network Type"]
        defaults = ["Person", "1", "300", "mask_rcnn_coco.h5", "resnet101"]
        dialog = multiple_inputs.InputDialog(labels=inputs, defaults=defaults, parent=None)
        if dialog.exec():
            inputs = dialog.getInputs()

            dataset_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Dataset to train')
            # output_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Dataset to train')
            
            
            classes = inputs[0].replace(',', "").split(",")
            batch = int(inputs[1])
            epochs = int(inputs[2])
            pretrained_model = inputs[3]

            output_path = classes[0] + "/models"
            train(classes, dataset_path, pretrained_model, output_path=output_path, batch_size=batch, num_epochs=epochs)
        else:
            print("Dialog Error")


    def processtrigger(self,q):
        try:
            self.log(q.text() + " is triggered")
            if q.text() == "Display Help":
                self.display_help()
            elif q.text() == "Add Region":
                self.region_state = True
            elif q.text() == "Delete Region":
                self.del_region_state = True
            elif q.text() == "Resize Video":
                width, okPressed = PyQt5.QtWidgets.QInputDialog.getInt(self, 'Width', 'Width:', self.resolution_x)
                height, okPressed = PyQt5.QtWidgets.QInputDialog.getInt(self, 'height', 'Height:', self.resolution_y)
                if okPressed and height >= 0 and width >= 0:
                    self.resolution_x = int(width)
                    self.resolution_y = int(height)

            elif q.text() == "Resize to default resolution":
                self.resolution_x = int(self.original_resolution[0])
                self.resolution_y = int(self.original_resolution[1])

            elif q.text() == "Quit":
                self.quit_State = True

                final_message = ""
                # Handles a case where information is not saved before exiting
                if self.exported_activity is False:
                    warning = PyQt5.QtWidgets.QMessageBox()
                    warning.setIcon(PyQt5.QtWidgets.QMessageBox.Warning)
                    warning.setWindowTitle("Export Activity")
                    warning.setText("Your activity is not exported (Export Activity not used)... \n Do you still want to continue? (Cancel will export data)")
                    warning.setStandardButtons(PyQt5.QtWidgets.QMessageBox.Save | PyQt5.QtWidgets.QMessageBox.Cancel | PyQt5.QtWidgets.QMessageBox.Close)
                    # answer = warning.buttonClicked.connect(warning)
                    answer = warning.exec()

                    
                    
                    if answer == PyQt5.QtWidgets.QMessageBox.Save:
                        self.quit_State = False
                        self.export_activity = True
                        self.exported_activity = True

                        final_message += "\n Activity Exported."


                    if answer == PyQt5.QtWidgets.QMessageBox.Cancel:
                        self.quit_State = False

                        final_message += "\n Activity Export skipped."


                if self.exported_data is False:
                    warning = PyQt5.QtWidgets.QMessageBox()
                    warning.setIcon(PyQt5.QtWidgets.QMessageBox.Warning)
                    warning.setWindowTitle("Export All")
                    warning.setText("All of your data is not saved (Export All not used)... \n Do you still want to continue?")
                    warning.setStandardButtons(PyQt5.QtWidgets.QMessageBox.Save | PyQt5.QtWidgets.QMessageBox.Cancel | PyQt5.QtWidgets.QMessageBox.Close)
                    # answer = warning.buttonClicked.connect(warning)
                    answer = warning.exec()
                    
                    if answer == PyQt5.QtWidgets.QMessageBox.Save:
                        self.quit_State = False
                        self.export_state = True
                        self.exported_data = True

                        final_message += "\n Export All called."

                    if answer == PyQt5.QtWidgets.QMessageBox.Cancel:
                        self.quit_State = False
                        final_message += "\n Export All skipped."

                if not self.quit_State:
                    self.show_warning(final_message)

                # exit(0)
            elif q.text() == "Active" or q.text() == "Inactive" or q.text() == "Read" or q.text() == "Write":
                self.set_all_tabs(q.text())
            elif q.text() == "Snap Closest":
                self.snap_state = "Closest"
            elif q.text() == "Snap Forward":
                self.snap_state = "Forward"
            elif q.text() == "Snap Backward":
                self.snap_state = "Backward"
            elif q.text() == "Set Tracker":
                # print("SETTING")
                self.set_tracker_state = True
            elif q.text() == "Retain Region":
                self.toggle_retain_region()
            elif q.text() == "Export All":
                self.export_all_state = True
                self.exported_data = True
            elif q.text() == "Play/Pause":
                self.mediaStateChanged()

            elif q.text() == "Predict":
                self.log("Predict selected. Please wait while it loads the model...")
                progress = PyQt5.QtWidgets.QProgressDialog(self)
                PyQt5.QtCore.QCoreApplication.processEvents()
                self.predict_state = True
                h5, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self,"PyQt5.QtWidgets.QFileDialog.getOpenFileName()", "Select A Model","H5 (*.h5);;All Files (*)")
                import maskrcnn
                frame, rois, scores = maskrcnn.predict(self.filename, model=h5, step=self.skip_frames.value(), display=True, progress=progress, logger=self.log, class_names=['BG', 'Vervet'])  
                
            elif q.text() == "Load Predictions":
                self.load_predictions_state = True
            elif q.text() == "Load Trackers From CSV":
                self.load_tracked_video_state = True
                self.was_loaded = True
                self.version_name, self.newest_version = self.get_new_data_version()
                # self.log(filename)
                self.data_version_updated = True
                
            elif q.text() == "Return to Beginning":
                self.set_scrollbar(0)
                self.scrollbar_changed = True
                # self.scrollbar_changed = True
            elif q.text() == "Snap Scroll to Skip":
                self.snap_to_frame_skip = not self.snap_to_frame_skip
            elif q.text() == "Mask-RCNN Options":
                self.mcrnn_options.show()
            elif q.text() == "Predictor Options":
                self.predictor_options.show()
            elif q.text() == "Image Options":
                self.image_options.show()
            elif q.text() == "Evaluate Errors":
                csv, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self,"PyQt5.QtWidgets.QFileDialog.getOpenFileName()", "Select Tracked CSV","CSV (*.csv);;All Files (*)")
                is_prediction = PyQt5.QtWidgets.QMessageBox()
                ret = is_prediction.question(self,'', "Was this data predicted by Mask RCNN?", PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.No)
                self.evaluate_errors(csv, ret)
            elif q.text() == "Evaluate Occlusion":
                self.evaluate_occlusion = True
            elif q.text() == "Train":
                self.train_model()
            elif q.text() == "Track Predictions":
                self.track_preds_state = True
            elif q.text() == "Export Charactoristics":
                self.export_charactoristics = True
            elif q.text() == "Export Activity":
                self.export_activity = True
                self.exported_activity = True # To test if this has been called
            elif q.text() == "Remove Tracked Frame":
                self.del_frame = True
            elif q.text() == "Remove All Active Tracked Frame":
                self.del_active_frame = True
            elif q.text() == "Open Live Camera":
                self.record_live = True
                self.nothing_loaded = False
                self.filename = 2
                self.vidScroll.setEnabled(False)
                self.skip_frames.setEnabled(False)
                self.play_state = True
                # self.skip_frames = 2


            

                
        except:
            crashlogger.log(str(traceback.format_exc()))

    def initUI(self, start_file):

        try:
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
            self.setWindowIcon(PyQt5.QtGui.QIcon('person.svg'))

            self.mcrnn_options = MaskRCNN_IOU_Options()
            self.predictor_options = Predictor_Options()
            self.image_options = Image_Enhancement()

            self.log_label = PyQt5.QtWidgets.QLabel(self)
            self.log_label.setText("Info:")
            self.log_label.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
            self.log_label.setFixedHeight(18)
            # self.log_label.setFixedWidth(24)
            
            if start_file is None:
                self.filename = self.openFileNameDialog()
                print(self.filename)
            else:
                self.filename = start_file
            # self.openFileNamesDialog()
            # self.saveFileDialog()

            self.layout = PyQt5.QtWidgets.QVBoxLayout(self)

            # Menu bar
            bar = PyQt5.QtWidgets.QMenuBar()
            file = bar.addMenu("File")

            live_camera = PyQt5.QtWidgets.QAction("Open Live Camera", self)
            file.addAction(live_camera)

            file.addAction("New")

            save = PyQt5.QtWidgets.QAction("Save",self)
            save.setShortcut("Ctrl+S")
            file.addAction(save)

            load_trackers = PyQt5.QtWidgets.QAction("Load Trackers From CSV", self)
            file.addAction(load_trackers)

            export_videochar = PyQt5.QtWidgets.QAction("Export Charactoristics", self)
            file.addAction(export_videochar)

            export_activity = PyQt5.QtWidgets.QAction("Export Activity", self)
            file.addAction(export_activity)

            train = PyQt5.QtWidgets.QAction("Train", self)
            file.addAction(train)
            predict = PyQt5.QtWidgets.QAction("Predict",self)
            file.addAction(predict)
            load_preds = PyQt5.QtWidgets.QAction("Load Predictions", self)
            file.addAction(load_preds)

            track_preds = PyQt5.QtWidgets.QAction("Track Predictions", self)
            file.addAction(track_preds)

            evaluate_menu = file.addMenu("Evaluate")

            evaluate_errors = PyQt5.QtWidgets.QAction("Evaluate Errors", self)
            evaluate_occlusion = PyQt5.QtWidgets.QAction("Evaluate Occlusion", self)
            # file.addAction(evaluate_errors)
            evaluate_menu.addAction(evaluate_errors)
            evaluate_menu.addAction(evaluate_occlusion)
            
            quit = PyQt5.QtWidgets.QAction("Quit", self)
            file.addAction(quit)
            
            
            export_all = PyQt5.QtWidgets.QAction("Export All", self)
            export_all.setShortcut("Ctrl+E")
            file.addAction(export_all)

            file.triggered[PyQt5.QtWidgets.QAction].connect(self.processtrigger)


            edit = bar.addMenu("Edit")
            add_region = PyQt5.QtWidgets.QAction("Add Region", self)
            add_region.setShortcut("Ctrl+R")
            del_region = PyQt5.QtWidgets.QAction("Delete Region", self)
            del_region.setShortcut("Ctrl+Shift+R")

            remove_tracked_frame = PyQt5.QtWidgets.QAction("Remove Tracked Frame", self)
            remove_tracked_frame.setShortcut("Backspace")

            remove_all_active_tracked_frame = PyQt5.QtWidgets.QAction("Remove All Active Tracked Frame", self)
            remove_all_active_tracked_frame.setShortcut("Shift+Backspace")

            set_all_read = PyQt5.QtWidgets.QAction("Set All Read", self)
            set_all_write = PyQt5.QtWidgets.QAction("Set All Write", self)

            snap_closest = PyQt5.QtWidgets.QAction("Snap Closest", self)
            snap_closest.setShortcut(PyQt5.QtCore.Qt.Key_Up)
            snap_forward = PyQt5.QtWidgets.QAction("Snap Forward", self)
            snap_forward.setShortcut(PyQt5.QtCore.Qt.Key_Right)
            snap_backward = PyQt5.QtWidgets.QAction("Snap Backward", self)
            snap_backward.setShortcut(PyQt5.QtCore.Qt.Key_Left)
            snap_skip_frame = PyQt5.QtWidgets.QAction("Snap Scroll to Skip", self)

            set_tracker = PyQt5.QtWidgets.QAction("Set Tracker", self)
            # set_tracker.setShortcut(PyQt5.QtCore.Qt.Key_Space)
            set_tracker.setShortcuts([PyQt5.QtGui.QKeySequence(PyQt5.QtCore.Qt.CTRL + PyQt5.QtCore.Qt.Key_Z), PyQt5.QtGui.QKeySequence(PyQt5.QtCore.Qt.CTRL + PyQt5.QtCore.Qt.Key_Space), PyQt5.QtGui.QKeySequence(PyQt5.QtCore.Qt.CTRL + PyQt5.QtCore.Qt.Key_D) ])

            retain_moveing_region = PyQt5.QtWidgets.QAction("Retain Region", self)
            retain_moveing_region.setShortcut("Ctrl+C")

            beginning = PyQt5.QtWidgets.QAction("Return to Beginning", self)
            

            play_key = PyQt5.QtWidgets.QAction("Play/Pause",self)
            play_key.setShortcuts([PyQt5.QtGui.QKeySequence(PyQt5.QtCore.Qt.Key_P), PyQt5.QtGui.QKeySequence(PyQt5.QtCore.Qt.CTRL + PyQt5.QtCore.Qt.Key_P) ])

            maskrcnn_options_action = PyQt5.QtWidgets.QAction("Mask-RCNN Options", self)
            predictor_options_action = PyQt5.QtWidgets.QAction("Predictor Options", self)
            image_options_action = PyQt5.QtWidgets.QAction("Image Options", self)


            clear_selection_focus = PyQt5.QtWidgets.QAction("Clear Selection Focus", self)
            clear_selection_focus.triggered.connect(lambda: self.clear_focus())
            clear_selection_focus.setShortcut("Escape")


            # edit2 = file.addMenu("Edit")
            # AddTab	Ctrl+T
            edit.addAction(remove_tracked_frame)
            edit.addAction(remove_all_active_tracked_frame)
            edit.addAction(add_region)
            edit.addAction(del_region)
            edit.addAction(snap_closest)
            edit.addAction(snap_forward)
            edit.addAction(snap_backward)
            edit.addAction(snap_skip_frame)
            edit.addAction(set_tracker)
            edit.addAction(retain_moveing_region)
            edit.addAction(play_key)
            edit.addAction(beginning)
            edit.addAction(maskrcnn_options_action)
            edit.addAction(predictor_options_action)
            edit.addAction(image_options_action)
            edit.addAction(clear_selection_focus)
            edit.triggered[PyQt5.QtWidgets.QAction].connect(self.processtrigger)

            
            
            #Set_All
            set_all =  edit.addMenu("Set All")
            # active = PyQt5.QtWidgets.QAction("Active", self)
            # active.setShortcut("Ctrl+A")
            # active.setToolTip("Sets ALL tabs tracking to active.")
            # inactive = PyQt5.QtWidgets.QAction("Inactive", self)
            # inactive.setShortcut("Ctrl+Shift+A")
            # inactive.setToolTip("Sets ALL tabs traking to inactive. (Deselects active)")
            read_on = PyQt5.QtWidgets.QAction("Read", self)
            read_on.setShortcut("Ctrl+W")
            read_on.setToolTip("Sets ALL tabs to read only. Will not overwrite data. (Good for scrolling)")
            write_on = PyQt5.QtWidgets.QAction("Write", self)
            write_on.setShortcut("Ctrl+Shift+W")
            write_on.setToolTip("Sets ALL tabs to WRITE. WILL OVERWRITE DATA WHEN SCROLLING (WARNING)")

            # set_all.addAction(active)
            # set_all.addAction(inactive)
            set_all.addAction(read_on)
            set_all.addAction(write_on)


            viewMenu = bar.addMenu("View")
            resizeVideo = PyQt5.QtWidgets.QAction("Resize Video", self)
            viewMenu.addAction(resizeVideo)
            viewMenu.triggered[PyQt5.QtWidgets.QAction].connect(self.processtrigger)
            
            resize_to_original = PyQt5.QtWidgets.QAction("Resize to default resolution", self)
            viewMenu.triggered[PyQt5.QtWidgets.QAction].connect(self.processtrigger)
            viewMenu.addAction(resize_to_original)

            helpMenu = bar.addMenu("Help")
            helpButton = PyQt5.QtWidgets.QAction("Display Help", self)
            helpButton.setShortcut("Ctrl+H")
        
            helpMenu.addAction(helpButton)
            helpMenu.triggered[PyQt5.QtWidgets.QAction].connect(self.processtrigger)

            # self.setLayout(layout)
            self.layout.addWidget(bar)

            # self.toolbar.addAction(exitAct)
            self.add_tab_state = False

            self.tab_control_layout = PyQt5.QtWidgets.QHBoxLayout()
            self.add_tab_btn = PyQt5.QtWidgets.QPushButton()
            self.add_tab_btn.setText("Add Tab")
            self.add_tab_btn.setToolTip("Adds a tracking object to the project.\n\n"+
            "Left Click and Drag on the video to create a tracking box.\n" + 
            "If unsatisfied with the selection, repeat the Left Click and Drag.\n" +
            "When satisfied, press Space Bar.")
            self.add_tab_btn.clicked.connect(self.add_tab)
            self.tab_control_layout.addWidget(self.add_tab_btn)

            # self.add_tab_btn.setEnabled(False)

            self.export_tab_btn = PyQt5.QtWidgets.QPushButton()
            self.export_tab_btn.setText("Export Data")
            self.export_tab_btn.setToolTip("Exports data and appends it to a .csv named after the video.")
            self.tab_control_layout.addWidget(self.export_tab_btn)
            self.export_tab_btn.clicked.connect(self.export_tab_pressed)

            self.del_tab_state = False
            self.del_tab_btn = PyQt5.QtWidgets.QPushButton()
            self.del_tab_btn.setText("Delete Tab")
            self.del_tab_btn.setToolTip("Deletes Tracked Object from project. \n\nWARNING!Export before removing.\nWait until box is cleared to click again.")
            self.del_tab_btn.setEnabled(False)
            self.del_tab_btn.clicked.connect(self.remove_tab)
            
            
            self.tab_control_layout.addWidget(self.del_tab_btn)

            # self.del_tab_btn.setEnabled(False)

            self.row2 = PyQt5.QtWidgets.QHBoxLayout()

            self.num_people_btn = PyQt5.QtWidgets.QPushButton("Total in view")
            self.num_people_btn.clicked.connect(lambda: self.get_integer())
            self.num_people_btn.setFixedWidth(100)
            # self.num_people_btn.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
            self.num_people = PyQt5.QtWidgets.QSpinBox()
            # self.num_people.setValidator(PyQt5.QtGui.QIntValidator(0,999))
            self.num_people.setFixedWidth(50)
            self.num_people.setAlignment(PyQt5.QtCore.Qt.AlignLeft)

            self.row2.addWidget(self.num_people_btn, 0 , PyQt5.QtCore.Qt.AlignLeft)
            self.row2.addWidget(self.num_people, 1 , PyQt5.QtCore.Qt.AlignLeft)

            self.layout.addLayout(self.tab_control_layout)
            self.layout.addLayout(self.row2)

            # Initialize tab screen
            self.tabs = PyQt5.QtWidgets.QTabWidget()
            
            # self.tabs.tabText()
            self.tabs.setMovable(False)
            self.tab_list = []
            
            # Add tabs        
            # for i in range(5):
            # self.tabs.changeEvent.connect(lambda: self.tabs.setTabText(self.tabs.currentIndex(),self.tab_list[self.tabs.currentIndex()].name_line.text()))
            # self.tabs.childEvent.connect(lambda: self.tabs.setTabText(self.tabs.currentIndex(),self.tab_list[self.tabs.currentIndex()].name_line.text()))
            # Add tabs to widget
            self.layout.addWidget(self.tabs)

            # setup scrollbar for video
            self.scrollframe = PyQt5.QtWidgets.QLabel(self)
            self.scrollframe.setText("00:00")
            self.scrollframe.setFixedWidth(50)

            self.vidScroll = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal,self)
            self.vidScroll.setMinimum(0)
            self.vidScroll.setFocusPolicy(PyQt5.QtCore.Qt.NoFocus)

            #assign a 
            self.vidScroll.sliderMoved.connect(self.slider_update)
            self.vidScroll.valueChanged.connect(self.slider_new)

            #setup play/pause buttons
            self.playButton = PyQt5.QtWidgets.QPushButton()
            self.playButton.setIcon(self.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPlay))
            self.playButton.clicked.connect(self.mediaStateChanged)

            media_layout = PyQt5.QtWidgets.QHBoxLayout()
            media_layout.addWidget(self.playButton)
            media_layout.addWidget(self.vidScroll)
            media_layout.addWidget(self.scrollframe)
            # media_layout.addWidget(self.skip_frames)
            self.layout.addLayout(media_layout)

            bottom_layout = PyQt5.QtWidgets.QHBoxLayout()
            
            self.skip_frames = PyQt5.QtWidgets.QSpinBox(self)
            self.onlyInt = PyQt5.QtGui.QIntValidator()
            # self.skip_frames.setValidator(self.onlyInt)
            self.skip_frames.setValue(10)
            self.skip_frames.setFixedWidth(40)
            self.skip_frames.setAlignment(PyQt5.QtCore.Qt.AlignRight)
            self.skip_frames.setMaximum(200)
            self.skip_frames.setMinimum(1)
            # self.skip_frames.setValidator(PyQt5.QtGui.QIntValidator(-999,999))
            self.skip_frames.setToolTip("The number of frames to increment by and 'skip'. This acts as a fast forward (positive) and reverse (negative).")
            bottom_layout.addWidget(self.skip_frames)

            self.skip_label = PyQt5.QtWidgets.QLabel(self)
            self.skip_label.setText("Frame Skip")
            self.skip_label.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
            self.skip_label.setFixedHeight(12)
            bottom_layout.addWidget(self.skip_label)
            bottom_layout.addWidget(self.log_label)
            self.layout.addLayout(bottom_layout)
            self.setLayout(self.layout)
            self.show()
        except Exception as e:
            print(e)
            crashlogger.log(e)
            crashlogger.log(str(traceback.format_exc()))
        # self.skip_f
        
        
        




        # @pyQtSlot()
        # def on_click(self):
        #     print("\n")
        #     for currentQTableWidgetItem in self.tableWidget.selectedItems():
        #         print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
    
    def show_warning(self, message):
        error_dialog = PyQt5.QtWidgets.QErrorMessage()
        error_dialog.showMessage(message)
        error_dialog.setWindowTitle("PeopleTracker ERROR")
        error_dialog.exec()

    def get_integer(self):
        i, okPressed = PyQt5.QtWidgets.QInputDialog.getInt(self, "PyQt5.QtWidgets.QInputDialog().getInteger()",
                                 "Number of people in room:", 1, 0, 999, 1)
        # self.num_people.setText(str(i))
        self.num_people.setValue(i)
        if okPressed:
            return i

    def set_max_scrollbar(self, maximum):
        self.vidScroll.setMaximum(maximum)
        self.vidScroll.maximum()

    def set_scrollbar(self, value):
        if not self.record_live:
            self.vidScroll.setValue(value)
    
    def slider_update(self, value, func=None):
        self.scrollbar_changed = True

    def slider_new(self, value, func=None):
        seconds = (value/self.vid_fps) %60
        minutes = int(((value/self.vid_fps)/60)%60)
        hours = int(minutes/60)
        self.scrollframe.setText( str(hours) + ":"+ str(minutes) + ":" + str(round( ((value/self.vid_fps) %60),2 ) ))
    


    def get_scrollbar_value(self):
        # print(self.vidScroll.value())
        return self.vidScroll.value()

    def mediaStateChanged(self):
        if self.play_state == False:
            self.playButton.setIcon(
                    self.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPause))
            self.play_state = True
            self.pause_to_play = True
            # print(self.play_state)
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPlay))
            self.play_state = False
            self.play_to_pause = True
            # print(self.play_state)
        self.log("Play State: " + str(self.play_state))

    def set_pause_state(self):
        self.playButton.setIcon(
                self.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPlay))
        self.play_state = False

    def set_play_state(self):
        self.playButton.setIcon(
                self.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPause))
        self.play_state = True

    def btn_state(self, b):
        if b.text() == "Button1":
            if b.isChecked() == True:
                
                self.log(b.text()+" is selected")
            else:
                self.log(b.text()+" is deselected")
                
        if b.text() == "Button2":
            if b.isChecked() == True:
                self.log(b.text()+" is selected")
            else:
                self.log(b.text()+" is deselected")

    def set_tab_names(self):
        i = 0
        # print(tab.name_line.text())
        for tab in self.tab_list:
            if tab.name_line.text() == "":
                self.tabs.setTabText(i,"Person " + str(i + 1))
            else:
                self.tabs.setTabText(i,tab.name_line.text())
                # print(tab.name_line.text())
                # self.parent.tab_list(i).setTabText(i,tab.name_line.text())
            i += 1

    def get_current_tab_info(self):
        current_tab = self.tab_list[self.tabs.currentIndex()]
        name = current_tab.name_line.getText()
        sex = current_tab.sex_line.getText()
        group = current_tab.group_line.getText()
        desc = current_tab.desc_line.getText()
        time = current_tab.getText()
        active = current_tab.toggle_active()
        return name, sex, desc, time

    def add_tab(self, name=None):
        self.tab_list.append(TrackerTab.person_tab(self))

        self.tabs.addTab(self.tab_list[-1].tab, ("Person " + str(self.tabs.count())))
        self.add_tab_state = True
        if len(self.tab_list) > 0:
            self.del_tab_btn.setEnabled(True)
        
        # return the last tab index
        return len(self.tab_list), self.tab_list[-1]

    def remove_tab(self):
        try:
            warning = PyQt5.QtWidgets.QMessageBox()
            warning.setIcon(PyQt5.QtWidgets.QMessageBox.Warning)
            warning.setWindowTitle("Delete Tracker Warning")
            warning.setText("You are about to delete the tracker and all of the recorded information... \n Do you still want to continue?")
            warning.setStandardButtons(PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.Cancel)
            # answer = warning.buttonClicked.connect(warning)
            answer = warning.exec()
            if answer == PyQt5.QtWidgets.QMessageBox.Yes:
                self.del_tab_state = True
                self.tabs.removeTab(self.tabs.currentIndex())
                del self.tab_list[self.tabs.currentIndex()]
                self.set_tab_names()
                if len(self.tab_list) == 0:
                    self.del_tab_btn.setEnabled(False)
        except:
            crashlogger.log(str(traceback.format_exc()))

    def export_tab_pressed(self):
        self.export_state = True
    
    def set_fps_info(self, fps):
        self.vid_fps = fps

    def get_frame_skip(self):
        try:
            skip = int(self.skip_frames.value())
        except:
            self.log("Skip value non-valid. Please enter a number.")
            skip = 0
        # #ensure skip is not backwards?
        # if skip == 0:
        #     skip = 50
        return skip
    
    def openFileNameDialog(self, task=None, extensions=None):
        filename = ""
        if extensions and task:
            filename, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, task, "",("All Files (*);;(" + extensions +")") )
        else:
        # options = PyQt5.QtWidgets.QFileDialog.Options()
        # options |= PyQt5.QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self,"PyQt5.QtWidgets.QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")

        return filename
            
    def saveFileDialog(self):
        options = PyQt5.QtWidgets.QFileDialog.Options()
        options |= PyQt5.QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = PyQt5.QtWidgets.QFileDialog.getSaveFileName(self,"PyQt5.QtWidgets.QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            self.log("Saving " + fileName)
        return fileName

    def display_help(self):
        webbrowser.open('https://github.com/hobbitsyfeet/PeopleTracker/wiki')
        # msg = PyQt5.QtWidgets.QMessageBox()
        # msg.setIcon(PyQt5.QtWidgets.QMessageBox.Question)
        # msg.setWindowTitle("Help")
        # msg.setText("1) \n2)  \n3)  \n4)  \n")

    def log(self, text):
        self.log_label.setText("Info: " + str(text))
        crashlogger.log(text, "Crashlog.txt")
        # print(text)

    def set_all_tabs(self, value):
        for tab in self.tab_list:
            if value == "Read":
                # tab.read_only = True
                tab.read_only_button.setChecked(True)
            elif value == "Write":
                # tab.read_only = False
                tab.read_only_button.setChecked(False)

    def toggle_retain_region(self):
        self.log("Retain Region set to " + str(not self.retain_region))
        self.retain_region = not self.retain_region


    def get_new_data_version(self):
        # Use the video path loaded
        if self.filename is None or self.filename == "":
            return -1, -1
        # version_filename = (export_filename[:-4] + "_V")
        # version_filename = self.filename
        # out_path = os.path.abspath(self.filename)
        out_path, filename = os.path.split(self.filename)
        version_filename = filename.split(".")[0] + "_V"
        # Collect all csv in video folder
        current_csv = glob.glob((out_path + "/*.csv"))

        # Get version file number
        newest_version = 0
        for file in current_csv:
            # Check if there's a version and get the highest version count
            if version_filename in file:
                # Get the version number in it
                version = int(file.split("_V")[-1].split(".")[0])
                if newest_version <= version:
                    newest_version = version

        # Newest version will be the largest version +1
        if self.data_version_updated:
            newest_version += 1   

        version_filename += str(newest_version) + ".csv"     
        self.newest_version_path = version_filename
        self.newest_version = newest_version
        return version_filename, newest_version
        # export_meta(version_filename, new_version=True)
        # export_csv = df.to_csv (version_filename, index = None, header=False, mode='a') #Don't forget to add '.csv' at the end of the path

    def clear_focus(self):
        if self.focusWidget() is not None:
            self.focusWidget().clearFocus()

    def track_live_video(self):
        
        keep_testing = True

        while keep_testing:
            error_dialog = PyQt5.QtWidgets.QErrorMessage()
            error_dialog.showMessage("Can you see the video?")
            error_dialog.setWindowTitle("Test Live")
            error_dialog.exec()



class MaskRCNN_IOU_Options(PyQt5.QtWidgets.QWidget):
    """
    UI for setting Mask-RCNN IOU options
    """
    def __init__(self):
        super().__init__()
        layout = PyQt5.QtWidgets.QVBoxLayout()
        self.setWindowTitle("Mask-RCNN Options")
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = PyQt5.QtWidgets.QLabel("Intersection over Union is the measure comparing how well two rectangles fit over eachother. \n Like a percent fit [0-1]")
        layout.addWidget(self.label)
        iou_form = PyQt5.QtWidgets.QFormLayout()
        layout.addLayout(iou_form)

        self.setLayout(layout)

        self.active = PyQt5.QtWidgets.QCheckBox("Activate",self)
        self.active.setTristate(False)
        self.active.setCheckState(PyQt5.QtCore.Qt.Checked)
        # self.active.connect(lambda: self._toggle_checkbox(self.active))
        iou_form.addRow(self.active)

        self.min_value = PyQt5.QtWidgets.QDoubleSpinBox(self)
        # self.onlyInt = PyQt5.QtGui.QIntValidator()
        # self.skip_frames.setValidator(self.onlyInt)
        self.min_value.setValue(0.45)
        self.min_value.setFixedWidth(70)
        self.min_value.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        self.min_value.setMaximum(1)
        self.min_value.setMinimum(0)
        self.min_value.setSingleStep(0.05)
        # self.skip_frames.setValidator(PyQt5.QtGui.QIntValidator(-999,999))
        self.min_value.setToolTip("The minimum IOU value which considers to be 'out of range' (1 means that it will always be out of range, 0 means it will never be out of range)")
        iou_form.addRow("Out of Range", self.min_value)


        self.auto_assign_value = PyQt5.QtWidgets.QDoubleSpinBox(self)
        self.auto_assign_value.setValue(0.8)
        self.auto_assign_value.setFixedWidth(70)
        self.auto_assign_value.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        self.auto_assign_value.setMaximum(1)
        self.auto_assign_value.setMinimum(0)
        self.auto_assign_value.setSingleStep(0.05)
        self.auto_assign_value.setToolTip("The minimum IOU value which is used to automatically re-assign (1 needs perfect alignment to assign, 0 will always align no matter how different)")
        iou_form.addRow("Auto Assign", self.auto_assign_value)

        self.similarity = PyQt5.QtWidgets.QDoubleSpinBox(self)
        self.similarity.setValue(0.15)
        self.similarity.setFixedWidth(70)
        self.similarity.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        self.similarity.setMaximum(1)
        self.similarity.setMinimum(0)
        self.similarity.setSingleStep(0.05)
        self.similarity.setToolTip("The minimum IOU difference which is used to detect when trackers are too similar/close. (0 means there no difference, and therefore perfect alignment. 1 means they are completely different) ")
        iou_form.addRow("Too close", self.similarity)

    def get_min_value(self):
        return self.min_value.value()

    def get_auto_assign(self):
        return self.auto_assign_value.value()

    def get_similarity(self):
        return self.similarity.value()
    
    def get_active(self):
        return self.active.isChecked()

    def _toggle_checkbox(self, checkbox):
        if checkbox.isChecked():
            checkbox.setCheckState(PyQt5.QtCore.Qt.Checked)
        else:
            checkbox.setCheckState(PyQt5.QtCore.Qt.Unchecked)

class Predictor_Options(PyQt5.QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = PyQt5.QtWidgets.QVBoxLayout()
        self.setWindowTitle("IOU Options")
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = PyQt5.QtWidgets.QLabel("Intersection over Union is the measure comparing how well two rectangles fit over eachother. \n Like a percent fit [0-1]\n \n Distance is line distance between pixels.")
        layout.addWidget(self.label)
        iou_form = PyQt5.QtWidgets.QFormLayout()
        layout.addLayout(iou_form)

        self.setLayout(layout)

        self.activate_iou = PyQt5.QtWidgets.QCheckBox("Activate IOU",self)
        self.activate_iou.setTristate(False)
        self.activate_iou.setCheckState(PyQt5.QtCore.Qt.Checked)
        # self.activate_iou.connect(lambda: self._toggle_checkbox(self.activate_iou))
        iou_form.addRow(self.activate_iou)

        self.min_value_IOU = PyQt5.QtWidgets.QDoubleSpinBox(self)
        # self.onlyInt = PyQt5.QtGui.QIntValidator()
        # self.skip_frames.setValidator(self.onlyInt)
        self.min_value_IOU.setValue(0.45)
        self.min_value_IOU.setFixedWidth(70)
        self.min_value_IOU.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        self.min_value_IOU.setMaximum(1)
        self.min_value_IOU.setMinimum(0)
        self.min_value_IOU.setSingleStep(0.05)
        # self.skip_frames.setValidator(PyQt5.QtGui.QIntValidator(-999,999))
        self.min_value_IOU.setToolTip("The minimum IOU value which considers to be 'out of range' (1 means that it will always be out of range, 0 means it will never be out of range)")
        iou_form.addRow("Bounding Box Out of Range", self.min_value_IOU)

        self.activate_centroid = PyQt5.QtWidgets.QCheckBox("Activate Centroid",self)
        self.activate_centroid.setTristate(False)
        self.activate_centroid.setCheckState(PyQt5.QtCore.Qt.Checked)
        # self.activate_centroid.connect(lambda: self._toggle_checkbox(self.activate_centroid))
        iou_form.addRow(self.activate_centroid)

        self.min_value_distance = PyQt5.QtWidgets.QDoubleSpinBox(self)
        self.min_value_distance.setValue(20)
        self.min_value_distance.setFixedWidth(70)
        self.min_value_distance.setAlignment(PyQt5.QtCore.Qt.AlignLeft)
        # self.min_value_distance.setMaximum()
        self.min_value_distance.setMinimum(0)
        self.min_value_distance.setSingleStep(10)
        self.min_value_distance.setToolTip("The minimum straight line (euclidean) distance (in pixels) which is used to detect when the predicted tracker is too different")
        iou_form.addRow("Centroid Out of Range", self.min_value_distance)

    def get_min_IOU(self):
        return self.min_value_IOU.value()
    
    def get_min_distance(self):
        return self.min_value_distance.value()
    
    def get_active_iou(self):
        return self.activate_iou.isChecked()
    
    def get_active_centroid(self):
        return self.activate_centroid.isChecked()

    def _toggle_checkbox(self, checkbox):
        if checkbox.isChecked():
            checkbox.setCheckState(PyQt5.QtCore.Qt.Checked)
        else:
            checkbox.setCheckState(PyQt5.QtCore.Qt.Unchecked)

class Image_Enhancement(PyQt5.QtWidgets.QWidget):
    """
    UI for setting Mask-RCNN IOU options
    """
    def __init__(self):
        super().__init__()
        layout = PyQt5.QtWidgets.QVBoxLayout()
        self.setWindowTitle("Image Options")
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.label = PyQt5.QtWidgets.QLabel("Image options for brightness and contrast.")
        layout.addWidget(self.label)
        bar_form = PyQt5.QtWidgets.QFormLayout()
        layout.addLayout(bar_form)
        self.setLayout(layout)

        # self.brightness = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal)
        # self.brightness.setMaximum(255)
        # self.brightness.setMinimum(-255)
        # self.brightness.setValue(0)
        
        
        # self.brightness_label = PyQt5.QtWidgets.QLabel("brightness: " + str(self.brightness.value()))
        # self.brightness.valueChanged.connect(lambda: self.update_brightness())
        # bar_form.addRow(self.brightness_label , self.brightness)



        self.alpha = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal)
        self.alpha.setValue(10)
        self.alpha.setMinimum(0)
        self.alpha.setMaximum(100)
        self.alpha_label = PyQt5.QtWidgets.QLabel("Contrast: " + str(self.alpha.value()))
        self.alpha.valueChanged.connect(lambda: self.update_alpha())
        bar_form.addRow(self.alpha_label , self.alpha)

        self.beta = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal)
        self.beta.setValue(10)
        self.beta.setMinimum(-200)
        self.beta.setMaximum(2000)
        self.beta_label = PyQt5.QtWidgets.QLabel("Brightness: "+ str(self.beta.value()))
        self.beta.valueChanged.connect(lambda: self.update_beta())
        bar_form.addRow(self.beta_label, self.beta)

        self.gamma = PyQt5.QtWidgets.QSlider(PyQt5.QtCore.Qt.Horizontal)
        self.gamma.setValue(10)
        self.gamma.setMinimum(0)
        self.gamma.setMaximum(250)
        self.gamma.setSingleStep(0.1)
        self.gamma_label = PyQt5.QtWidgets.QLabel("gamma: "+ str(self.gamma.value()/10))
        self.gamma.valueChanged.connect(lambda: self.update_gamma())
        bar_form.addRow(self.gamma_label, self.gamma)

        self.roi_normalize = PyQt5.QtWidgets.QPushButton("Normalize Region")
        self.roi_normalize.clicked.connect(lambda: self.set_normalize_flag())
        self.roi_normalize_region = None
        self.roi_normalize_flag = False
        # self.roi_clear_button = PyQt5.QtWidgets.QPushButton("Clear Region")
        # self.roi_clear_button.clicked.connect(lambda: self.clear_normalized_region())
        # bar_form.addWidget(self.roi_normalize)
        # bar_form.addWidget(self.roi_clear_button)
        self.reset_button = PyQt5.QtWidgets.QPushButton("Default")
        self.reset_button.clicked.connect(lambda: self.reset_default())
        bar_form.addWidget(self.reset_button)

        self.equalize_hist_button = PyQt5.QtWidgets.QCheckBox("Equalize Histogram")
        self.equalize_hist_button.setChecked(False)
        self.equalize_hist_button.clicked.connect(lambda: self.toggle_equalize_hist())
        self.equalize_hist_button.setToolTip("Uses OpenCV's Equalize Histogram method. 'Improves the contrast in an image'")
        bar_form.addWidget(self.equalize_hist_button)
        

        self.equalize_chahe_hist_button = PyQt5.QtWidgets.QCheckBox("CLAHE Equalize Histogram")
        self.equalize_chahe_hist_button.setChecked(False)
        self.equalize_chahe_hist_button.clicked.connect(lambda: self.toggle_chahe_hist())
        self.equalize_chahe_hist_button.setToolTip("Uses OpenCV's Equalize Histogram method. 'Improves the contrast in an image'")
        bar_form.addWidget(self.equalize_chahe_hist_button)

    def reset_default(self):
        self.gamma.setValue(10)
        self.beta.setValue(10)
        self.alpha.setValue(10)
        self.roi_normalize_region = None
        # self.brightness.setValue(0)

    def update_brightness(self):
        self.brightness_label.setText("brightness: "+ str(self.brightness.value()))

    def update_gamma(self):
        self.gamma_label.setText("gamma: "+ str(self.gamma.value()/10))

    def update_alpha(self):
        self.roi_normalize_region = None
        self.alpha_label.setText("Alpha: " + str(self.alpha.value()/10))

    def update_beta(self):
        self.beta_label.setText("Beta: "+ str(self.beta.value()/10))
    
    def toggle_equalize_hist(self):
        self.equalize_hist_button.setChecked(self.equalize_hist_button.isChecked())

    def toggle_chahe_hist(self):
        self.equalize_chahe_hist_button.setChecked(self.equalize_chahe_hist_button.isChecked())

    def get_equalize_hist(self):
        return self.equalize_hist_button.isChecked()

    def get_equalize_clahe_hist(self):
        return self.equalize_chahe_hist_button.isChecked()

    def get_alpha(self):
        return self.alpha.value()

    def get_beta(self):
        return self.beta.value()
    
    def get_gamma(self):
        return self.gamma.value()

    

    # def get_contrast()

    def set_normalized_region(self, image):
        self.roi_normalize_region = cv2.selectROI("Frame", image)
    
    def clear_normalized_region(self):
        self.roi_normalize_region = None

    def set_normalize_flag(self):
        self.roi_normalize_flag = True

    def enhance_normalized_roi(self, image):
        if self.roi_normalize_region is not None:
            # Calculate mean and STD
            # norm_crop = image.crop(self.roi_normalize_region)
            x = self.roi_normalize_region[0]
            y = self.roi_normalize_region[1]
            w = self.roi_normalize_region[2]
            h = self.roi_normalize_region[3]
            norm_crop = image[y:y+h,x:x+w]

        
            gray_crop = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2GRAY)
            avg = np.average(gray_crop)
            increase = 255 - avg
            normalized = (increase-0)/(255-0)
            # print("normalized", normalized)
            try:
                normalized = ((normalized + 1)/(10 + 1))  *100
                # print("-1 to 10" , normalized)
                self.alpha.setValue(normalized)
            except Exception as e:
                print(e)
                
            # print(image)
            # mean, STD  = cv2.meanStdDev(self.roi_normalize_region)
            # print(mean,STD)

            # # Clip frame to lower and upper STD
            # offset = 0.2
            # clipped = np.clip(image, mean - offset*STD, mean + offset*STD).astype(np.uint8)

            # res = cv.convertScaleAbs(img, alpha = alpha, beta = beta)
            # self.alpha.setValue(255/mean)
            # self.beta.setValue()
            # print(clipped)
            # # Normalize to range
            # return cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
        return image

    def add_brightness(self, image):
        bright = image + self.brightness.value()
        bright = np.clip(bright,0,255)
        return bright.astype('uint8')

    def enhance_brightness_contrast(self, image):
        return cv2.convertScaleAbs(image, alpha=self.alpha.value()/10, beta=self.beta.value()/10)
    
    def enhance_gamma(self, image):
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, self.gamma.value()/10) * 255.0, 0, 255)
        new_image = cv2.LUT(image, lookUpTable)
        return new_image
        
    def equalize_hist(self, image):
        R, G, B = cv2.split(image)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)

        return cv2.merge((output1_R, output1_G, output1_B))

    def equalize_clahe_hist(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr

    def auto_enhance(self, image):
        # METHOD 1: RGB
        # convert img to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(gray)
        gamma = math.log(mid*255)/math.log(mean)
        print(gamma)

        # do gamma correction
        img_gamma1 = np.power(image, gamma).clip(0,255).astype(np.uint8)



        # METHOD 2: HSV (or other color spaces)

        # convert img to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(val)
        gamma = math.log(mid*255)/math.log(mean)
        print(gamma)

        # do gamma correction on value channel
        val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

        # combine new value channel with original hue and sat channels
        hsv_gamma = cv2.merge([hue, sat, val_gamma])
        img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

        return img_gamma1, img_gamma2


    # def update_tab_name(self):
    #     self.tab.parentWidget().setTabText(self.tab.parent.currentIndex(),self.name_line.getText())
    #     print(self.tab.parentWidget)

# class VideoWindow(PyQt5.QtWidgets.QWidget):
#     """
#     This "window" is a PyQt5.QtWidgets.QWidget. If it has no parent, it 
#     will appear as a free-floating window as we want.
#     """
#     def __init__(self):
#         super().__init__()
#         layout = PyQt5.QtWidgets.QVBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         self.setLayout(layout)
#         self.label = label()
#         self.image = PyQt5.QtGui.QPixmap("./Blank.jpg")
#         self.label.setPixmap(self.image)
#         layout.addWidget(self.label)
#         self.pressed_pos = None
#         self.dragged_pos = None
#         self.setting = False

    
#     def mousePressEvent(self, event):
#         print("Pressed")
#         self.setting = True
#         print(event.x(), event.y())
#         self.pressed_pos = (event.x(), event.y())

#     def mouseMoveEvent(self, event):
#         print(event.x(), event.y())
#         self.dragged_pos = (event.x(), event.y())

#     def mouseReleaseEvent(self, event):
#         print("Released")
#         self.setting = False

#     def paintEvent(self, event):

#         painter = PyQt5.QtGui.QPainter(self)

#         painter.setPen(PyQt5.QtGui.QPen(PyQt5.QtCore.Qt.black,  5, PyQt5.QtCore.Qt.SolidLine))

#         painter.drawRect(40, 40, 400, 200)

#     def show_image(self, cv2Frame):
#         print("ShowImage")
#         if cv2Frame is not None:
#             # print(cv2Frame)
#             height, width, channel = cv2Frame.shape
#             bytesPerLine = 3 * width
#             qimg = PyQt5.QtGui.QImage(cv2Frame.data, width, height, bytesPerLine, PyQt5.QtGui.QImage.Format_RGB888).rgbSwapped()
#             # print(qimg)
#             self.label.setPixmap(PyQt5.QtGui.QPixmap(qimg))



# class label (PyQt5.QtWidgets.QLabel):
#     def __init__(self):
#         super().__init__()
#         def mousePressEvent(self, event):
#             super(self).mousePressEvent(QMouseEvent)
#             print("MOUSE")

if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
