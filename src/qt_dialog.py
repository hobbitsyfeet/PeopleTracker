import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QPlainTextEdit, QSlider, QStyle, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QCheckBox, QMenuBar, QSpinBox, QErrorMessage, QProgressDialog

from PyQt5.QtGui import QIcon, QIntValidator, QPixmap, QImage, QPainter, QPen, QKeySequence
from PyQt5.QtCore import Qt, QRect, QCoreApplication
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import webbrowser
import crashlogger
import traceback
# import maskrcnn



class App(QWidget):
# TODO ADD A CURRENT STATE INFO TAG WHICH LETS THE USER KNOW WHAT TO DO AT A CERTAIN TIME!
    def __init__(self):
        super().__init__()
        self.title = 'Person'
        self.left = 250
        self.top = 250
        self.width = 480
        self.height = 240
        self.initUI()

        self.play_state = False
        self.export_state = False
        self.region_state = False
        self.del_region_state = False
        self.export_all_state = False
        self.scrollbar_changed = False
        self.resolution_x = 720
        self.resolution_y = 480
        self.vid_fps = 30
        self.snap_state = None
        self.set_tracker_state = False
        self.retain_region = True
        self.quit_State = False
        self.image = None

        self.predict_state = False
        
    
        # self.videoWindow = VideoWindow()
        # self.videoWindow.show()

    def keyPressEvent(self, event):
            self.test_method()
            if int(event.modifiers()) == (Qt.ControlModifier+Qt.AltModifier):
                self.log("Setting Tracker")
                self.set_tracker_state = True

    def mousePressEvent(self, event):
        if event.button() == Qt.MidButton:
            self.set_tracker_state = True
        elif event.button() == Qt.RightButton:
            self.log("Play Toggled.")
            self.mediaStateChanged()

            
    def test_method(self):
        print('key pressed')

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
                width, okPressed = QInputDialog.getInt(self, 'Width', 'Width:', self.resolution_x)
                height, okPressed = QInputDialog.getInt(self, 'height', 'Height:', self.resolution_y)
                if okPressed and height >= 0 and width >= 0:
                    self.resolution_x = int(width)
                    self.resolution_y = int(height)
            elif q.text() == "Quit":
                self.quit_State = True
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
            elif q.text() == "Play/Pause":
                
                self.mediaStateChanged()
            elif q.text() == "Predict":
                self.log("Predict selected. Please wait while it loads the model...")
                progress = QProgressDialog(self)
                QCoreApplication.processEvents()
                # self.predict_state = True
                # frame, rois, scores = maskrcnn.predict(self.filename, step=self.skip_frames.value(), display=True, progress=progress, logger=self.log)  
                
        except:
            crashlogger.log(str(traceback.format_exc()))

    def initUI(self):

        try:
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)
            self.setWindowIcon(QIcon('person.svg'))

            self.log_label = QLabel(self)
            self.log_label.setText("Info:")
            self.log_label.setAlignment(Qt.AlignLeft)
            self.log_label.setFixedHeight(18)
            # self.log_label.setFixedWidth(24)
            
            
            self.openFileNameDialog()
            # self.openFileNamesDialog()
            # self.saveFileDialog()

            self.layout = QVBoxLayout(self)

            # Menu bar
            bar = QMenuBar()
            file = bar.addMenu("File")
            file.addAction("New")

            save = QAction("Save",self)
            save.setShortcut("Ctrl+S")
            predict = QAction("Predict",self)
            file.addAction(predict)
            file.addAction(save)
            quit = QAction("Quit", self)
            file.addAction(quit)
            
            
            export_all = QAction("Export All", self)
            export_all.setShortcut("Ctrl+E")
            file.addAction(export_all)

            file.triggered[QAction].connect(self.processtrigger)

            edit = bar.addMenu("Edit")
            add_region = QAction("Add Region", self)
            add_region.setShortcut("Ctrl+R")
            del_region = QAction("Delete Region", self)
            del_region.setShortcut("Ctrl+Shift+R")

            set_all_read = QAction("Set All Read", self)
            set_all_write = QAction("Set All Write", self)

            snap_closest = QAction("Snap Closest", self)
            snap_closest.setShortcut(Qt.Key_Up)
            snap_forward = QAction("Snap Forward", self)
            snap_forward.setShortcut(Qt.Key_Right)
            snap_backward = QAction("Snap Backward", self)
            snap_backward.setShortcut(Qt.Key_Left)

            set_tracker = QAction("Set Tracker", self)
            # set_tracker.setShortcut(Qt.Key_Space)
            set_tracker.setShortcuts([QKeySequence(Qt.CTRL + Qt.Key_Z), QKeySequence(Qt.CTRL + Qt.Key_Space) ])

            retain_moveing_region = QAction("Retain Region", self)
            retain_moveing_region.setShortcut("Ctrl+C")


            play_key = QAction("Play/Pause",self)
            play_key.setShortcuts([QKeySequence(Qt.Key_P), QKeySequence(Qt.CTRL + Qt.Key_P) ])


            # edit2 = file.addMenu("Edit")
            # AddTab	Ctrl+T
            edit.addAction(add_region)
            edit.addAction(del_region)
            edit.addAction(snap_closest)
            edit.addAction(snap_forward)
            edit.addAction(snap_backward)
            edit.addAction(set_tracker)
            edit.addAction(retain_moveing_region)
            edit.addAction(play_key)
            edit.triggered[QAction].connect(self.processtrigger)

            
            
            #Set_All
            set_all =  edit.addMenu("Set All")
            # active = QAction("Active", self)
            # active.setShortcut("Ctrl+A")
            # active.setToolTip("Sets ALL tabs tracking to active.")
            # inactive = QAction("Inactive", self)
            # inactive.setShortcut("Ctrl+Shift+A")
            # inactive.setToolTip("Sets ALL tabs traking to inactive. (Deselects active)")
            read_on = QAction("Read", self)
            read_on.setShortcut("Ctrl+W")
            read_on.setToolTip("Sets ALL tabs to read only. Will not overwrite data. (Good for scrolling)")
            write_on = QAction("Write", self)
            write_on.setShortcut("Ctrl+Shift+W")
            write_on.setToolTip("Sets ALL tabs to WRITE. WILL OVERWRITE DATA WHEN SCROLLING (WARNING)")

            # set_all.addAction(active)
            # set_all.addAction(inactive)
            set_all.addAction(read_on)
            set_all.addAction(write_on)


            viewMenu = bar.addMenu("View")
            resizeVideo = QAction("Resize Video", self)

            # resizeVideo.setShortcut("Ctrl+R")
            
            viewMenu.addAction(resizeVideo)
            viewMenu.triggered[QAction].connect(self.processtrigger)


            helpMenu = bar.addMenu("Help")
            helpButton = QAction("Display Help", self)
            helpButton.setShortcut("Ctrl+H")
        
            helpMenu.addAction(helpButton)
            helpMenu.triggered[QAction].connect(self.processtrigger)

            # self.setLayout(layout)
            self.layout.addWidget(bar)

            # self.toolbar.addAction(exitAct)
            self.add_tab_state = False

            self.tab_control_layout = QHBoxLayout()
            self.add_tab_btn = QPushButton()
            self.add_tab_btn.setText("Add Tab")
            self.add_tab_btn.setToolTip("Adds a tracking object to the project.\n\n"+
            "Left Click and Drag on the video to create a tracking box.\n" + 
            "If unsatisfied with the selection, repeat the Left Click and Drag.\n" +
            "When satisfied, press Space Bar.")
            self.add_tab_btn.clicked.connect(self.add_tab)
            self.tab_control_layout.addWidget(self.add_tab_btn)

            # self.add_tab_btn.setEnabled(False)

            self.export_tab_btn = QPushButton()
            self.export_tab_btn.setText("Export Data")
            self.export_tab_btn.setToolTip("Exports data and appends it to a .csv named after the video.")
            self.tab_control_layout.addWidget(self.export_tab_btn)
            self.export_tab_btn.clicked.connect(self.export_tab_pressed)

            self.del_tab_state = False
            self.del_tab_btn = QPushButton()
            self.del_tab_btn.setText("Delete Tab")
            self.del_tab_btn.setToolTip("Deletes Tracked Object from project. \n\nWARNING!Export before removing.\nWait until box is cleared to click again.")
            self.del_tab_btn.setEnabled(False)
            self.del_tab_btn.clicked.connect(self.remove_tab)
            
            
            self.tab_control_layout.addWidget(self.del_tab_btn)

            # self.del_tab_btn.setEnabled(False)

            self.row2 = QHBoxLayout()

            self.num_people_btn = QPushButton("Total in view")
            self.num_people_btn.clicked.connect(lambda: self.get_integer())
            self.num_people_btn.setFixedWidth(100)
            # self.num_people_btn.setAlignment(Qt.AlignLeft)
            self.num_people = QSpinBox()
            # self.num_people.setValidator(QIntValidator(0,999))
            self.num_people.setFixedWidth(50)
            self.num_people.setAlignment(Qt.AlignLeft)

            self.row2.addWidget(self.num_people_btn, 0 , Qt.AlignLeft)
            self.row2.addWidget(self.num_people, 1 , Qt.AlignLeft)

            self.layout.addLayout(self.tab_control_layout)
            self.layout.addLayout(self.row2)

            # Initialize tab screen
            self.tabs = QTabWidget()
            
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
            self.scrollframe = QLabel(self)
            self.scrollframe.setText("00:00")
            self.scrollframe.setFixedWidth(50)

            self.vidScroll = QSlider(Qt.Horizontal,self)
            self.vidScroll.setMinimum(0)
            self.vidScroll.setFocusPolicy(Qt.NoFocus)

            #assign a 
            self.vidScroll.sliderMoved.connect(self.slider_update)
            self.vidScroll.valueChanged.connect(self.slider_new)

            #setup play/pause buttons
            self.playButton = QPushButton()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.playButton.clicked.connect(self.mediaStateChanged)

            media_layout = QHBoxLayout()
            media_layout.addWidget(self.playButton)
            media_layout.addWidget(self.vidScroll)
            media_layout.addWidget(self.scrollframe)
            # media_layout.addWidget(self.skip_frames)
            self.layout.addLayout(media_layout)

            bottom_layout = QHBoxLayout()
            
            self.skip_frames = QSpinBox(self)
            self.onlyInt = QIntValidator()
            # self.skip_frames.setValidator(self.onlyInt)
            self.skip_frames.setValue(10)
            self.skip_frames.setFixedWidth(40)
            self.skip_frames.setAlignment(Qt.AlignRight)
            self.skip_frames.setMaximum(200)
            self.skip_frames.setMinimum(1)
            # self.skip_frames.setValidator(QIntValidator(-999,999))
            self.skip_frames.setToolTip("The number of frames to increment by and 'skip'. This acts as a fast forward (positive) and reverse (negative).")
            bottom_layout.addWidget(self.skip_frames)

            self.skip_label = QLabel(self)
            self.skip_label.setText("Frame Skip")
            self.skip_label.setAlignment(Qt.AlignLeft)
            self.skip_label.setFixedHeight(12)
            bottom_layout.addWidget(self.skip_label)
            bottom_layout.addWidget(self.log_label)
        except:
            crashlogger.log(str(traceback.format_exc()))
        # self.skip_f
        
        
        


        self.layout.addLayout(bottom_layout)

        self.setLayout(self.layout)

        self.show()

        # @pyQtSlot()
        # def on_click(self):
        #     print("\n")
        #     for currentQTableWidgetItem in self.tableWidget.selectedItems():
        #         print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
    
    def show_warning(self, message):
        error_dialog = QErrorMessage()
        error_dialog.showMessage(message)
        error_dialog.setWindowTitle("PeopleTracker ERROR")
        error_dialog.exec()

    def get_integer(self):
        i, okPressed = QInputDialog.getInt(self, "QInputDialog().getInteger()",
                                 "Number of people in room:", 1, 0, 999, 1)
        # self.num_people.setText(str(i))
        self.num_people.setValue(i)
        if okPressed:
            return i

    def set_max_scrollbar(self, maximum):
        self.vidScroll.setMaximum(maximum)

    def set_scrollbar(self, value):
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
                    self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_state = True
            # print(self.play_state)
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_state = False
            # print(self.play_state)

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

    def add_tab(self):
        self.tab_list.append(person_tab(self))
        self.tabs.addTab(self.tab_list[-1].tab, ("Person " + str(self.tabs.count())))
        self.add_tab_state = True
        if len(self.tab_list) > 0:
            self.del_tab_btn.setEnabled(True)

    def remove_tab(self):
        try:
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Warning)
            warning.setWindowTitle("Delete Tracker Warning")
            warning.setText("You are about to delete the tracker and all of the recorded information... \n Do you still want to continue?")
            warning.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            # answer = warning.buttonClicked.connect(warning)
            answer = warning.exec()
            if answer == QMessageBox.Yes:
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
    
    def openFileNameDialog(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)")
        if fileName:
            self.filename = fileName
            self.log("Opening" + fileName)
            
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            self.log("Saving " + fileName)

    def display_help(self):
        webbrowser.open('https://github.com/hobbitsyfeet/PeopleTracker/wiki')
        # msg = QMessageBox()
        # msg.setIcon(QMessageBox.Question)
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


class person_tab():
    def __init__(self, window):
        self.parent = window
        self.tab = QWidget()
        

        #these are used to record metadata
        self.name_line = QLineEdit(window)
        self.id_line = QLineEdit(window)
        self.group_line = QLineEdit(window)
        self.group_line.setValidator(QIntValidator(0,999))
        # self.name_line.textChanged.connect(self.update_tab_name)
        self.sex_line = QLineEdit(window)
        self.desc_line = QPlainTextEdit(window)
        
        self.length_tracked = QLabel(window)
        self.length_tracked.setText("00:00")
    
        self.active = True
        self.read_only = False
        self.other_room = False
        self.beginning = False
        self.is_region = False
        self.is_chair = False
        self.init_tab(window)

    def init_tab(self, parent_window):
        try:
            self.tab.layout = QVBoxLayout(parent_window)

            #Start Name
            name_layout = QHBoxLayout()
            self.tab.layout.addLayout(name_layout)
            
            name_btn = QPushButton('Name', parent_window)
            name_btn.clicked.connect(lambda: self.getText(input_name="Name:",line=self.name_line))

            name_layout.addWidget(name_btn)

            name_layout.addWidget(self.name_line)
            

            id_btn = QPushButton('ID', parent_window)
            id_btn.clicked.connect(lambda: self.getText(input_name="ID:",line=self.id_line))

            name_layout.addWidget(id_btn)

            name_layout.addWidget(self.id_line)
            # #Start Sex
            # sex_layout = QHBoxLayout()

            sex_btn = QPushButton('Sex', parent_window)
            sex_btn.clicked.connect(self.getChoice)
            
            name_layout.addWidget(sex_btn)
            name_layout.addWidget(self.sex_line)

            group_size_btn = QPushButton('Group Size', parent_window)
            group_size_btn.clicked.connect(lambda: self.getInteger())

            name_layout.addWidget(group_size_btn)
            name_layout.addWidget(self.group_line)

            #Start Description
            desc_layout = QHBoxLayout()

            desc_btn = QPushButton('Description', parent_window)
            desc_btn.clicked.connect(lambda: self.getText(input_name="Description:",line=self.desc_line))
            
            desc_layout.addWidget(desc_btn)
            desc_layout.addWidget(self.desc_line)

            self.image = QPixmap()
            self.tab.layout.addLayout(desc_layout)
            
            #setup length of time tracked
            length_layout = QHBoxLayout()
            length_label = QLabel(parent_window)
            length_label.setText("Total Tracked (mm:ss): ")

            length_layout.addWidget(length_label)
            length_layout.addWidget(self.length_tracked)
            length_layout.setAlignment(Qt.AlignCenter)

            # self.active_button = QCheckBox("Active")
            # self.active_button.setChecked(True)
            # self.active_button.stateChanged.connect(lambda:self.toggle_active())
            # self.active_button.setToolTip("Sets the current tracking to actively record. \nIf unchecked, no box will be processed, displayed or recorded.")
            # length_layout.addWidget(self.active_button)

            self.read_only_button = QCheckBox("Read Only")
            self.read_only_button.setChecked(False)
            self.read_only_button.stateChanged.connect(lambda:self.toggle_read())
            self.read_only_button.setToolTip("Sets the person to read only. \nThis is useful for scrolling through the video without overwriting data.\n Also useful for people exiting the frame")
            length_layout.addWidget(self.read_only_button)

            self.other_room_button = QCheckBox("Other Room")
            self.other_room_button.setChecked(False)
            self.other_room_button.stateChanged.connect(lambda:self.toggle_other_room())
            self.other_room_button.setToolTip("Sets the person to Other Room. \n This is useful for maintaining time  without location.")
            length_layout.addWidget(self.other_room_button)


            self.beginning_button = QCheckBox("Beginning")
            self.beginning_button.setChecked(False)
            self.beginning_button.stateChanged.connect(lambda:self.toggle_beginning())
            self.beginning_button.setToolTip("Sets the 'present at beginning' to be True or False for this person.")
            length_layout.addWidget(self.beginning_button)
            self.tab.layout.addLayout(length_layout)

            self.is_region_button = QCheckBox("Is Region")
            self.is_region_button.setChecked(False)
            self.is_region_button.stateChanged.connect(lambda:self.toggle_region())
            length_layout.addWidget(self.is_region_button)

            self.is_chair_button = QCheckBox("Chair")
            self.is_chair_button.setChecked(False)
            self.is_chair_button.stateChanged.connect(lambda:self.toggle_chair())
            length_layout.addWidget(self.is_chair_button)


            self.tab.layout.addLayout(length_layout)


            self.tab.setLayout(self.tab.layout)
        
        except:
            crashlogger.log(str(traceback.format_exc()))

        # #inital load of variables
        # self.getText(input_name="Name:",line=self.name_line)
        # self.getChoice()
        # self.getText(input_name="Description:",line=self.desc_line)
        


    def getInteger(self):
        i, okPressed = QInputDialog.getInt(self.parent, "QInputDialog().getInteger()",
                                 "Number:", 1, 0, 999, 1)
        self.group_line.setText(str(i))
        if okPressed:
            return i

    # def getDouble(self):
    #     d, okPressed = QInputDialog.getDouble(self.parent, "Get double","Value:", 10.50, 0, 100, 10)
    #     if okPressed:
    #         return d
        
    def getChoice(self):
        items = ("Female","Male", "Other")
        item, okPressed = QInputDialog.getItem(self.parent, "Get item","Sex:", items, 0, False)
        if okPressed and item:
            # print(item)
            self.sex_line.setText(item)
            # return item

    def getText(self, input_name, line):
        text, okPressed = QInputDialog.getText(self.parent, "Get text", input_name, QLineEdit.Normal, "")
        if okPressed and text != '':
            if type(line) == type(QPlainTextEdit()):
                line.setPlainText(text)
            else:
                line.setText(text)
            # print(text)
            
            # return text

    def get_beginning(self):
        return self.beginning
    
    def get_is_region(self):
        return self.is_region

    def get_is_chair(self):
        return self.is_chair
    
    def get_read_only(self):
        return self.read_only

    def get_other_room(self):
        return self.other_room

    def toggle_active(self):
        self.parent.log("Setting Active to " + str(not self.active))
        self.active = not self.active
        return self.active
    
    def toggle_read(self):
        self.parent.log("Setting Read only to " + str(not self.read_only))
        self.read_only = not self.read_only
        return self.read_only
    
    def toggle_beginning(self):
        self.parent.log("Setting person present at beginning to " + str(not self.beginning))
        self.beginning = not self.beginning
        return self.beginning

    def toggle_region(self):
        self.parent.log("Setting person to a region " + str(not self.is_region))
        self.is_region = not self.is_region
        return self.is_region

    def toggle_other_room(self):
        self.parent.log("Setting person to other room " + str(not self.other_room))
        self.other_room = not self.other_room
        return self.other_room

    def toggle_chair(self):
        self.parent.log("Setting person in chair " + str(not self.is_region))
        self.is_chair = not self.is_chair
        return self.is_chair

    def update_length_tracked(self, time):
        self.length_tracked.setText("00:00")
        seconds = round((time)%60,2)
        minutes = int(((time)/60)%60)
        self.length_tracked.setText( str(minutes) + ":" + str(seconds))


    
    # def update_tab_name(self):
    #     self.tab.parentWidget().setTabText(self.tab.parent.currentIndex(),self.name_line.getText())
    #     print(self.tab.parentWidget)

# class VideoWindow(QWidget):
#     """
#     This "window" is a QWidget. If it has no parent, it 
#     will appear as a free-floating window as we want.
#     """
#     def __init__(self):
#         super().__init__()
#         layout = QVBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         self.setLayout(layout)
#         self.label = label()
#         self.image = QPixmap("./Blank.jpg")
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

#         painter = QPainter(self)

#         painter.setPen(QPen(Qt.black,  5, Qt.SolidLine))

#         painter.drawRect(40, 40, 400, 200)

#     def show_image(self, cv2Frame):
#         print("ShowImage")
#         if cv2Frame is not None:
#             # print(cv2Frame)
#             height, width, channel = cv2Frame.shape
#             bytesPerLine = 3 * width
#             qimg = QImage(cv2Frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
#             # print(qimg)
#             self.label.setPixmap(QPixmap(qimg))



# class label (QLabel):
#     def __init__(self):
#         super().__init__()
#         def mousePressEvent(self, event):
#             super(self).mousePressEvent(QMouseEvent)
#             print("MOUSE")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
