import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QPlainTextEdit, QSlider, QStyle, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QCheckBox, QMenuBar

from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot

class App(QWidget):
# TODO ADD A CURRENT STATE INFO TAG WHICH LETS THE USER KNOW WHAT TO DO AT A CERTAIN TIME!
# TODO PICKLE THE PROJECT TO SAVE THE STATE!!!
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
        self.scrollbar_changed = False
        self.resolution_x = 720
        self.resolution_y = 480
        self.vid_fps = 30
    
    def keyPressEvent(self, event):
            self.test_method()
            # self.log(event)
            
    def test_method(self):
        print('key pressed')

    def processtrigger(self,q):
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
            exit(0)
        elif q.text() == "Active" or q.text() == "Inactive" or q.text() == "Read" or q.text() == "Write":
            self.set_all_tabs(q.text())

    def initUI(self):


        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

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
        file.addAction(save)
        quit = QAction("Quit", self)
        file.addAction(quit)
        file.triggered[QAction].connect(self.processtrigger)

        edit = bar.addMenu("Edit")
        add_region = QAction("Add Region", self)
        add_region.setShortcut("Ctrl+R")
        del_region = QAction("Delete Region", self)
        del_region.setShortcut("Ctrl+Shift+R")

        edit2 = file.addMenu("Edit")
        edit2.addAction("copy")
        edit2.addAction("paste")
        # AddTab	Ctrl+T
        edit.addAction(add_region)
        edit.addAction(del_region)
        edit.addAction("copy")
        edit.addAction("paste")
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

        self.layout.addLayout(self.tab_control_layout)

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
        self.vidScroll.valueChanged.connect(self.slider_update)

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
        
        self.skip_frames = QLineEdit(self)
        self.onlyInt = QIntValidator()
        self.skip_frames.setValidator(self.onlyInt)
        self.skip_frames.setText("50")
        self.skip_frames.setFixedWidth(25)
        self.skip_frames.setAlignment(Qt.AlignRight)
        self.skip_frames.setToolTip("The number of frames to increment by and 'skip'. This acts as a fast forward (positive) and reverse (negative).")
        bottom_layout.addWidget(self.skip_frames)

        self.skip_label = QLabel(self)
        self.skip_label.setText("Frame Skip")
        self.skip_label.setAlignment(Qt.AlignLeft)
        self.skip_label.setFixedHeight(12)
        bottom_layout.addWidget(self.skip_label)
        bottom_layout.addWidget(self.log_label)
        # self.skip_f
        
        
        


        self.layout.addLayout(bottom_layout)

        self.setLayout(self.layout)

        self.show()

        @pyqtSlot()
        def on_click(self):
            print("\n")
            for currentQTableWidgetItem in self.tableWidget.selectedItems():
                print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
    


    def set_max_scrollbar(self, maximum):
        self.vidScroll.setMaximum(maximum)

    def set_scrollbar(self, value):
        self.vidScroll.setValue(value)
    
    def slider_update(self, value, func=None):
        self.scrollbar_changed = True
        seconds = (value/self.vid_fps) %60
        minutes = int(((value/self.vid_fps)/60)%60)
        hours = int(minutes/60)
        self.scrollframe.setText( str(hours) + ":"+ str(minutes) + ":" + str(round( ((value/self.vid_fps) %60),2 ) ))
    


    def get_scrollbar_value(self):
        # print(self.vidScroll.value())
        return self.vidScroll.value()

    def mediaStateChanged(self, state):
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

    def export_tab_pressed(self):
        self.export_state = True
    
    def set_fps_info(self, fps):
        self.vid_fps = fps

    def get_frame_skip(self):
        try:
            skip = int(self.skip_frames.text())
        except:
            self.log("Skip value non-valid. Please enter a number.")
            skip = 50
        # #ensure skip is not backwards?
        if skip == 0:
            skip = 50
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
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Help")
        msg.setText("1) \n2)  \n3)  \n4)  \n")
        
        x = msg.exec_()  # this will show our messagebox

    def log(self, text):
        self.log_label.setText("Info: " + str(text))
        print(text)

    def set_all_tabs(self, value):
        for tab in self.tab_list:
            if value == "Active":
                # tab.active = True
                tab.active_button.setChecked(True)
            elif value == "Inactive":
                # tab.active = False
                tab.active_button.setChecked(False)
            elif value == "Read":
                # tab.read_only = True
                tab.read_only_button.setChecked(True)
            elif value == "Write":
                # tab.read_only = False
                tab.read_only_button.setChecked(False)




class person_tab():
    def __init__(self, window):
        self.parent = window
        self.tab = QWidget()
        

        #these are used to record metadata
        self.name_line = QLineEdit(window)
        # self.name_line.textChanged.connect(self.update_tab_name)
        self.sex_line = QLineEdit(window)
        self.desc_line = QPlainTextEdit(window)
        
        self.length_tracked = QLabel(window)
        self.length_tracked.setText("00:00")
    
        self.active = True
        self.read_only = False
        self.beginning = False
        self.init_tab(window)

    def init_tab(self, parent_window):
        self.tab.layout = QVBoxLayout(parent_window)

        #Start Name
        name_layout = QHBoxLayout()
        
        name_btn = QPushButton('Name', parent_window)
        name_btn.clicked.connect(lambda: self.getText(input_name="Name:",line=self.name_line))

        name_layout.addWidget(name_btn)
        name_layout.addWidget(self.name_line)
        self.tab.layout.addLayout(name_layout)

        #Start Sex
        sex_layout = QHBoxLayout()

        sex_btn = QPushButton('Sex', parent_window)
        sex_btn.clicked.connect(self.getChoice)
        
        name_layout.addWidget(sex_btn)
        name_layout.addWidget(self.sex_line)

        #Start Description
        desc_layout = QHBoxLayout()

        desc_btn = QPushButton('Description', parent_window)
        desc_btn.clicked.connect(lambda: self.getText(input_name="Description:",line=self.desc_line))
        
        desc_layout.addWidget(desc_btn)
        desc_layout.addWidget(self.desc_line)
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
        self.tab.layout.addLayout(length_layout)

        self.read_only_button = QCheckBox("Beginning")
        self.read_only_button.setChecked(False)
        self.read_only_button.stateChanged.connect(lambda:self.toggle_beginning())
        self.read_only_button.setToolTip("Sets the 'present at beginning' to be True or False for this person.")
        length_layout.addWidget(self.read_only_button)
        self.tab.layout.addLayout(length_layout)

        self.tab.setLayout(self.tab.layout)

        # #inital load of variables
        # self.getText(input_name="Name:",line=self.name_line)
        # self.getChoice()
        # self.getText(input_name="Description:",line=self.desc_line)
        


    # def getInteger(self):
    #     i, okPressed = QInputDialog.getInt(self.parent, "Get integer","Percentage:", 28, 0, 100, 1)
    #     if okPressed:

    #         # return i

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

    def update_length_tracked(self, time):
        self.length_tracked.setText("00:00")
        seconds = round((time)%60,2)
        minutes = int(((time)/60)%60)
        self.length_tracked.setText( str(minutes) + ":" + str(seconds))
    
    # def update_tab_name(self):
    #     self.tab.parentWidget().setTabText(self.tab.parent.currentIndex(),self.name_line.getText())
    #     print(self.tab.parentWidget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
