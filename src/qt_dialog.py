import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QPlainTextEdit, QSlider, QStyle, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QMessageBox

from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot

from math import trunc

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Person'
        self.left = 250
        self.top = 250
        self.width = 640
        self.height = 480
        self.initUI()

        self.play_state = False
        self.export_state = False
        self.scrollbar_changed = False
        self.vid_fps = 30
        
    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.layout = QVBoxLayout(self)

        self.tab_control_layout = QHBoxLayout()
        self.add_tab_btn = QPushButton()
        self.add_tab_btn.setText("Add Tab")
        self.add_tab_btn.clicked.connect(self.add_tab)
        self.tab_control_layout.addWidget(self.add_tab_btn)

        self.add_tab_btn.setEnabled(False)

        self.export_tab_btn = QPushButton()
        self.export_tab_btn.setText("Export Data")
        self.tab_control_layout.addWidget(self.export_tab_btn)
        self.export_tab_btn.clicked.connect(self.export_tab_pressed)

        self.del_tab_state = False

        self.del_tab_btn = QPushButton()
        self.del_tab_btn.setText("Delete Tab")
        self.del_tab_btn.clicked.connect(self.remove_tab)
        
        self.tab_control_layout.addWidget(self.del_tab_btn)
        
        self.del_tab_btn.setEnabled(False)

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
        self.skip_frames.setText("1")
        self.skip_frames.setFixedWidth(25)
        self.skip_frames.setAlignment(Qt.AlignRight)
        bottom_layout.addWidget(self.skip_frames)

        self.skip_label = QLabel(self)
        self.skip_label.setText("Frame Skip")
        self.skip_label.setAlignment(Qt.AlignLeft)
        self.skip_label.setFixedHeight(12)
        bottom_layout.addWidget(self.skip_label)
        
        
        


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
        print(self.vidScroll.value())
        return self.vidScroll.value()

    def mediaStateChanged(self, state):
        if self.play_state == False:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_state = True
            print(self.play_state)
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_state = False
            print(self.play_state)

    def set_tab_names(self):
        i = 0
        for tab in self.tab_list:
            if tab.name_line.text() == "":
                self.tabs.setTabText(i,"Person " + str(i + 1))
            else:
                self.tabs.setTabText(i,tab.name_line.text())
            i += 1

    def get_current_tab_info(self):
        current_tab = self.tab_list[self.tabs.currentIndex()]
        name = current_tab.name_line.getText()
        sex = current_tab.sex_line.getText()
        desc = current_tab.desc_line.getText()
        time = current_tab.getText()
        return name, sex, desc, time

    def add_tab(self):
        self.tab_list.append(person_tab(self))
        self.tabs.addTab(self.tab_list[-1].tab, ("Person " + str(self.tabs.count())))
        
    
    def remove_tab(self):
        self.del_tab_state = True
        self.tabs.removeTab(self.tabs.currentIndex())
        del self.tab_list[self.tabs.currentIndex()]

    def export_tab_pressed(self):
        self.export_state = True
    
    def set_fps_info(self, fps):
        self.vid_fps = fps

    def get_frame_skip(self):
        try:
            skip = int(self.skip_frames.text())
        except:
            print("Skip value non-valid. Please enter a number.")
            skip = 0
        # #ensure skip is not backwards?
        # if skip < 1:
        #     skip = 1
        return skip

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
        self.tab.layout.addLayout(length_layout)

        self.tab.setLayout(self.tab.layout)

        # #inital load of variables
        # self.getText(input_name="Name:",line=self.name_line)
        # self.getChoice()
        # self.getText(input_name="Description:",line=self.desc_line)
        


    def getInteger(self):
        i, okPressed = QInputDialog.getInt(self.parent, "Get integer","Percentage:", 28, 0, 100, 1)
        if okPressed:
            print(i)
            # return i

    def getDouble(self):
        d, okPressed = QInputDialog.getDouble(self.parent, "Get double","Value:", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
            # return d
        
    def getChoice(self):
        items = ("Female","Male", "Other")
        item, okPressed = QInputDialog.getItem(self.parent, "Get item","Sex:", items, 0, False)
        if okPressed and item:
            print(item)
            self.sex_line.setText(item)
            # return item

    def getText(self, input_name, line):
        text, okPressed = QInputDialog.getText(self.parent, "Get text", input_name, QLineEdit.Normal, "")
        if okPressed and text != '':
            if type(line) == type(QPlainTextEdit()):
                line.setPlainText(text)
            else:
                line.setText(text)
            print(text)
            
            # return text

    def update_length_tracked(self, time):
        self.length_tracked.setText("00:00")
    
    # def update_tab_name(self):
    #     self.tab.parentWidget().setTabText(self.tab.parent.currentIndex(),self.name_line.getText())
    #     print(self.tab.parentWidget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
