import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QPlainTextEdit, QSlider, QStyle, QAction, QTabWidget, QVBoxLayout, QHBoxLayout

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot

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
        
        
    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.setMovable(True)
        self.tab_list = []
        
        # Add tabs        
        # for i in range(5):
        self.add_tab()

        # Add tabs to widget
        self.layout.addWidget(self.tabs)

        # setup scrollbar for video
        self.scrollframe = QLabel(self)
        self.scrollframe.setText("00:00")

        vidScroll = QSlider(Qt.Horizontal,self)
        vidScroll.setMinimum(0)
        vidScroll.setFocusPolicy(Qt.NoFocus)

        #assign a 
        vidScroll.valueChanged.connect(self.slider_update)

        #setup play/pause buttons
        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.mediaStateChanged)


        media_layout = QHBoxLayout()
        media_layout.addWidget(self.playButton)
        media_layout.addWidget(vidScroll)
        media_layout.addWidget(self.scrollframe)

        self.layout.addLayout(media_layout)

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
        self.scrollframe.setText(str(value))

    def mediaStateChanged(self, state):
        if self.play_state == False:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_state = True
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_state = False

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

class person_tab():
    def __init__(self, window):
        self.parent = window
        self.tab = QWidget()
        

        #these are used to record metadata
        self.name_line = QLineEdit(window)
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
    

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
