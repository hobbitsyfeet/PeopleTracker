import PyQt5

import uuid

class person_tab():
    def __init__(self, window):
        self.id = uuid.uuid1()
        self.parent = window
        self.tab = PyQt5.QtWidgets.QWidget()
        

        #these are used to record metadata
        self.name_line = PyQt5.QtWidgets.QLineEdit(window)
        self.id_line = PyQt5.QtWidgets.QLineEdit(window)
        self.group_line = PyQt5.QtWidgets.QLineEdit(window)
        self.group_line.setValidator(PyQt5.QtGui.QIntValidator(0,999))
        # self.name_line.textChanged.connect(self.update_tab_name)
        self.sex_line = PyQt5.QtWidgets.QLineEdit(window)
        self.desc_line = PyQt5.QtWidgets.QPlainTextEdit(window)
        
        self.length_tracked = PyQt5.QtWidgets.QLabel(window)
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
            self.tab.layout = PyQt5.QtWidgets.QVBoxLayout(parent_window)

            #Start Name
            name_layout = PyQt5.QtWidgets.QHBoxLayout()
            self.tab.layout.addLayout(name_layout)
            
            name_btn = PyQt5.QtWidgets.QPushButton('Name', parent_window)
            name_btn.clicked.connect(lambda: self.getText(input_name="Name:",line=self.name_line))

            name_layout.addWidget(name_btn)

            name_layout.addWidget(self.name_line)
            

            id_btn = PyQt5.QtWidgets.QPushButton('ID', parent_window)
            id_btn.clicked.connect(lambda: self.getText(input_name="ID:",line=self.id_line))

            name_layout.addWidget(id_btn)

            name_layout.addWidget(self.id_line)
            # #Start Sex
            # sex_layout = PyQt5.QtWidgets.QHBoxLayout()

            sex_btn = PyQt5.QtWidgets.QPushButton('Sex', parent_window)
            sex_btn.clicked.connect(self.getChoice)
            
            name_layout.addWidget(sex_btn)
            name_layout.addWidget(self.sex_line)

            group_size_btn = PyQt5.QtWidgets.QPushButton('Group Size', parent_window)
            group_size_btn.clicked.connect(lambda: self.getInteger())

            name_layout.addWidget(group_size_btn)
            name_layout.addWidget(self.group_line)

            #Start Description
            desc_layout = PyQt5.QtWidgets.QHBoxLayout()

            desc_btn = PyQt5.QtWidgets.QPushButton('Description', parent_window)
            desc_btn.clicked.connect(lambda: self.getText(input_name="Description:",line=self.desc_line))
            
            desc_layout.addWidget(desc_btn)
            desc_layout.addWidget(self.desc_line)

            self.image = PyQt5.QtGui.QPixmap()
            self.tab.layout.addLayout(desc_layout)
            
            #setup length of time tracked
            length_layout = PyQt5.QtWidgets.QHBoxLayout()
            length_label = PyQt5.QtWidgets.QLabel(parent_window)
            length_label.setText("Total Tracked (mm:ss): ")

            length_layout.addWidget(length_label)
            length_layout.addWidget(self.length_tracked)
            length_layout.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

            # self.active_button = PyQt5.QtWidgets.QCheckBox("Active")
            # self.active_button.setChecked(True)
            # self.active_button.stateChanged.connect(lambda:self.toggle_active())
            # self.active_button.setToolTip("Sets the current tracking to actively record. \nIf unchecked, no box will be processed, displayed or recorded.")
            # length_layout.addWidget(self.active_button)

            self.read_only_button = PyQt5.QtWidgets.QCheckBox("Read Only")
            self.read_only_button.setChecked(False)
            self.read_only_button.stateChanged.connect(lambda:self.toggle_read())
            self.read_only_button.setToolTip("Sets the person to read only. \nThis is useful for scrolling through the video without overwriting data.\n Also useful for people exiting the frame")
            length_layout.addWidget(self.read_only_button)

            self.other_room_button = PyQt5.QtWidgets.QCheckBox("Other Room")
            self.other_room_button.setChecked(False)
            self.other_room_button.stateChanged.connect(lambda:self.toggle_other_room())
            self.other_room_button.setToolTip("Sets the person to Other Room. \n This is useful for maintaining time  without location.")
            length_layout.addWidget(self.other_room_button)


            self.beginning_button = PyQt5.QtWidgets.QCheckBox("Beginning")
            self.beginning_button.setChecked(False)
            self.beginning_button.stateChanged.connect(lambda:self.toggle_beginning())
            self.beginning_button.setToolTip("Sets the 'present at beginning' to be True or False for this person.")
            length_layout.addWidget(self.beginning_button)
            self.tab.layout.addLayout(length_layout)

            self.is_region_button = PyQt5.QtWidgets.QCheckBox("Is Region")
            self.is_region_button.setChecked(False)
            self.is_region_button.stateChanged.connect(lambda:self.toggle_region())
            length_layout.addWidget(self.is_region_button)

            self.is_chair_button = PyQt5.QtWidgets.QCheckBox("Chair")
            self.is_chair_button.setChecked(False)
            self.is_chair_button.stateChanged.connect(lambda:self.toggle_chair())
            length_layout.addWidget(self.is_chair_button)


            self.tab.layout.addLayout(length_layout)


            self.tab.setLayout(self.tab.layout)
        
        except Exception as e:
            print(e)
            
            #  crashlogger.log(str(traceback.format_exc()))

        # #inital load of variables
        # self.getText(input_name="Name:",line=self.name_line)
        # self.getChoice()
        # self.getText(input_name="Description:",line=self.desc_line)
        


    def getInteger(self):
        i, okPressed = PyQt5.QtWidgets.QInputDialog.getInt(self.parent, "PyQt5.QtWidgets.QInputDialog().getInteger()",
                                 "Number:", 1, 0, 999, 1)
        self.group_line.setText(str(i))
        if okPressed:
            return i

    # def getDouble(self):
    #     d, okPressed = PyQt5.QtWidgets.QInputDialog.getDouble(self.parent, "Get double","Value:", 10.50, 0, 100, 10)
    #     if okPressed:
    #         return d
        
    def getChoice(self):
        items = ("Female","Male", "Other")
        item, okPressed = PyQt5.QtWidgets.QInputDialog.getItem(self.parent, "Get item","Sex:", items, 0, False)
        if okPressed and item:
            # print(item)
            self.sex_line.setText(item)
            # return item

    def getText(self, input_name, line):
        text, okPressed = PyQt5.QtWidgets.QInputDialog.getText(self.parent, "Get text", input_name, PyQt5.QtWidgets.QLineEdit.Normal, "")
        if okPressed and text != '':
            if type(line) == type(PyQt5.QtWidgets.QPlainTextEdit()):
                line.setPlainText(text)
            else:
                line.setText(text)
            # print(text)
            
            # return text

    def get_uuid(self):
        return self.id

    def get_beginning(self):
        return self.beginning
    
    def get_is_region(self):
        return self.is_region
    
    # def get_region_state(self):
    #     return self.region_state

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
        print("TOGGLING")
        self.parent.log("Setting Read only to " + str(not self.read_only))
        # self.read_only = False
        self.read_only = not self.read_only
        return self.read_only
    
    def toggle_beginning(self):
        self.parent.log("Setting person present at beginning to " + str(not self.beginning))
        self.beginning = not self.beginning
        return self.beginning

    def toggle_region(self):
        # self.region_state = True
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