from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QApplication, QComboBox, QInputDialog, QLineEdit,
                             QMessageBox, QWidget,QSplashScreen)

import uuid

import cv2
import math
class Regions(QWidget):

    def __init__(self, log = None):
        super().__init__()
        self.region_dict = dict()
        self.log = log #logs the information

    def add_region(self, frame):
        """
        Creates an ellipse given a rectangle ROI.
        """
        name, okPressed = QInputDialog.getText(self, 'Region', 'Region Name:')
        if okPressed and name != '':
            if self.log is not None:
                self.log(name)
            key = uuid.uuid1()
            point_x, point_y, width, height = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            self.region_dict[key] = (name, point_x, point_y, width, height)

    def add_moving_region(self, name, point, dimensions, id=None):
        if name not in [self.region_dict.keys()]:

            if uuid is None:
                id = uuid.uuid1()

            self.region_dict[id] = (name, point[0], point[1], dimensions[0], dimensions[1])
        else:
            print("Radius already Exists")

    def set_moving_region(self, name, point, dimensions, id):
        """
         Creates and sets a region with a given name, and given dimensions
         Point (x, y) : (int, int)
         Dimensions (width, height), (int, int)
        """
        # if id in [self.region_dict.keys()]:
        self.region_dict[id] = (name, point[0], point[1], dimensions[0], dimensions[1])

    def del_region(self):
        items = (self.region_dict.values())
        name_list = []
        for item in items:
            name_list.append(item[0])
        # items = ("Red","Blue","Green")
        item, okPressed = QInputDialog.getItem(self, "Select Region","Delete Regions:", name_list, 0, False)
        if okPressed and item:
            for index, value in enumerate(items):
                print(value)
                if item in value:
                    if self.log is not None:
                        self.log(str(("Deleting " + str(item))))

                    # key = items.index(index)
                    # print(items)
                    key = list(self.region_dict.keys())[index]
                    print("KEY", key)
                    del self.region_dict[key]
                    return
        # name, okPressed = QInputDialog.getText(self, 'Region', 'Delete Region Name:')
        
    def del_moving_region(self, name, id=None):
        if id in self.region_dict:
            del self.region_dict[id]
        # combo_box = QComboBox(self)
        # for item in items:
        #     combo_box.addItem(item)
        # combo_box.move(50, 250)
        # combo_box.showPopup() 
        # selected = combo_box.activated[str]
                # creating a combo box widget 

  
        # adding action to the button 
        # button.pressed.connect(self.action) 
  

        
        # comboBox.activated[str].connect(lambda parameter_list: expression)
        # del self.region_dict[selected]
        # item, okPressed = QInputDialog.getItem(self.parent, "Get item","Region Name", items, 0, False)
        # if okPressed and item:
        #     input_dialog.log(item)
    
    def display_region(self, frame):
        """
        Displays all region created Radius on given frame
        """
        for key, region in self.region_dict.items():
            name = region[0]
            x, y, w, h = region[1], region[2], region[3], region[4]
            ellipse_center = (int(x + (w/2)) ,int( y + (h/2)))

            frame = cv2.ellipse(frame, ellipse_center, (int((w/2)),(int(h/2))), 0, 0,360, (0,255,0) )
            # cv2.ellipse(frame, box=w/2,color=(0,255,0))
            cv2.rectangle(frame, (x + int(w/2.1) , y - 1), (x + int(w/2.1) + 10 * (len(name)) , y - 15),(255,255,255),-1)
            cv2.putText(frame, name, (x + int(w/2.1) , y - 1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0),1)
        return frame

    def test_region(self, test_point):
        """
        Tests weather a point value exists within an eclipse
         p <= 1 exists in or on eclipse

        Equation: Given p = (x-h)^2/a^2 + (y-k)^2/b^2
            Where test_point is (x,y) and ellipse_center is (h,k), 
            radius_width, radius_height as a,b respectivly

        Returns list(regions), p
        """
        #overlapping areas may result in multiple True tests
        within_points = []
        test_x = test_point[0]
        test_y = test_point[1]
        p = float('inf')
        for key, region in self.region_dict.items():
            name = region[0]
            x, y, w, h = region[1], region[2], region[3], region[4]

            #Invalid inputs are areas with zero or less, return no region and invalid p value
            if w <= 0 or h <= 0:
                return [], float("inf")
            
            #handle if devisor == 0
            denom_x = math.pow((w/2), 2)
            denom_y = math.pow((h/2), 2)
            if denom_x == 0:
                denom_x = 1
            elif denom_y == 0:
                denom_y = 1
            ellipse_center = (x + (w/2) , y + (h/2))

                # checking the equation of
                # ellipse with the given point
            try:
                p = ((math.pow((test_x - ellipse_center[0]), 2) / denom_x) + 
                    (math.pow((test_y - ellipse_center[1]), 2) / denom_y))
            except ZeroDivisionError as zerodiverr:
                p = float("inf") # Does not count inside the eclipse

            if p <= 1: #point exists in or on eclipse
                within_points.append(name)
        return within_points, p

    def handle_inputs():
        pass