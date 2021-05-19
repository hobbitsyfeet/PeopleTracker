from multitracker import Regions
from qt_dialog import person_tab
from PyQt5 import QtWidgets
import sys
import unittest
import logging

class TestRegions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRegions, self).__init__(*args, **kwargs)
        self.longMessage = True

    def new_region(self):
        #Requires app before regions can be declared
        app = QtWidgets.QApplication(sys.argv)
        region = Regions()
        self.assertNotEqual(region, None, "Should not be None")
        return region

    def test_set_moving_region(self):
        regions = self.new_region()

        #Test Null Entry or Invalid
        regions.set_moving_region(name="Joe", point=(0,0), dimensions=(0,0), id = "1")
        regions.set_moving_region(name="Joe", point=(0,0), dimensions=(-1,-1),id = "2")
        regions.set_moving_region(name="Joe", point=(0,0), dimensions=(10,10), id = "3")

        # regions.set_moving_region


    def del_moving_region(self):
        regions = self.new_region()

        # Test Removal of name
        regions.set_moving_region(name="Test", point=(0,0), dimensions=(0,0), id="1")
        regions.del_moving_region(name="Test")
        self.assertEqual(regions.region_dict, dict())

        # Test does not remove subset
        regions.set_moving_region(name="Test", point=(0,0), dimensions=(0,0), id="1")
        regions.del_moving_region(name="Tes")
        
        self.assertEqual(regions.region_dict.keys(), dict("1").keys())
        
        # Test does not remove superset
        regions.set_moving_region(name="Test", point=(0,0), dimensions=(0,0), id="1")
        regions.del_moving_region(name="Tests")
        self.assertEqual(regions.region_dict.keys(), dict("1").keys())


    def test_test_region(self):
        """
            if p <= 1: #point exists in or on ellipse
            Otherwise the point exists outside of the ellipse

            Note: Eclipse Center = (PointX + 1/2 W, PointY + H/2) because (x,y) describe top left corner of a bounding box, and (x + width, y + height) describe the bottom right
        """
        regions = self.new_region()

        # Test on Empty
        regions.set_moving_region(name="Test", point=(0,0), dimensions=(0,0), id="1")

        # Zero Area cube has no center, therefore should not be in eclipse
        self.assertEqual(regions.test_region((0,0))[0], [])
        self.assertGreaterEqual(regions.test_region((0,0))[1], 1)
        
        # Point outside of zero area eclispe should not be in or on the eclipse
        self.assertEqual(regions.test_region((10,10))[0], [])
        self.assertGreaterEqual(regions.test_region((10,10))[1], 1)

        # Test Invalid
        regions.set_moving_region(name="Test", point=(-1,-1), dimensions=(-1,-1), id="1")

        self.assertEqual(regions.test_region((0,0))[0], [])
        self.assertGreaterEqual(regions.test_region((0,0))[1], 1)

        self.assertEqual(regions.test_region((-10,-10))[0], [])
        self.assertGreaterEqual(regions.test_region((-10,-10))[1], 1)

        # Test Valid
        regions.set_moving_region(name="Test", point=(100,100), dimensions=(5,10),id="1")
        self.assertGreaterEqual(regions.test_region((0,0))[1], 1)
        self.assertGreaterEqual(regions.test_region((100,100))[1], 1) # The top left of a rectangle which describes an eclipse is not included in ellipse

        self.assertLessEqual(regions.test_region((102.5,100))[1], 1) # Tangent Edge should exist within ellipse (top)
        self.assertLessEqual(regions.test_region((102.5,110))[1], 1) # Tangent Edge should exist within ellipse (bottom)
        self.assertLessEqual(regions.test_region((100,105))[1], 1) # Tangent Edge should exist within ellipse (left)
        self.assertLessEqual(regions.test_region((105,105))[1], 1) # Tangent Edge should exist within ellipse (right)  
        # self.assertEqual(regions.test_region((,-10)), -1)

if __name__ == '__main__':
    unittest.main()
