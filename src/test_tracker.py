from multitracker import MultiTracker
from qt_dialog import person_tab
from PyQt5 import QtWidgets
import sys
import unittest
import logging

class TestMultitracker(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMultitracker, self).__init__(*args, **kwargs)
        self.longMessage = True

    def new_tracker(self):
        app = QtWidgets.QApplication(sys.argv)
        tracker = MultiTracker(person_tab(QtWidgets.QWidget()))
        self.assertNotEqual(tracker, None, "Should not be None")
        return tracker
    
    def test_merge_intervals(self):

        logging.info("Testing Multitracker.merge_intervals")
        tracker = self.new_tracker()
        
        interval = [[0,20], [10,50]]
        merge_result = tracker.merge_intervals(interval)
        answer =  [[0,50]]
        string = (str(interval) + "And", str(answer) + "Should be equal")
        self.assertEqual(merge_result, answer, msg=string)

        interval = tracker.merge_intervals([[0, 500]])
        answer = [[0, 500]]
        string = (str(interval) + "And", str(answer) + "Should be equal")
        self.assertEqual(interval, answer, msg=string)

        interval = tracker.merge_intervals([[0, 0]])
        answer = [[0, 0]]
        string = (str(interval) + "And", str(answer) + "Should be equal")
        self.assertEqual(interval, answer, msg=string)
        

        interval = tracker.merge_intervals([[0]])
        answer = [[0, 0]]
        string = (str(interval) + "And", str(answer) + "Should be equal")
        self.assertEqual(interval, answer, msg=string)


    def test_part_time_into_segments_Empty(self):
        logging.info("Testing Multitracker.part_time_into_segments")

        tracker = self.new_tracker()

        self.assertIsNot(tracker.part_time_to_segments([10]), None)

        segments = []
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        answer = [[0]]
        self.assertEqual(part_time, answer)

    def test_part_time_into_segments_single(self):
        tracker = self.new_tracker()

        segments = [0]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        answer = [[0]]
        self.assertEqual(part_time, answer)
    
    def test_part_time_into_segments_multiple(self):
        tracker = self.new_tracker()

        #Test Single segment
        segments = [0, 200]
        answer = [[0, 200]]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        string = (str(segments) + " And ", str(answer) + " Should be equal")
        self.assertEqual(part_time, answer, msg=string)

        # Test segment out of range
        segments = [0,500]  
        answer = [[0,0], [500,500]]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)

        #Test multiple larger segments within range
        segments = [0, 200, 300]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        answer = [[0,300]]
        self.assertEqual(part_time, answer)

    def test_part_into_segments_complex(self):
        tracker = self.new_tracker()

        #Test complex segments
        segments = [0, 10, 20, 30, 100, 200, 300, 400, 1000, 1010, 1020]
        answer = [[0, 400], [1000, 1020]]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        string = (str(segments) + " And ", str(answer) + " Should be equal")
        self.assertEqual(part_time, answer, msg=string)

        # Test Unsorted Segments equal to the one above
        segments = [200, 0, 30, 1010, 10, 20, 100, 300, 400, 1000, 1020]
        answer = [[0, 400], [1000, 1020]]
        part_time = tracker.part_time_to_segments(segments, segment_size=300)
        string = (str(segments) + " And ", str(answer) + " Should be equal")
        self.assertEqual(part_time, answer, msg=string)

    def test_calculate_time(self):
        tracker = self.new_tracker()

        calculated_time = tracker.calculate_time(0,0,fps=30)
        answer = 0
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_time(0,30,fps=30)
        answer = 1
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_time(0,60,fps=30)
        answer = 2
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_time(0,15,fps=30)
        answer = 0.5
        self.assertEqual(calculated_time, answer)

    def test_calculate_total_time(self):
        tracker = self.new_tracker()
   
        self.assertRaises(Exception, tracker.calculate_time, None, None)

        calculated_time = tracker.calculate_time(0, 30, fps=30)
        answer = 1
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_time(1000, 1030, fps=30)
        answer = 1
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_time(0, 300, fps=300)
        answer = 1
        self.assertEqual(calculated_time, answer)
        
        tracker.calculate_total_time(tracker.part_time_to_segments([0, 30]), fps=30)
        answer = 1
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_total_time(tracker.part_time_to_segments([0, 300]), fps=30)
        answer = 10
        self.assertEqual(calculated_time, answer)

        calculated_time = tracker.calculate_total_time(tracker.part_time_to_segments([]), fps=30)
        answer = 0
        self.assertEqual(calculated_time, answer)

if __name__ == '__main__':
    unittest.main()
