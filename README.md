Dependencies:
  Exiftool: https://exiftool.org/
    - This is for extracting metadata from files.
    
  -  Please Note: Pyexiftool has a bug on windows 10, python 3.6 and need to replace exiftool.py with the following pull request:
   https://github.com/smarnach/pyexiftool/commit/8738ae963afc784fcef76de6bcebf277a58379ab

Ref: https://github.com/smarnach/pyexiftool/issues/26


If Opencv-python-contrib does not work, try:

    pip install opencv-contrib-python-headless
  
(Date: January 21, 2020)
