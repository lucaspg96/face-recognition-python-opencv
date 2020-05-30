# Face recognition with Python and OpenCV

This tutorial code is inspired by Ashish at his [Youtube Live](https://www.youtube.com/watch?v=ukL_UjrqZFw).

## Running

First things first: install the OpenCV lib:`pip install opencv-contrib-python`.

Then, for each person that you want to recognize, run the script `data_generator.py`, changing the **face_id** variable to person id (try to use ids from 0 to *n*, being *n* the number of people). This script will generate 100 pictures for each person.

Once the data is generated, just run the `training.py` script and wait until the model is generated and saved.

Then, at `run_live_video.py`, the dictionary **names** is the mapping id -> person's name. Edit it according the people names and see the magic (it's actually science
) happening!

**WARNING:** to finish the live mode execution, press **q** in the webcam screen. If you just close it, it may open again or block the program.