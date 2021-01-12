# EyeTracker
Demo. The idea of EyeTracker is to use cascades to detect a face from a video, and observer both eyes separately to determine whether they are open or closed.
This is obviously a stripped down version, it does not include neither pre-trained model nor pre-collected data. User has to collect the data by themselves (tool included) and to create + train the model.

## Contents:
### eye_tracker.py
This is the main application. Define the correct model (load model) to use and launch.

### gather_data.py
Use this to gather data for the CNN model. This tool starts a webcam stream, detects faces from the stream and grabs cropped images from the approximated eye locations. Approximated eye locations (top left and right corners with small offset in Y direction) are used for simplicity reasons, feel free to modify the Y offset. After launching, the user can select which eyes (left, right, both) to capture, and what is the state of the eye (open, closed). Every frame is All files are stored in './data/train' directory. Use model.py to split the data according to defined split ratios.

### model.py
Provides functionalities to split the data to validation, test and train data sets, and to build, train and evaluate the CNN model. User has to specify the CNN architecture ('create_model' method).

### utils.py
Some additional helper functions.
 
