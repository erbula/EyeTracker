# EyeTracker
Demo. The idea of EyeTracker is to use cascades to detect a face from a video, and observe both eyes, within the detected face, separately to predict whether they are open or closed. The prediction is done by using a trained CNN model.
This is obviously a stripped down version, it does not include neither a pre-trained model nor pre-collected data. Despite being very pretty, at least according to my mom, I thought it might be better not to import thousands of images of my star-like eyes into this repository. User has to collect the data by themselves (tool included) and to create + train the model as well.

## Contents:
### eye_tracker.py
This is the main application. Define the correct model (load model) to use and launch.

### gather_data.py
Use this to gather data for the CNN model. This tool starts a webcam stream, detects faces from the stream / frames and grabs cropped images from the approximated eye locations. Approximated eye locations (top left and right corners with a small offset in Y direction) are used for simplicity reasons, feel free to modify the Y offset. After launching, the user can select which eyes (left, right, both) to capture, and what is the state of the eye (open, closed). Every frame is cropped and stored. All files are stored in './data/train' directory. Use model.py to split the data according to defined split ratios.

### model.py
Provides functionalities to split the data to validation, test and train data sets, and to build, train and evaluate the CNN model. User has to specify the CNN architecture (contents of create_model()).

### utils.py
Some additional helper functions.
