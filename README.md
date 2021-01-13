# EyeTracker
Simple demo. The idea of EyeTracker is to use a face cascade to detect a face from a video, and then to observe both eyes, within the detected face, separately to predict whether they are open or closed. The prediction is done by using a trained DNN (CNN) model.
This is obviously a stripped down version, it does not include neither a pre-trained model nor pre-collected data. Despite being very pretty, at least according to my mom, I thought it might be better not to import thousands of images of my star-like eyes into this repository. User has to collect the data by themselves (tool included) and to create + train the model as well.

For cascades, see for example: https://github.com/opencv/opencv/tree/master/data/haarcascades

## Contents:
### eye_tracker.py
This is the main application. Define the correct model (load model), a path to the face cascade file and the input shape for the model and you're ready to go.

### gather_data.py
Use this to gather data for the DNN model. This tool starts a webcam stream, detects faces from the stream / frames and grabs cropped images from the approximated eye locations. Approximated eye locations (top left and right corners with a small offset in Y direction) are used for simplicity reasons, the Y offset is obviously modifiable. After launching, the user can select which eyes (left, right, both) to capture, and what is the state of the eye (open, closed). Data collenction starts after the countdown, press Esc to stop collecting. Every frame is cropped and saved. All files (images of both eyes) are stored in './data/train' directory. Use model.py to split the data according to defined split ratios.

### model.py
Provides functionalities to split the data to validation, test and train sets, and to build, train and evaluate the model. User has to specify the model architecture (contents of create_model()).

### utils.py
Some additional functions.
