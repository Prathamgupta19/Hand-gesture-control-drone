# Hand-Gesture-Drone-Navigation

This repo consists of the model --> model.hdf5 along with the training script to train the model on hand landmarks extracted usingn mediapipe framework.
There are 42 hand landmarks (x, y) which have been collected to train a custom keras ANN model to accurately classify hand gestures for drone navigation. 
The model's accuracy is 99%. 

The commented_tello.py file consists of script that controls a DJI Tello drone but this model and architecture can be deployed on any drone using ROS.
Just make sure you use the same video frame processing logic before feeding it into the model for classification. 
I'm providing the link to the mediapipe documentation for the hand landmarks solution as it has been updated. (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#get_started)

Enjoy!!
