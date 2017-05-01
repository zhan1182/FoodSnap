# FoodSnap

FoodSnap is an Android mobile application for Chinese food recognition. The system only requires users to upload their target image to be recognized, then it would resize the image and generate a numpy array and finally output the name and description of the food. According to the performance in both test data and real-life experiment, the recognition is satisfying. 

## Description

* chinese-food-detector-frontend: App frontend.
* chinese_food_detector_backend: App backend.
* data: Collected images.
* Intro.txt: Description of each type of dishes.
* setting.py: Set up the directory.
* smallcnn.py: Train and evaluate a small CNN.
* vgg16.py: Train and evaluate VGG16 model.
* inceptionv3.py: Train and evaluate InceptionV3 model.
* wrapper.py: A socket server for the VGG16 model. 
* wrapper_test.py: To test the socket server.


