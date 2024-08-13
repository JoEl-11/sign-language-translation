This ml project aim to translate sign language into speech this readme file explains the working and what each files and directories contains.

>generating_dataset:- this python file is used for generating the dataset for training the model.we used mediapipe for that,instead of training the model with
                      images of handsigns we plotted landmarks of hands using device camera and then plotted that on a black image,then stored it.

>dataforprocessing,dataimage:- these two folders contains the dataset for training the model.Each alphabets have 1000 images each.

>implementation:- this folder conatin two implementations of the model where the 
                  realtime prediction predict the symbols realtime.
                  also there is a simple flask implementation of the model which captures image using webrtc and then send it to flask using fetch api then returns 
                  the prediction result and displays it.

note: This Flask implementation is provided as a basic sample and is not intended to be user-friendly or fully optimized. It is included to demonstrate the various steps and work done during the development of this project.

>model_creation:- this python file creates a model and then saves it.

>video_to_speech:- This file loads the pre-trained model and accepts a video input with dimensions of 640x480 pixels, matching the dimensions of the images used during training. The video input should be structured such that each sign lasts for a duration of 5 seconds. For example, if the word to be translated is 'hai,' the sign for 'h' should last for 5 seconds, resulting in a total video length of 15 seconds.

The program splits the video into 5-second clips, then iterates through each clip to classify each frame. The majority class is assigned as the class for the clip, and the results are accumulated into a string. Finally, the program converts the string into an MP3 file using the Google Text-to-Speech module.

The clip duration can be customized by modifying the video_splitting function definition.

   