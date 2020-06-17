# PES University -- Intel India Technovation Contest 2020
## 2nd Edition, Jan to May 2020
## CIE in collaboration with Department of CS and ECE

## Problem Statement
The purpose of the activity would be to enhance the vision of the driver with other sensors or smarter algorithms that can work in low light scenarios – dusk, night, in lights from headlights, etc. <br />
The solution would be able to detect common objects like cars, two-wheelers, auto-rickshaws, pedestrians, riders in the night/low-light conditions. The system can be developed to work in the visible spectrum, infra-red, or ultra-sound. 
Target performance for an ideal scenario would have the algorithm running at 30Hz.

## Object Detection: An Overview
Object Detection, localizing objects in image frames, has been the focus of the Deep Learning, and Computer vision fraternity for decades now.  It plays a crucial role in the autonomous industry and the algorithms here can be used for Image classification, which indeed can be used almost everywhere. <br />
The major problem faced during Object Detection using state of the art models is the data available works well for the test images but ground truth images can have outliers and the image context can change which can adversely affect the output.<br />
Here are a few examples where the object detection algorithm fails:<br />
<img src="images/Failure1.jpg" width="300"><br />
If you are using object detection algorithm or even using any segmentation model in autonomous car and a person wearing such an adversial t-shirt, it could cause a crash.<br />
<img src="images/Failure2.png" width="300"><br />
Misunderstanding the traffic signboard on the road could result in a ticket for a semi-automated/automated car which is not a desirable feature.<br />
The generalization of object detectors is the key to the success of the autonomous industry and in this project. Here, in this project, we try to make the object detection in low light conditions/ twilight and propose a network that could help in the generalization of object detectors.




## Previous Works
### VGG-16
### R-CNN
### ResNet
The ResNet that stands for Residual Neural Network is an Artificial Neural Network that alleviates the problems present with the Deep Network. This is done by implementing a new neural network layer - <b>The Residual Block.</b><br /> Following is the image representing Residual Block: a building block.<br />
<img src="images/image_1.png" width="300"><br />
ResNet solves many problems but one of the major issues solved with this was the <b>vanishing gradient</b> in the deep neural network which was caused due to very deep network and thus making them hard to train. This further results in the shrinking of the gradient and the weights not being updated. Hence, ResNet utilizes <i>skip connection</i> so that the gradient can directly be propagated backward from later layers to initial filters.<br/>
In skip connection the convolution layers are stacked one over the other, same way as before but now we also add the original input to the output of the later convolution block.

Here is the architecture of ResNet model that explains it in a better way.<br />
<img src="images/Resnet_Architecture.png"><br />

For best results, it is recommended that the activation function is applied after the skip connection.
### Densenet
### Squeezenet
### MobileNet
### YOLOv3

## Datasets
### COCO Dataset
The majority of the state of the art, modern object detection, and segmentation models are trained and tested on this Common Objects in Context Dataset, often referred to as the COCO dataset. For our problem statement, we need to detect Common Objects on roads like Car, Person, Motorcycle, Animals, Truck, Train, etc. that are available in this dataset. This dataset has a high number of images to the number of the class ratio that helps in training. The modern architectures trained on this dataset can detect the objects in low light conditions but only with deep architectures, also the complexity of the model architecture is high because of the higher number of redundant classes. 
### Indian Driving Dataset
### BDD Dataset

## Innovation Component
### Cycle-GANs

## Our Approach
### YOLO implementation
### Comparison of different Models

## Results
### Cycle GAN
### YOLO implementation
### Comparitive Study Results

## Credits
This project was completed by a team of students from PES University, Kashish Oberoi, Shobhit Tuli, Anuvrat Singhal, and Anushk Misra 
under the Guidance of Prof. Veena S., Department of ECE, PES University.

