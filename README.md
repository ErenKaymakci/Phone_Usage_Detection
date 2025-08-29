# Assignment Phone Usage Detection

<a href='https://www.python.org/' target="_blank"><img alt='Python' src='https://img.shields.io/badge/Made_With Python-100000?style=for-the-badge&logo=Python&logoColor=white&labelColor=3774A7&color=FFD445'/></a>

## Approach
Object detection and human pose estimation combined to recognize phone usage.  
Instead of checking or comparing with only bounding boxes like in below:

![](https://machinelearningspace.com/wp-content/uploads/2023/01/IOU7-1024x626.jpg)

I preferred to add human pose-estimation to the system. Therefore, system check the keypoint of right and left hand which is 4 and 7 in image 

![](https://nanonets.com/blog/content/images/2019/04/Screen-Shot-2019-04-11-at-5.17.56-PM.png)

Afterwards the distance between hands and phone center can be calculated. 

<p align="center">
  <img src="./image.png"/>
</p>

## Improvement Points 
I have used pretrained YOLOv12 (phone detection) and YOLOv11 (pose) model which is trained with COCO dataset due to my limited time.
It needs optimization. Therefore, custom trained model can be used especially for cell phone detection. 
More than thousands phones can be labelled from similar scenes and trained on YOLOv12 or RF-DETR that are state-of-art models on object detection.
Computational performance also can be increased with optimization methods such as TensorRT or ONNX. 

Note: this is not the final status of project, it will be updated.
