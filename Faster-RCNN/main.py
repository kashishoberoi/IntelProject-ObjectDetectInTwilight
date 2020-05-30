import os
import json
import pandas as pd
import json
import torchvision.transforms as T
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import cv2 
from PIL import Image
import time
from skimage import data, io
import torchvision
import time



def get_prediction(img, model,COCO_INSTANCE_CATEGORY_NAMES,transform ,threshold):
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  if(len(pred[0]['boxes'])) == 0:
    return [],[],0
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class,1

def object_detection_api(img, model,COCO_INSTANCE_CATEGORY_NAMES,transform ,threshold=0.5, rect_th=3, text_size=3, text_th=3):
  boxes, pred_cls,flag = get_prediction(img, model,COCO_INSTANCE_CATEGORY_NAMES,transform ,threshold) # Get predictions
  if flag:
      for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  return img


if __name__ == "__main__":
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	model.eval()
	COCO_INSTANCE_CATEGORY_NAMES = [
		'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
		'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
		'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
		'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
		'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
		'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
		'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
		'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
		'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
	]
	transform = T.Compose([T.ToTensor()])
	cap = cv2.VideoCapture(0)
	number_of_frames = 0
	start = time.time()
	while(True):
        # Capture frame-by-frame
		ret, frame = cap.read()
        # Our operations on the frame come here
		frame = object_detection_api(frame,model,COCO_INSTANCE_CATEGORY_NAMES,transform,threshold=0.8)
        # Display the resulting frame
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		number_of_frames = number_of_frames+ 1
    # When everything done, release the capture
	end = time.time()
	print("FPS: ",number_of_frames/(end-start))
	cap.release()
	cv2.destroyAllWindows()