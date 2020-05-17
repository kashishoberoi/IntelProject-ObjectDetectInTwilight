from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
from PIL import Image
import imutils

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    frame_count = 0
    cap = cv2.VideoCapture(0)
    time_algo = []
    start_time = time.time()
    while(True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = opt.img_size
        height = opt.img_size
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        frame_copy = frame
        frame = torch.from_numpy(frame.transpose(2, 0, 1))
        frame = frame.unsqueeze(0).float()
        _, _, h, w = frame.size()
        ih, iw = (opt.img_size, opt.img_size)
        dim_diff = np.abs(h - w)
        pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
        pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
        x = F.pad(frame, pad=pad, mode='constant', value=127.5) / 255.0
        frame = F.upsample(x, size=(ih, iw), mode='bilinear') 
        frame = Variable(frame.type(Tensor))
        start_time_algo = time.time()
        with torch.no_grad():
            detections = model(frame)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        frame = frame_copy
        end_time_algo = time.time()
        time_algo.append(end_time_algo-start_time_algo)
        if not(detections is None):
            # Bounding-box colors
            cmap = plt.get_cmap("tab20b")
            colors = [cmap(i) for i in np.linspace(0, 1, 20)]
            detections = rescale_boxes(detections[0], opt.img_size, (opt.img_size,opt.img_size))
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if cls_conf> opt.conf_thres and conf> opt.conf_thres:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    start_point = (int(x1),int(y1))
                    end_point = (int(x2),int(y2))
                    frame = cv2.rectangle(frame, start_point, end_point, color, 1) 
                    frame = cv2.putText(frame,classes[int(cls_pred)], start_point, cv2.FONT_HERSHEY_SIMPLEX,1, color, 1, cv2.LINE_AA)
        cv2.imshow('output',frame)
        frame_count = frame_count+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end_time = time.time()
    time = end_time - start_time
    time_algo = sum(time_algo)
    FPS = frame_count/time
    FPS_algo = frame_count/time_algo
    print("Frame Rate:", FPS)
    print("Frame Rate of algorithm:",FPS_algo)
    cap.release()
    cv2.destroyAllWindows()