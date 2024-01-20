import yolov5
import torch
import numpy as np
from typing import Optional
import cv2
import matplotlib.pyplot as plt


CONF = 0.25  # NMS confidence threshold
IOU = 0.45  # NMS IoU threshold
AGNOSTIC = False  # NMS class-agnostic
MULTI_LABEL = False  # NMS multiple labels per box
MAX_DETECTIONS = 1000  # maximum number of detections per image


def get_model(half: bool, device: Optional[torch.device] = None):
    model = yolov5.load('keremberke/yolov5m-license-plate', device=device).eval()

    # set nms config
    model.conf = CONF
    model.iou = IOU
    model.agnostic = AGNOSTIC
    model.multi_label = MULTI_LABEL
    model.max_det = MAX_DETECTIONS

    # set half precision config
    if half:
        model.half()
    return model

def plot_detections(img: np.ndarray, pred: torch.Tensor):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    img = (img * 255).astype(np.uint8).copy()
    d = [0, 1, 0, 1]
    for i in range(pred.shape[0]):
        x1, y1, x2, y2 = tuple(pred[i][:4].cpu().numpy().astype(int))
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=1)
    plt.imshow(img)
