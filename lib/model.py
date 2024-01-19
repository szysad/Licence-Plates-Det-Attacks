import yolov5
from torch import device
from typing import Optional


CONF = 0.25  # NMS confidence threshold
IOU = 0.45  # NMS IoU threshold
AGNOSTIC = False  # NMS class-agnostic
MULTI_LABEL = False  # NMS multiple labels per box
MAX_DETECTIONS = 1000  # maximum number of detections per image


def get_model(half: bool, device: Optional[device] = None):
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
