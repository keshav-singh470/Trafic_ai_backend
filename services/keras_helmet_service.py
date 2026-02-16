import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from .base import BaseTrafficService

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c       = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score      

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

class KerasHelmetService(BaseTrafficService):
    def __init__(self, model_path="helmet_backend.h5"):
        # We don't use the base YOLO model here, so we skip super().__init__ detection model loading
        # but we can still initialize self.device if needed, though TF handles its own device
        self.model_path = model_path
        self.net_h, self.net_w = 416, 416
        self.obj_thresh = 0.5
        self.nms_thresh = 0.45
        self.anchors = [14,20, 21,41, 30,19, 32,147, 37,66, 52,203, 72,295, 110,384, 35311,38369]
        # Classes: ["License Plate", "Person Bike", "With Helmet", "Without Helmet"]
        self.labels = ["License Plate", "Person Bike", "With Helmet", "Without Helmet"]
        
        print(f"Loading Keras Helmet Model from {self.model_path}...")
        try:
            self.model = load_model(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_input(self, image):
        new_h, new_w, _ = image.shape
        # determine the new size of the image
        if (float(self.net_w)/new_w) < (float(self.net_h)/new_h):
            new_h = (new_h * self.net_w)//new_w
            new_w = self.net_w
        else:
            new_w = (new_w * self.net_h)//new_h
            new_h = self.net_h

        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

        # embed the image into the standard letter box
        new_image = np.ones((self.net_h, self.net_w, 3)) * 0.5
        new_image[(self.net_h-new_h)//2:(self.net_h+new_h)//2, (self.net_w-new_w)//2:(self.net_w+new_w)//2, :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image

    def decode_netout(self, netout, anchors, obj_thresh):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        
        boxes = []
        netout[..., :2]  = _sigmoid(netout[..., :2])
        netout[..., 4]   = _sigmoid(netout[..., 4])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i // grid_w
            col = i % grid_w
            
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[row, col, b, 4]
                if(objectness <= obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[row,col,b,:4]
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / self.net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / self.net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[row,col,b,5:]
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                boxes.append(box)
        return boxes

    def correct_yolo_boxes(self, boxes, image_h, image_w):
        if (float(self.net_w)/image_w) < (float(self.net_h)/image_h):
            new_w = self.net_w
            new_h = (image_h*self.net_w)/image_w
        else:
            new_h = self.net_w
            new_w = (image_w*self.net_h)/image_h
            
        for i in range(len(boxes)):
            x_offset, x_scale = (self.net_w - new_w)/2./self.net_w, float(new_w)/self.net_w
            y_offset, y_scale = (self.net_h - new_h)/2./self.net_h, float(new_h)/self.net_h
            
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    def do_nms(self, boxes):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= self.nms_thresh:
                        boxes[index_j].classes[c] = 0

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1: return 0
            else: return min(x2,x4) - x1
        else:
            if x2 < x3: return 0
            else: return min(x2,x4) - x3 

    def run_detection(self, frame, frame_count=0):
        # Frame preprocessing
        image_h, image_w, _ = frame.shape
        batch_input = self.preprocess_input(frame)

        # Inference
        batch_output = self.model.predict_on_batch(batch_input)
        
        # Decode Output
        # YOLOv3 usually has 3 outputs. The order might depend on the model architecture.
        # Assuming standard YOLOv3 order: 13x13, 26x26, 52x52
        
        # From repo utils.py:
        # yolo_anchors = anchors[(2-j)*6:(3-j)*6]
        # It loops j in range(len(yolos)) -> 0, 1, 2
        
        boxes = []
        for j in range(len(batch_output)):
            yolo_anchors = self.anchors[(2-j)*6:(3-j)*6]
            boxes += self.decode_netout(batch_output[j][0], yolo_anchors, self.obj_thresh)

        # Correct boxes
        self.correct_yolo_boxes(boxes, image_h, image_w)
        
        # NMS
        self.do_nms(boxes)

        violations = []
        # Filter "Without Helmet" (Index 3)
        for box in boxes:
            label = box.get_label()
            score = box.get_score()
            if score > self.obj_thresh and label == 3: # 3 is Without Helmet
                # Ensure coordinates are within frame
                x1 = max(0, int(box.xmin))
                y1 = max(0, int(box.ymin))
                x2 = min(image_w, int(box.xmax))
                y2 = min(image_h, int(box.ymax))
                
                # Assign a dummy ID since tracking is not implemented in this simpler loop yet
                # Or use simple centroid tracking if needed. For now, use a random or sequential ID.
                # In robust system, would integrate tracking.
                
                # We can use frame_count + index as simplified ID
                
                violations.append({
                    "id": int(frame_count), # Simplified ID
                    "type": "NO HELMET",
                    "box": [x1, y1, x2, y2],
                    "confidence": float(score)
                })
        
        return violations

    def process_frame(self, frame):
        # Override to do nothing as we handle everything in run_detection
        return None, None
