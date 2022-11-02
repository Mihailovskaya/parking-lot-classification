import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import json
import datetime
import configparser
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from configparser import ConfigParser
import sys
import logging
from base64 import b64encode

from src.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from src.utils.plots import plot_one_box
from src.utils.torch_utils import select_device, load_classifier, time_synchronized

from src.models.models import *
from src.utils.datasets import *
from src.utils.general import *



class Model_inference():

    def __init__(self, 
                 output='run/',
                 config_path = 'files/configs/service/config.ini',
                 camera_id = 0,
                 confidence = None,
                 weights = None,
                 image_size = None,
                 iou_threshold = None
                ):
       
        """Initializing Model class:

        Keyword arguments:
        save_txt -- save results to *.txt: True or False
        classes -- 'filter by class: None or list of int'
        weights -- model.pt path(s): string, 
        source -- source: string
        output -- output: string
        augment -- augmented inference: True or False
        agnoistic -- class-agnostic NMS: True or False
        imgs -- inference size (pixels): int
        conf -- object confidence threshold: float
        iou -- IOU threshold for NMS: float
        device -- cuda device, i.e. 0 or 0,1,2,3 or cpu: string
        cfg -- *.cfg path: string
        names --*.cfg path: string
        """
        config = ConfigParser()

        config.read(config_path)
        self._source = None
        self._output = output
        self._weights = weights or [config.get('model', 'weights'), ]
        self._image_size = image_size or config.getint('model', 'imgs')
        self._confidence = confidence or config.getfloat('model', 'conf')
        self._iou_threshold = iou_threshold or config.getfloat('model', 'iou')
        self._device = config.get('model', 'device')
        self._model_config_path = config.get('model', 'model_config_path')
        self._classes_names = json.loads(config.get('model', 'classes_names'))
        self._augment = config.getboolean('model', 'augment')
        self._agnostic = config.getboolean('model', 'agnoistic')
        self._classes = json.loads(config.get('model', 'classes')) if config.get('model', 'classes') != 'None' else None
        self._prediction = None
        self._devicecuda = None
        self._half = None
        self._model_darknet = None
        self._camera_id = camera_id
        self._load_model()

    
    def detect(self, image):
        self._prediction = None
        self._source = image
        start_time = time.time()

        image = self._image_conversion()

        self._make_prediction(image)
        data_json = self._make_json()

        working_time = time.time() - start_time
        
        logging.info('Done. Time: '+ str(working_time))
        print(f'Done. ({working_time})')
        
        image = self._add_bounding_boxes_to_image()
        
        return image, data_json
    
    def _add_bounding_boxes_to_image(self):
        colors = [[0, 0, 255],[0, 0, 255],[0, 0, 255],[0, 0, 255]]
        
        if self._prediction is None:
            return self._source
        for i, recognized_car in enumerate(self._prediction):
            *xyxy, confidence, cls = recognized_car
            # Add bbox to image
            class_id = self._classes_names[int(cls)]
            #label = f'{class_id} {confidence}'
            image_with_box = plot_one_box(xyxy, 
                                 self._source, 
                               #  label=label, 
                                 color=colors[int(cls)], 
                                 line_thickness=1)
            
        return image_with_box
       # if cv2.imwrite(self._output + 'pred.jpg', image_with_box):
        #    logging.info('Photo with prediction saved')
       # else:
       #     logging.warning('Photo witn prediction not saved')
      #  return 


        
            
    def save_txt(self):
        try:
            for i, recognized_car in enumerate(self._prediction):
                *xyxy, conf, cls = recognized_car

                xywh = self._normalize_xywh(xyxy)  # normalized xywh
                with open(self._output + 'pred.txt', 'a') as f:
                    f.write(f'{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
            logging.info('txt witn prediction saved')
        except:
            logging.warning('txt witn prediction not saved')


    def _initialize_cuda(self):
        # Initialize cuda
        self._devicecuda = select_device(self._device)
        self._half = self._devicecuda.type != 'cpu'  # half precision only supported on CUDA

    def _load_model(self):
        self._initialize_cuda()
        model_darknet = Darknet(self._model_config_path, self._image_size).cuda()
        model_darknet.load_state_dict(torch.load(self._weights[0], map_location=self._devicecuda)['model'])

        model_darknet.to(self._devicecuda).eval()
        if self._half:
            model_darknet.half()  # to FP16
        self._model_darknet = model_darknet

    def _image_conversion(self):
        image = letterbox(self._source, new_shape=self._image_size, auto_size=64)[0]
        # Convert
        image_rgb = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image_cont = np.ascontiguousarray(image_rgb)
        image_torch = torch.from_numpy(image_cont).to(self._devicecuda)
        image_torch = image_torch.half() if self._half else img_torch.float()  # uint8 to fp16/32
        image_norm = image_torch / 255.0  # 0 - 255 to 0.0 - 1.0
        if image_norm.ndimension() == 3:
            image_norm = image_norm.unsqueeze(0)
        return image_norm

    def _normalization_gain(self):
        return torch.tensor(self._source.shape)[[1, 0, 1, 0]]

    def _make_prediction(self, image):
        # Inference
        prediction = self._model_darknet(image, augment=self._augment)[0]
        # Apply NMS
        prediction_nms = non_max_suppression(prediction, 
                                         self._confidence, 
                                         self._iou_threshold, 
                                         classes=self._classes, 
                                         agnostic=self._agnostic)
        detection = prediction_nms[0]

        if detection is not None and len(detection):
            # Rescale boxes from img_size to source size
            #заменяет первые 4 координаты в каждом предикшине на новые
            new_coords = scale_coords(image.shape[2:], 
                                      detection[:, :4], 
                                      self._source.shape)
            new_coords_round = new_coords.round()
            detection[:, :4] = new_coords_round
            self._prediction = detection

    def _normalize_xywh(self, xyxy):
        # xyxy2xywh Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        normalization_gain = self._normalization_gain()
        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
        normalize_xywh = (xywh / normalization_gain).view(-1).tolist()
        return normalize_xywh


    def _make_json(self):
        

        data_json = ({"CamID": self._camera_id,
                      "Time": str(datetime.datetime.now()),
                      "Cars": [],
                      })
        if self._prediction is not None and len(self._prediction):
            for i, detection in enumerate(self._prediction):
                *xyxy, conf, cls = detection
                xywh = self._normalize_xywh(xyxy) 
                data_json["Cars"].append({
                    "cls": int(cls),
                    "Confidence": float(conf),
                    "Coords": {"x": float(xywh[0]),
                               "y": float(xywh[1]),
                               "w": float(xywh[2]),
                               "h": float(xywh[3])
                               }
                })
        return data_json

       
