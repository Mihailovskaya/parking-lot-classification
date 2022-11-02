import numpy as np 
import torch
import sys
import json
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '~/src/db')


from db.occupied_parking_lots import OccupiedParkingLots
from db.photo_creation_date import PhotoCreationDate

from db.utils import add_element

class IOU_Parking():

    def __init__(self, 
                 data_json, 
                 length = 640, 
                 width = 480, 
                 path_to_empty_space = "../files/empty_places.txt"
                ):
        self._data_json = data_json
        self._length = length
        self._width = width
        self._path_to_empty_space = path_to_empty_space
        
        image_shape = [self._length, self._width, self._length, self._width]
        self._normalization = torch.tensor(image_shape)
        self._occupied_places = []
        
   

    def send_data(self):
        
        date = self._data_json["Time"]
        camera_id = self._data_json["CamID"]
        photo_creation = PhotoCreationDate(DateTime=date, Camera_ID=camera_id)
        add_element(photo_creation)
        
        self.find_ocuppied_places()
        
        Cars = self._data_json["Cars"]
        for i, place, confidence in self._occupied_places:
            x = Cars[i]["Coords"]["x"]
            y = Cars[i]["Coords"]["y"]
            w = Cars[i]["Coords"]["w"]
            h = Cars[i]["Coords"]["h"]
            occupied_place = OccupiedParkingLots(DateTime=date,
                                                 Place=place, 
                                                 Camera_ID=camera_id, 
                                                 Confidence=confidence,
                                                 x = float(x), 
                                                 y = float(y), 
                                                 w = float(w), 
                                                 h = float(h))
            add_element(occupied_place)
       
        
    def find_ocuppied_places(self):
        if len(self._data_json["Cars"])!=0:
            coords_xyxy_parking = self._load_coords_parking()
            parking_spaces = self._load_parking_spaces()
            parking_spaces_coords = [place[1::] for place in parking_spaces]
            
            iou_matrix = self._box_iou(coords_xyxy_parking, torch.tensor(parking_spaces_coords))
            self._occupied_places = []
            places=[]
            for i, car in enumerate(iou_matrix):
                max_elements, max_indices = torch.max(car, 0)
                place = parking_spaces[int(max_indices)][0]
                data_to_send = [i, place, float(max_elements)]
                
                if max_elements > 0.3 and place not in places:
                    self._occupied_places.append(data_to_send)
                    places.append(place)
        return self._occupied_places
                    
                                      
        
    def _xywh2xyxy(self, xywh):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        xyxy= xywh.clone()
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
        return xyxy

    def _box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


    def _load_coords_parking(self):
        coords_xyxy_parking = []
        for car in self._data_json["Cars"]:
            coord = [car["Coords"]["x"], car["Coords"]["y"], car["Coords"]["w"], car["Coords"]["h"]]
            coord_tensor = torch.tensor(coord).view(1, 4) * self._normalization
            coord_xyxy = self._xywh2xyxy(coord_tensor)
            coords_xyxy_parking.append(coord_xyxy[0])
        coords_xyxy_parking=torch.stack(coords_xyxy_parking)
        return coords_xyxy_parking

    def _load_parking_spaces(self):
        with open(self._path_to_empty_space) as spaces:
            parking_spaces = json.load(spaces)
            
        ''''with open('empty_space_0', "r") as empty_space:
            for car in empty_space:
                coord = [float(x) for x in car.split()][1::]
                coord_tensor = torch.tensor(coord).view(1, 4) * self._normalization
                coord_xyxy = self._xywh2xyxy(coord_tensor)
                coord_xyxy_EmptySpace.append(coord_xyxy[0])
        coord_xyxy_EmptySpace=torch.stack(coord_xyxy_EmptySpace)'''
        return parking_spaces
    
    
        
 