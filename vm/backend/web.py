from db.base import engine as engine
from sqlalchemy.orm import Session
from db.occupied_parking_lots import OccupiedParkingLots
from db.photo_creation_date import PhotoCreationDate
from sqlalchemy import func
import datetime as dt
from datetime import timedelta
import json
from sqlalchemy import desc
import cv2
import torch
import numpy as np
import pandas as pd
import logging
import os
import time

class Web_Data():
    def __init__(self,
                 path_to_priorities_cam0 = "files/priorities_cam0.txt",
                 path_to_all_names = 'files/all_names.txt'
                ):
        self._all_parking_names = None
        self._priorities_cam0 = None
        self._load_files(path_to_all_names,path_to_priorities_cam0)
        
        

    
    def places_now(self):
        json_occupied_places = []
        occupied_places=[]
        
        occupied_places, camera_none = self._occupied_places_in_datetime(dt.datetime.utcnow())
        if occupied_places == None:
            return None
        print(camera_none)
       
        for place in occupied_places:
            json_occupied_places.append({f'SE{place}':1})
        
        if camera_none!=None:
            if camera_none==1:
                camera_id=0
            else:
                camera_id=1
                
            with open(f'files/parking_spaces_{camera_id}.txt') as parking_spaces:
                parking_spaces_coord = json.load(parking_spaces)
           
            #for name in parking_spaces_coord:
             #   if {f'SE{name[0]}':1} not in json_occupied_places:
              #      json_occupied_places.append({f'SE{name[0]}':0})
            parking_spaces_coord = self._files_for_mark_free_spaces(camera_id = camera_id, image_add=False)
            for place in parking_spaces_coord:
                json_occupied_places.append({f'SE{place[0]}':0})
        else:
            for name in self._all_parking_names:
                if {name:1} not in json_occupied_places:
                    json_occupied_places.append({name: 0})
                    
        if camera_none!=None:
            parking_spaces_coord = self._files_for_mark_free_spaces(camera_id = camera_none, image_add=False)
            for place in parking_spaces_coord:
                if ({f'SE{place[0]}':0} or {f'SE{place[0]}':1}) not in json_occupied_places:
                    json_occupied_places.append({f'SE{place[0]}':None})
        return json_occupied_places
    
    def graph_data(self):
        datetime_yesterday = dt.datetime.utcnow()-timedelta(hours=24)
        datetime = dt.datetime.utcnow()
        json_graph = []
        while datetime > datetime_yesterday:
            occupied_places, camera_none = self._occupied_places_in_datetime(datetime)
            if occupied_places == None: # or camera_none != None:
                json_graph.append({datetime: [None,0]})
            elif occupied_places != None and camera_none != None:
                json_graph.append({datetime: [len(occupied_places),1]}) 
            elif occupied_places != None and camera_none == None:
                json_graph.append({datetime: [len(occupied_places),2]}) 
            datetime = datetime - timedelta(minutes=5)
            #json_graph=json_graph.reverse()
        return json_graph

    def print_free_spaces(self, camera_id):
        datetime_UTC = dt.datetime.utcnow()
        current_image, parking_spaces_coord = self._files_for_mark_free_spaces(camera_id)
        
        photo_ctime = os.path.getmtime(f'files/current_photo_{camera_id}.jpg')
        datetime_photo = dt.datetime.strptime(time.ctime(photo_ctime), "%a %b %d %H:%M:%S %Y")
        datetime_photo_UTC = datetime_photo-timedelta(hours=3)
        
        query_return_occupied_places = self._return_occupied_places(camera_id=camera_id, DateTime=datetime_UTC)
        if query_return_occupied_places == None:
            logging.info("No occupied places found")
            return current_image, datetime_photo_UTC
        
        logging.info("Occupied places found")
        occupied_places = [place[0] for place in query_return_occupied_places]
        image_with_boxes = self._mark_free_parking_spaces(camera_id,
                                                          current_image,
                                                          parking_spaces_coord,
                                                          occupied_places
                                                         )
        return image_with_boxes, datetime_photo_UTC
    
    
    def _load_files(self, path_to_all_names, path_to_priorities_cam0):
        with open(path_to_all_names) as names:
            self._all_parking_names = json.load(names)
        with open(path_to_priorities_cam0) as names:
            self._priorities_cam0 = json.load(names)
            
    
    def _mark_free_parking_spaces(self,camera_id, image, parking_spaces_coord, occupied_places):
       
        green_color = [0, 255, 0]
        
        for car in parking_spaces_coord:
            if car[0] not in occupied_places:
                car_coord = car[1::]
                label = f"{int(car[0])}"
                image = self._plot_one_box(car_coord, 
                             image, 
                             color=green_color,
                            # label=label,           
                             line_thickness=1)
        #cv2.imwrite('files/222.jpg', image)
        return image
    
    def _files_for_mark_free_spaces(self,camera_id, image_add=True):
        with open(f'files/parking_spaces_{camera_id}.txt') as parking_spaces:
            parking_spaces_coord = json.load(parking_spaces)
        if image_add == True:
            image = cv2.imread(f'files/current_photo_{camera_id}.jpg')
            return image, parking_spaces_coord
        else:
            return parking_spaces_coord
    

                
    def _occupied_places_in_datetime(self, datetime = dt.datetime.utcnow()):
        
        query_return_occupied_places_cam0 = self._return_occupied_places(camera_id=0, DateTime=datetime)
        query_return_occupied_places_cam1 = self._return_occupied_places(camera_id=1, DateTime=datetime)
        print(query_return_occupied_places_cam1)
        camera_none = None
        #добавить, чтобы мог смотреть только одну камеру. заполнять единицами с запроса и none имена со второй камеры 
        if query_return_occupied_places_cam0 == None and query_return_occupied_places_cam1 == None:
            return None,None
        
        elif query_return_occupied_places_cam0 != None and query_return_occupied_places_cam1 == None:
            occupied_places_cam0 = [place[0] for place in query_return_occupied_places_cam0]
            camera_none = 1
            
            return occupied_places_cam0, camera_none
        
        elif query_return_occupied_places_cam0 == None and query_return_occupied_places_cam1 != None:
            occupied_places_cam1 = [place[0] for place in query_return_occupied_places_cam1]
            camera_none = 0
            return occupied_places_cam1, camera_none
        

        
        occupied_places_cam0 = [place[0] for place in query_return_occupied_places_cam0]
        occupied_places_cam1 = [place[0] for place in query_return_occupied_places_cam1]
        
        occupied_places=self._cross_check(occupied_places_cam0, occupied_places_cam1)
        return occupied_places, camera_none
    
    def _cross_check(self, occupied_places_cam0, occupied_places_cam1):
        
        occupied_places=[]
        for place in occupied_places_cam0:
            if place not in occupied_places:
                if place in self._priorities_cam0:
                    occupied_places.append(place)
                elif place not in self._priorities_cam0 and place in occupied_places_cam1:
                    occupied_places.append(place)
            
        for place in occupied_places_cam1:
            if place not in occupied_places and place not in self._priorities_cam0:
                occupied_places.append(place)
        return(occupied_places)

    def _return_occupied_places(self, camera_id, DateTime):
        with Session(engine) as session:
            #DateTime = DateTime - timedelta(hours=36)
            datetime_5minutes_ago = DateTime-timedelta(minutes=5)
            DateTime_all = session.query(PhotoCreationDate.DateTime).filter(PhotoCreationDate.Camera_ID == camera_id)
            data_in_interval  = DateTime_all.filter(PhotoCreationDate.DateTime <= DateTime, PhotoCreationDate.DateTime >= datetime_5minutes_ago)
            DateTime_LastPfoto_in_interval = data_in_interval.order_by(desc(PhotoCreationDate.DateTime)).first()
            if DateTime_LastPfoto_in_interval==None:
                return None
            query_occupied_places = session.query(OccupiedParkingLots.Place).filter(OccupiedParkingLots.DateTime == DateTime_LastPfoto_in_interval[0]).all()
            session.commit()
        return query_occupied_places
    
    
    
    
    
    def _plot_one_box(self, x, img, color,  line_thickness , label=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        image=cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            image=cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            image=cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return image


    def _xywh2xyxy(self,xywh):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            xyxy= xywh.clone()
            xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
            xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
            return xyxy
        
    
    

    



