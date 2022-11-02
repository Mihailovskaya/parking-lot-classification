from typing import Optional
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from save_file import save_json_file
from fastapi import Request
import numpy as np
import cv2
import base64
from iou import IOU_Parking
from web import Web_Data
import json

from fastapi.middleware.cors import CORSMiddleware

class Coord(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Car(BaseModel):
    cls: int
    Confidence: float
    Coords: Coord

class CarCoords(BaseModel):
    CamID: int
    Time: str
    Cars: Optional[List[Car]] = None

class Image(BaseModel):
    base64code: str

app = FastAPI()
origins = [
    "https://parking.caas-t02.telekom.de/web_data",
    "https://parking.caas-t02.telekom.de",
    "http://10.216.0.131:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
import logging
logging.basicConfig(level=logging.INFO, filename='files/log.txt', filemode='a', format='%(asctime)s %(message)s')


@app.post("/last_image/{cam_id}/")
async def save_last_image(cam_id: int,request: Request):
    data = await request.body()
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(f"files/current_photo_{cam_id}.jpg", img)
    return {"file_size": len(data)}



@app.post("/car_coords/")
def create_item(carCoords: CarCoords, request: Request):
    try:
        json_compatible_data = jsonable_encoder(carCoords)
        #print(json_compatible_data)
        cam_id = json_compatible_data["CamID"]
        iou = IOU_Parking(json_compatible_data, path_to_empty_space=f"files/parking_spaces_{cam_id}.txt")
        iou.send_data()
        time_of_photo = json_compatible_data["Time"]
        save_json_file(json_compatible_data=json_compatible_data, name=f"json1/cam_{cam_id}_{time_of_photo}")
    except Exception as e:
        logging.warning(f"Saving data to database failed. Error: {e}")
        return str(e)

    return carCoords



@app.get("/camera/{cam_id}/IP/")
def save_camera_IP(cam_id: int, request: Request):
    try:
        client_host = request.client.host
        with open(f'files/camera_{cam_id}_IP.txt', 'w') as f:
                f.write(str(client_host)+"\n")
    except Exception as e:
        logging.warning(f"Saving IP adress of camera {cam_id} failed. Error: {e}")        
        return str(e)
    return request.client.host

@app.get("/web_data/")
def get_data():
    places_now = None
    graph_data = None
    web = Web_Data()
    image_cam0, datetime_photo_UTC_0 = web.print_free_spaces(0)
    image_cam1, datetime_photo_UTC_1 = web.print_free_spaces(1)
      
    _, image_cam_0_encoded = cv2.imencode('.jpg', image_cam0)
    _, image_cam_1_encoded = cv2.imencode('.jpg', image_cam1)
     
    places_now = web.places_now()
    graph_data = web.graph_data()
    
    final_json = {}
    final_json["image_0_base64_jpg"] =  base64.b64encode(image_cam_0_encoded)
    final_json["image_0_datetime"] = datetime_photo_UTC_0
    final_json["image_1_base64_jpg"] =  base64.b64encode(image_cam_1_encoded)
    final_json["image_1_datetime"] = datetime_photo_UTC_1
    final_json["current_places"] =  places_now
    final_json["graph_data"] =  graph_data

    return final_json


