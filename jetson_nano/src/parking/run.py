import os
from src.CSI_Camera.super_csi_camera import SuperCSICamera
import time
from src.parking.model_inference import Model_inference
import requests
import logging
import getpass
import cv2
from configparser import ConfigParser



DESTINATION = "http://10.216.0.131:8001/"

CONFIG_PATH = "files/configs/service/config.ini"
CAMERA_ID_PATH = "files/configs/service/camera_id.ini"

def save_photo():
    camera = SuperCSICamera(width=640, height=480)
    logging.info('Camera connected')
    while True:
        camera.read()
        path=camera.save(path = 'image/')
        
        time.sleep(5*60)
        
def detect_parking():
    
    config = ConfigParser()

    config.read(CAMERA_ID_PATH)
    camera_id = config.get('camera','camera_id')
    try:
        camera = SuperCSICamera(width=640, height=480)
        logging.info('Camera connected')

    except Exception as e:
        logging.warning(f'Camera not connected. Exception: {e}, {type(e)}')
        print(f'Camera not connected. Exception: {e}, {type(e)}')
        return

    
    model_inference = Model_inference(camera_id=camera_id, config_path=CONFIG_PATH)
    logging.info('Model loaded')

    while True:
        try:
            image = camera.read()
            logging.info('Photo taken')
        except RuntimeError as e:
            logging.warning(f'Photo not taken. Exception: {e}, {type(e)}')
            camera.exit()
            return
            
        except Exception as e:
            logging.warning(f'Photo not taken. Exception: {e}, {type(e)}')
            continue
        
        path = camera.save(path = 'image/')
        
        image, data = model_inference.detect(image)
        if data is not None:
            logging.info('Occupied parking spaces found')
            r = requests.post(DESTINATION + "car_coords/", json=data)
            r = requests.get(DESTINATION + f"camera/{camera_id}/IP/")
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}
            _, img_encoded = cv2.imencode('.jpg', image)                  
            response = requests.post(DESTINATION + f"last_image/{camera_id}/", data=img_encoded.tobytes(), headers=headers)
            logging.info('Data sent to: '+ DESTINATION)
        else:
            logging.warning('Occupied parking spaces not found')

        time.sleep(5*60)

        
