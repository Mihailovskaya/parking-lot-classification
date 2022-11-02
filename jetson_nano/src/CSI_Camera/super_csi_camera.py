from jetcam.csi_camera import CSICamera
#from src.CSI_Camera.csi_camera import CSICamera
import cv2
import os
import re
import datetime
import logging

class SuperCSICamera (CSICamera):
    
    def save (self, path=""):
        date = datetime.datetime.now() 
        date.strftime('%d.%m.%Y %H:%M')
        path =  os.path.join(path, str(date)+".jpg")
        if cv2.imwrite(path, self.value):
            logging.info('Photo saved in '+path)
        else:
            logging.error('Photo not saved')
        return path
        
    def exit (self):
        self.cap.release()