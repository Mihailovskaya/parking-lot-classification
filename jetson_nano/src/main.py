import os
import sys
import datetime

sys.path.insert(1, '/root/parking_lot_classification')
os.chdir('/root/parking_lot_classification')

from src.parking.run import detect_parking

import logging
logging.basicConfig(level=logging.INFO, filename='logs/log.txt', filemode='a', format='%(asctime)s %(message)s')

if __name__ == '__main__':
    date = datetime.datetime.now() 
    date.strftime('%d.%m.%Y %H:%M')
    logging.info('\n PROGRAM LAUNCHED: '+ str(date))
    detect_parking()