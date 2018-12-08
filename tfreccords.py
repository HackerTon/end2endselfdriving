import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

filesname = '/home/hackerton/data_driving'
driving_log = pd.read_csv(filesname + "/driving_log.csv",
                          names=['center', 'left', 'right', 'angle', 'acceleration', 'brake', 'speed'])


def img_load(image_loc):
    return cv2.imread(image_loc, cv2.IMREAD_COLOR)


my_file = Path('diskarray.dat')

diskArray: np.ndarray = np.memmap('diskarray.dat', dtype=np.float32, mode='w+',
                                  shape=(driving_log.__len__(), 160, 320, 3))

if not my_file.is_file():
    for index, value in enumerate(driving_log['center']):
        diskArray[index] = img_load(filesname + '/' + value) / 255

diskArray = diskArray - diskArray.mean(axis=0)
