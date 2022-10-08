import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import polarTransform
from localize_iris import localize_iris
from PIL import Image

shape = (240,320)

def create_mask0(path):
    try:
        cx, cy, r_pupil, r_out_bound, _ , _ = localize_iris(path)
        c = (cx,cy)
        return [c, r_pupil, r_out_bound]
    except:
        return None
    

def create_mask(path, save_folder):
    obj = create_mask0(path)
    if obj is None:
        im = np.zeros(shape, np.uint8)
        im = Image.fromarray(im)
    else:
        im1 = np.zeros(shape, np.uint8)
        im2 = np.zeros(shape, np.uint8)
        c = obj[0]
        r1 = obj[1]
        r2 = obj[2]
        cv2.circle(im1, c, r1, (255,255,255),-1)
        cv2.circle(im2, c, r2, (255,255,255),-1)
        im = (im2==255.0) & (im1 == 0.0)
        im = Image.fromarray(im)
    save_path = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)[:-4]
    im.save(os.path.join(save_folder,save_path)+'.png')