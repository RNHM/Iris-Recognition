import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import polarTransform
from PIL import Image, ImageDraw
from localize_iris import localize_iris

def create_polar_im(path,save_folder):
    try:
        cx, cy, r_pupil, r_out_bound, _, img = localize_iris(path)
        polarImage, ptSettings = polarTransform.convertToPolarImage(img, center = (cx, cy),initialRadius=r_pupil,finalRadius=r_out_bound, angleSize=360)        
        im = np.transpose(polarImage)
        im = (im-np.min(im))/(np.max(im)-np.min(im))*255.0
        im = Image.fromarray(np.uint8(im))
        save_path = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)[:-4]
        im.save(os.path.join(save_folder,save_path)+'.png')
    except:
        pass
