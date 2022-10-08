import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import polarTransform
from PIL import Image, ImageDraw
from localize_iris_DL import localize_iris_DL

def create_polar_im_DL(path,path2,save_folder):
    try:
        img = cv2.imread(path, 0)
        img = cv2.medianBlur(img,5)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,4,20, param1=10,param2=100,minRadius=0,maxRadius=60)
        circles = np.uint16(np.around(circles))
        i = circles[0,:][0]
        cx = i[0]
        cy = i[1]
        r_pupil = i[2]
        line = img[i[1],:]
        deltaR = np.sum(line==255)/2
        r_out_bound = int(deltaR) + r_pupil
        img = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        img = np.float32(img)
        polarImage, ptSettings = polarTransform.convertToPolarImage(img, center = (cx, cy),initialRadius=r_pupil,finalRadius=r_out_bound, angleSize=360)        
        im = np.transpose(polarImage)
        im = (im-np.min(im))/(np.max(im)-np.min(im))*255.0
        im = Image.fromarray(np.uint8(im))
        save_path = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)[:-4]
        im.save(os.path.join(save_folder,save_path)+'.png')
    except:
        pass
