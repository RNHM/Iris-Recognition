import cv2
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import polarTransform
from PIL import Image, ImageDraw
from tqdm import tqdm

def localize_iris(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)
    smooth = cv2.GaussianBlur(img, (21,21), 5)
    diff = (img - smooth) <= 0
    op_size = 11
    opening = cv2.morphologyEx(np.float32(diff), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(op_size,op_size)))
    ret,binary = cv2.threshold(np.uint8(opening),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy= cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidate_contours = []
    ecc_vect = []
    circ_vect = []
    for i, c in enumerate(contours):
        ar = cv2.contourArea(c)
        if ar > 700:
            (center,axes,orientation) = cv2.fitEllipse(c)
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            ecc_vect.append(np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
            candidate_contours.append(c)
    if len(ecc_vect)==0:
        pass
    else:
        pupil_cont = candidate_contours[np.argmin(ecc_vect)]
        x,y,w,h = cv2.boundingRect(pupil_cont)
        r_pupil = int(w/2)
        cx = int(x + w/2)
        cy = (y+h) - r_pupil

        polar_thres, ptSettings = polarTransform.convertToPolarImage(smooth, center = (cx, cy),initialRadius=None, finalRadius=7*r_pupil, radiusSize=7*r_pupil,angleSize=360)
        polar_thres = (np.transpose(polar_thres))
        
        # Create a mask of the artificial 0s created by the polar system conversion
        mask = ((polar_thres)!=0.0)

        # Count how many 0s have been created row-wise 
        notnan = np.sum(mask, axis=1)
        unique = np.unique(notnan)
        if len(unique==1):
            notnan = notnan/np.max(notnan)
        else:
            notnan = (notnan-np.min(notnan))/(np.max(notnan)-np.min(notnan))
        # If the newly added 0s are more than 25%, then treat them as if there were no 0s
        notnan[notnan<0.85]=1.0

        sobely = cv2.Sobel(polar_thres,cv2.CV_32F,0,1,ksize=5)

        sobely = sobely*mask
        cum_sum = np.sum(sobely,axis=1)
        cum_sum = (cum_sum-np.min(cum_sum))/(np.max(cum_sum)-np.min(cum_sum))
        cum_sum = cum_sum/notnan

        margin = int(0.8*r_pupil)
        r_out_bound = np.argmax(cum_sum[r_pupil+margin:]) + r_pupil + margin


        r_out_bound = int(r_out_bound)

        return cx, cy, r_pupil, r_out_bound, pupil_cont, img