import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img(cx, cy, r_pupil, r_out_bound, pupil_cont, img, path):
    img_3 = np.zeros(img.shape+(3,))
    img_3[:,:,0] = img_3[:,:,1] = img_3[:,:,2] = img/255.0
    #cv2.rectangle(img, (cx,cy), (cx,cy), (120,0,0), 10)
    cv2.circle(img_3, (cx, cy), r_pupil, (1, 0, 0), 1)
    cv2.circle(img_3, (cx, cy), r_out_bound, (0, 0, 1), 1)
    cv2.drawContours(img_3, pupil_cont, -1, (0,1,0), 1)
    
    plt.imshow(img_3)
    plt.title(path)
    plt.show()