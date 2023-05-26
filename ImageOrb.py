import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageOrb:
    def make_orb(image_path):
        image = cv2.imread(image_path,  cv2.IMREAD_UNCHANGED)

        orb = cv2.ORB_create()

        kp = orb.detect(image, None)

        kp, des = orb.compute(image, kp)
        img2 = cv2.drawKeypoints(image, kp, None , color=(0,255,0), flags=0)
        plt.imshow(img2), plt.show()

        cv2.imshow('image window', image)

        cv2.waitKey(0)