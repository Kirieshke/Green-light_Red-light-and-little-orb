import cv2
import numpy as np
from matplotlib import pyplot as plt
import ImageOrb
import VideoOrb
from FaceFinder import face_finder

face_finder = face_finder()  
face_finder.find_body() 
