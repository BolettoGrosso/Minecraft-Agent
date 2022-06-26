import pyautogui
import keyboard
import time 
from PIL import ImageGrab
from PIL import Image
import numpy as np
import cv2
import pydirectinput

print("AKTIVAO")
time.sleep(3)

#Toquem el minecraft
#pydirectinput.press("w")
im1 = pyautogui.screenshot("currentImg.png", region=(0,35,2000, 985))

#Legim les fotografies
wim2 = cv2.imread("currentImg.png")
wheat_img = cv2.imread('slimerino.png')

#Anàlisi
print ("POST 1")
result = cv2.matchTemplate(wim2, wheat_img, cv2.TM_CCOEFF_NORMED)

#Mostrem la fotografia
cv2.imshow('Result', result)
cv2.waitKey()
cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print("resultat,",max_val)

#cv2.imshow("cropped", wim2)
#cv2.imshow("slimerino.png", wheat_img)
#cv2.waitKey(0)

print("finish")