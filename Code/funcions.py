
from detecto.utils import read_image
import pyautogui
import torch
import pandas as pd
import numpy as np
import cv2
import time
import win32api
import win32con
import pydirectinput


def adquirir_dataset(model):
    im1 = pyautogui.screenshot("currentImg.jpeg", region=(0,35,2000, 985))
    image = read_image("currentImg.jpeg")
    labels, boxes, scores = model.predict(image)
    threshold = 0.6
    import numpy as np
    llista_boxes = boxes.tolist()
    boxes_salvats=[]
    labels_salvats=[]
    for i in range(len(scores)):
        if scores[i]>threshold:
            labels_salvats.append(labels[i])
            boxes_salvats.append(llista_boxes[i]) 

    boxes_salvats_tensor = torch.FloatTensor(boxes_salvats)
    distancies = []
    entitats_input = []

    for i in boxes_salvats_tensor:
        distancies.append(str(i[3]-i[1]))
        entitats_input.append([  int( (i[3].item() - i[1].item()) ),  (i[2].item()+i[0].item())/2 ,  (i[3].item()+i[1].item())/2 
                              ])

    for j in range(len(distancies)):
        distancies[j] = distancies[j][7:-1]

    for k in range(len(labels_salvats)):
        entitats_input[k].append(labels_salvats[k])
        labels_salvats[k] = labels_salvats[k] + " " + distancies[k]

    if entitats_input != []:
        entitats_input = pd.DataFrame(entitats_input)
        entitats_input = entitats_input.sort_values(0, axis=0, ascending=False, inplace=False, kind='quicksort',
                         na_position='last', ignore_index=False, key=None)
        entitats_input = entitats_input.replace("zombie", 1)
        entitats_input = entitats_input.replace("pig", 0)

        del entitats_input[2]
    print ("entitats_input", entitats_input)
    return (entitats_input)

def adquirir_dataset_basic(model):
    im1 = pyautogui.screenshot("currentImg.jpeg", region=(0,35,2000, 985))
    image = read_image("currentImg.jpeg")
    labels, boxes, scores = model.predict(image)
    threshold = 0.6
    import numpy as np
    llista_boxes = boxes.tolist()
    boxes_salvats=[]
    labels_salvats=[]
    for i in range(len(scores)):
        if scores[i]>threshold:
            labels_salvats.append(labels[i])
            boxes_salvats.append(llista_boxes[i])
    boxes_salvats_tensor = torch.FloatTensor(boxes_salvats)
    distancies = []
    entitats_input = [] #Ha de ser una llista, un diccionari no pot tenir instancies diferentss amb la mateixa etiqueta

    for i in boxes_salvats_tensor:
        entitats_input.append([  (i[2].item()+i[0].item())/2,  i[3].item()
                              ])

    return entitats_input


def execucio_xarxa(function_inputs, solution):
    print("solution", solution)
    if len(function_inputs) != 0:
        function_inputs = list(function_inputs[0])
        print("inputs tractats", function_inputs)

    elif len(function_inputs) == 0:
        function_inputs=[0,0]
                
    else:
        print("nai què collons?", function_inputs)
        
    valor_dreta = (solution[0:2]*function_inputs).sum() 
    valor_esquerra = (solution[2:4]*function_inputs).sum()
    valor_w = (solution[4:6]*function_inputs).sum()
    print ("vd", valor_dreta, "ve", valor_esquerra, "vend", valor_w)        
    resultat = max(valor_dreta,valor_esquerra, valor_w) 
    return [resultat,valor_dreta,valor_esquerra, valor_w] 

def execucio_moviment(resultat,valor_dreta,valor_esquerra, valor_w):
    if resultat == valor_w:
        print("endavant")
    elif resultat == valor_dreta:
        print("dreta")
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 50, 0, 0, 0)
    elif resultat == valor_esquerra:     
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -50, 0, 0, 0)
        
def avaluar_xarxa(function_inputs,solution):
    visio = np.uint8(pyautogui.screenshot("currentImg.jpeg", region=(0,35,2000, 985)))
    puntuacions = []
    for puntuacio in range(1,10):
        nai = cv2.imread(str(str(puntuacio)+".jpg"))
        pointo = cv2.matchTemplate(visio, nai, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(pointo)
        puntuacions.append(max_val)
    
    max_val = max(puntuacions)
    max_index = puntuacions.index(max_val)+1
    puntuacio = puntuacions[int(max_val)]

    pyautogui.click(x=1091, y=605)
    pyautogui.click(x=1091, y=605)
    
    compta = 0
    if len(function_inputs) != 0:
        max_index = max_index+0.5
    print("les puntuacions son", max_index, "i els pesos son", solution)
    return(max_index)

def avaluar_xarxa2(function_inputs_zero):
    if len(function_inputs_zero) == 0:
        return 0
    else:
        puntuacio = 1/((function_inputs_zero[0][0]-960)**2)
        return puntuacio
