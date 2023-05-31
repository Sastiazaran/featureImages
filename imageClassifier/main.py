import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folders=["attackOnTitan_fotos", "deathNote_fotos", "Evangelion", "KimetsuNoYaiba", "LOTR","NBA", "onePiece_fotos", "SNKRS", "StarWars"]
names =["AOT (","deathNote (","Evangel (","KNY (","LOTR (", "nba (","OnePiece (", "Snkrs (", "SW ("]


# Función para calcular las características de una imagen
def extract_features(img, name, folder):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray)[0]
    std_dev = cv2.meanStdDev(gray)[1][0][0]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    edges = cv2.Canny(gray, 50, 150)
    n_edges = cv2.countNonZero(edges)

    #saturation of the red channel
    hsvR = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satR = cv2.mean(hsvR[:,:,0])[0]
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])


    #saturation of the green channel
    hsvG = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satG = cv2.mean(hsvG[:,:,1])[0]
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])

    #saturation of the blue channel
    hsvB = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    satB = cv2.mean(hsvB[:,:,2])[0]
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])

    histRList  = []
    histGList  = []
    histBList  = []

    for i in histR:
        histRList.append(float(i))
    for i in histG:
        histGList.append(float(i))
    for i in histB:
        histBList.append(float(i))



    # return [name,folder, mean, std_dev, min_val, max_val, n_edges, satR, satG, satB]
    return [name, folder, histRList]


with open('featuresHistR.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # writer.writerow(['Name','Folder', 'Media', 'Desviacion estandar', 'Valor minimo', 'Valor maximo', 'Numero de bordes', 'Saturación de rojo',
    #                  'Saturacion de verde', 'Saturacion de azul'])

    writer.writerow(['Name', 'Folder', 'RedHistogram'])

    for j in range(9):
        for i in range(1, 101):
            name = names[j]+str(i)+').png'
            img = cv2.imread('./fotosGabriel/'+folders[j]+'/'+name)
            features = extract_features(img, name, folders[j])
            writer.writerow(features)

print("Ready")

while True:
    print("1) attackOnTitan_fotos")
    print("2) deathNote_fotos")
    print("3) Evangelion")
    print("4) KimetsuNoYaiba")
    print("5) LOTR")
    print("6) onePiece_fotos")
    print("Seleccione el numero de carpeta")
    carpeta= int(input()) 
    print("Escooja una imagen del 1 al 100")
    id = input()
    nameOpc = names[carpeta-1]+id+').png'
    imgOpc = cv2.imread('./fotosGabriel/'+folders[carpeta-1]+'/'+name)
    print("Valores ", extract_features(img, nameOpc, folders[carpeta-1]))
    # trazar el histograma del canal rojo
    plt.plot(features[-3], color='r')
    plt.xlim([0, 256])
    plt.show()
    # trazar el histograma del canal verde
    plt.plot(features[-2], color='g')
    plt.xlim([0, 256])
    plt.show()
    # trazar el histograma del canal azul
    plt.plot(features[-1], color='b')
    plt.xlim([0, 256])
    plt.show()