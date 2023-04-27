import cv2
import csv
import os

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
    return [name,folder, mean, std_dev, min_val, max_val, n_edges]


with open('features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(['Archivo','Name','Folder', 'Media', 'Desviacion estandar', 'Valor minimo', 'Valor maximo', 'Numero de bordes'])

    for j in range(9):
        for i in range(1, 101):
            name = names[j]+str(i)+').png'
            img = cv2.imread('./fotosGabriel/'+folders[j]+'/'+name)
            features = extract_features(img, name, folders[j])
            writer.writerow([os.path.basename(str(i))]+ features)

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