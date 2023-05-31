import pandas as pd
import csv

histogramRsplit = pd.read_csv("featuresHistR.csv")
separarteList = histogramRsplit['RedHistogram'].str.split(',', expand=True)
print(separarteList)
with open("featuresHistR.csv", "w", newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    escritor_csv.writerows(separarteList)