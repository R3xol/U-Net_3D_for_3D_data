import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_row_from_csv(filename, row_name):
    # Wczytaj dane z pliku CSV
    df = pd.read_csv(filename, header=None)
    
    # Konwersja do macierzy numpy
    combined_matrix = df.to_numpy(dtype=object)
    
    # Znajdź odpowiedni wiersz
    row_data = None
    for row in combined_matrix:
        if row[0] == row_name:
            row_data = row[1:].astype(float)
            break
    
    if row_data is None:
        print(f"Wiersz '{row_name}' nie został znaleziony w pliku.")
        return
    
    

    # Tworzenie wykresu
    plt.figure(figsize=(14, 8))
    plt.plot(row_data, linestyle='-',lw = 2,  label=row_name)  #, marker='o', markersize=1
    plt.title(f"Wykres dla {row_name}")
    plt.xlabel("Epoka")
    if row_name == "Train_loss" or row_name == "Train_loss":
        row_name = "RMSE"
    plt.ylabel(row_name)
    
    plt.legend()
    plt.grid()
    plt.show()

# Przykładowe użycie
plot_row_from_csv("C:\\Python\\Studia\\PracaMagisterska\\Zapisane_dane\\Udane_uczenie\\combined_matrix.csv", "Train_loss")

# Zawartośc combined_matrix.csv
'''combined_matrix = np.array([
        ["MSE"] + MSE.tolist(),
        ["PSNR"] + PSNR.tolist(),
        ["SSIM"] + SSIM.tolist(),
        ["Test_loss"] + Test_loss.tolist(),
        ["Train_loss"] + Train_loss.tolist()
    ], dtype=object)'''