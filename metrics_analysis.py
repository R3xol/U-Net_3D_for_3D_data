import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Ścieżki do plików
input_csv_path = "C:/Python/Studia/U-NET/output/combined_matrix.csv"
output_directory = "C:/Python/Studia/U-NET/output"

# Funkcja do wczytania danych z pliku CSV
def load_data(csv_path):
    data = {}
    with open(csv_path, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            key = row[0]  # Pierwsza kolumna jako nazwa (MSE, PSNR, itp.)
            values = list(map(float, row[1:]))  # Reszta wiersza jako dane liczbowe
            data[key] = values
    return data

# Funkcja do rysowania wykresów i zapisywania ich jako pliki PNG
def plot_and_save(data, output_dir, y_limits=None):
    for key, values in data.items():
        plt.figure(figsize=(12, 7))
        plt.plot(values, label=key)
        plt.title(f"{key} over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)

        # Ustaw zakres osi Y, jeśli podano
        if y_limits and key in y_limits:
            plt.ylim(y_limits[key])

        output_file = os.path.join(output_dir, f"{key}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Wykres {key} zapisany do {output_file}")

# Główna część programu
if __name__ == "__main__":
    # Wczytanie danych z pliku CSV
    try:
        data = load_data(input_csv_path)
        print("Dane wczytane pomyślnie:", data)
        
        # Słownik zakresów osi Y dla poszczególnych metryk
        y_limits = {
            "MSE": (0, 2000),
            "PSNR": (0, 40),
            "SSIM": (0, 1),
            "Test_loss": (0, 60),
            "Train_loss": (0, 35)
        }

        # Rysowanie wykresów i zapisywanie ich do plików
        plot_and_save(data, output_directory, y_limits = y_limits)
        print("Wszystkie wykresy zostały zapisane.")
    except FileNotFoundError:
        print(f"Plik {input_csv_path} nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")