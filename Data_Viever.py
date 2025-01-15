import os
import numpy as np

import h5py
import matplotlib.pyplot as plt

def read_HDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        cell_matrix = f['Cell_base'][:]
        predict_matrix = f['Cell_predict'][:]
    return cell_matrix, predict_matrix

def display_and_save_images(cell_matrix, cell_predict_matrix, output_dir, index):
    """
    Wyświetla i zapisuje obrazy dla podanego indeksu w pierwszym wymiarze macierzy.
    """
    # Upewnij się, że folder wyjściowy istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Iteruj po wybranych indeksach pierwszego wymiaru
    for i in index:
        if i < 0 or i >= cell_matrix.shape[0]:
            print(f"Index {i} is out of bounds for the matrix with shape {cell_matrix.shape}")
            continue

        # Pobierz obrazy o wymiarach 240x240 dla danego indeksu
        cell_image = cell_matrix[i, :, :]
        cell_predict_image = cell_predict_matrix[i, :, :]

        # Ustal wspólny zakres wartości (od 0 do maksymalnej wartości w danych)
        max_value = max(cell_image.max(), cell_predict_image.max())

        # Utwórz figure z dwoma osiami
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        # Wyświetl obrazy z odpowiednimi skalami wartości
        im1 = axes[0].imshow(cell_image, cmap='viridis', vmin=0, vmax=max_value)
        axes[0].set_title(f'Cell (Index {i})')
        fig.colorbar(im1, ax=axes[0], orientation='vertical')

        im2 = axes[1].imshow(cell_predict_image, cmap='viridis', vmin=0, vmax=max_value)
        axes[1].set_title(f'Cell Predict (Index {i})')
        fig.colorbar(im2, ax=axes[1], orientation='vertical')

        # Zapisz obraz do pliku
        output_path = os.path.join(output_dir, f'Image_Index_{i}.png')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved image to {output_path}")

        # Pokaż obraz
        plt.show()


def load_data_and_display(data_directory, output_directory, indices):
    cnt = 0
    for i, file_name in enumerate(sorted(os.listdir(data_directory))):
        if file_name.endswith('.h5'):
            print(f"Processing file: {file_name}")

            image_path = os.path.join(data_directory, file_name)

            cell_matrix, cell_predict_matrix = read_HDF5(image_path)

            # Wyświetl i zapisz obrazy dla wybranych indeksów
            display_and_save_images(cell_matrix, cell_predict_matrix, output_directory, indices)

# Ścieżka do katalogu
data_predict = 'C:/Python/Studia/U-NET/output/predict'
output_directory = 'C:/Python/Studia/U-NET/output/Image'

# Wybrane indeksy pierwszego wymiaru do wyświetlenia
selected_indices = [10, 20, 30]

# Wczytaj dane i wyświetl obrazy
load_data_and_display(data_predict, output_directory, selected_indices)
