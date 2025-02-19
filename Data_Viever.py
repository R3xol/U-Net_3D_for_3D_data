import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

def read_HDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        OCT_matrix = f['OCT'][:]
        cell_matrix = f['Cell_base'][:]
        predict_matrix = f['Cell_predict'][:]
    return OCT_matrix, cell_matrix, predict_matrix

def display_and_save_images(OCT_matrix, cell_matrix, cell_predict_matrix, output_dir, index):
    """
    Wyświetla i zapisuje obrazy dla podanego indeksu w pierwszym wymiarze macierzy.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in index:
        if i < 0 or i >= cell_matrix.shape[0]:
            print(f"Index {i} is out of bounds for the matrix with shape {cell_matrix.shape}")
            continue

        OCT_image = OCT_matrix[i, :, :]
        cell_image = cell_matrix[i, :, :]
        cell_predict_image = cell_predict_matrix[i, :, :]

        max_value = max(cell_image.max(), cell_predict_image.max(), OCT_image.max())

        fig, axes = plt.subplots(1, 3, figsize=(24, 9))

        im0 = axes[0].imshow(OCT_image, cmap='viridis')
        axes[0].set_title(f'OCT (Index {i})')
        fig.colorbar(im0, ax=axes[0], orientation='vertical')

        im1 = axes[1].imshow(cell_image, cmap='viridis', vmin=0, vmax=max_value)
        axes[1].set_title(f'Cell (Index {i})')
        fig.colorbar(im1, ax=axes[1], orientation='vertical')

        im2 = axes[2].imshow(cell_predict_image, cmap='viridis', vmin=0, vmax=max_value)
        axes[2].set_title(f'Cell Predict (Index {i})')
        fig.colorbar(im2, ax=axes[2], orientation='vertical')

        output_path = os.path.join(output_dir, f'Image_Index_{i}.png')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved image to {output_path}")

        plt.show()

def load_data_and_display(data_directory, output_directory, indices):
    for file_name in sorted(os.listdir(data_directory)):
        if file_name.endswith('.h5'):
            print(f"Processing file: {file_name}")
            image_path = os.path.join(data_directory, file_name)
            OCT_matrix, cell_matrix, cell_predict_matrix = read_HDF5(image_path)
            display_and_save_images(OCT_matrix, cell_matrix, cell_predict_matrix, output_directory, indices)

# Ścieżka do katalogu
data_predict = 'C:/Python/Studia/U-NET/output/predict'
output_directory = 'C:/Python/Studia/U-NET/output/Image'

# Wybrane indeksy pierwszego wymiaru do wyświetlenia
selected_indices = [10, 20, 30]

# Wczytaj dane i wyświetl obrazy
load_data_and_display(data_predict, output_directory, selected_indices)
