import os
import h5py
import numpy as np

# Path to the directory
data_predict = 'C:/Python/Studia/U-NET/output/predict'

# Function to read HDF5 files
def read_HDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        cell_matrix = f['Cell_base'][:]
        predict_matrix = f['Cell_predict'][:]
    return cell_matrix, predict_matrix

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(matrix1, matrix2):
    return np.mean((matrix1 - matrix2) ** 2)

# Main function to process files
def process_h5_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    mse_results = {}

    for i, file_name in enumerate(files, start=1):
        file_path = os.path.join(directory, file_name)
        
        # Read matrices from file
        cell_matrix, predict_matrix = read_HDF5(file_path)

        # Calculate MSE
        mse = calculate_mse(cell_matrix, predict_matrix)
        mse_results[file_name] = mse

        # Dynamically create variable names
        globals()[f'cell_b_{i}'] = cell_matrix
        globals()[f'cell_p_{i}'] = predict_matrix

        print(f"File: {file_name}, MSE: {mse}")

    return mse_results

if __name__ == "__main__":
    try:
        mse_results = process_h5_files(data_predict)
        print("MSE Results:", mse_results)

        print(cell_b_1.min(), cell_b_1.max())
        cell_b_1 = cell_b_1[29, 100, :]
        print(cell_b_1)

        cell_p_1 = cell_p_1[29, 100, :]
        cell_p_1 = np.int8(cell_p_1)
        print(cell_p_1)

    except Exception as e:
        print(f"An error occurred: {e}")
