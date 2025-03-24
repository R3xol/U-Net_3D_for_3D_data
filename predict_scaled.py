# python predict.py
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import os
import joblib
from pyimagesearch.model import UNet3D
from sklearn.preprocessing import StandardScaler

# Ścieżka do folderu z skalerami
SCALERS_PATH = "./scalery all"

# Wczytaj scalery
def load_scalers():
    oct_scaler_path = os.path.join(SCALERS_PATH, 'oct_scaler.joblib')
    cell_scaler_path = os.path.join(SCALERS_PATH, 'cell_scaler.joblib')
    
    if not os.path.exists(oct_scaler_path):
        raise FileNotFoundError(f"Nie znaleziono pliku oct_scaler w {SCALERS_PATH}")
    if not os.path.exists(cell_scaler_path):
        raise FileNotFoundError(f"Nie znaleziono pliku cell_scaler w {SCALERS_PATH}")
    
    print("[INFO] Wczytano scalery z:", SCALERS_PATH)
    return joblib.load(oct_scaler_path), joblib.load(cell_scaler_path)

# odczyt z hdf5 z automatycznym skalowaniem
def read_HDF5(file_name, oct_scaler):
    with h5py.File(file_name, 'r') as f:
        cell_matrix = np.float32(f['Cell'][:])
        OCT_matrix = np.float32(f['OCT'][:])
        
        # Skaluj dane OCT
        original_shape = OCT_matrix.shape
        OCT_matrix = oct_scaler.transform(OCT_matrix.reshape(-1, 1)).reshape(original_shape)
        
    return cell_matrix, OCT_matrix

# Zapis do pliku HDF5 z odwróconym skalowaniem
def save_to_H5_file(file_name, OCT, cell, predict, cell_scaler):
    new_filename = file_name.replace(".h5", "_predict.h5")
    path_save = os.path.sep.join([config.PREDICT_PATHS, new_filename])
    
    # Odwróć skalowanie predykcji
    original_shape = predict.shape
    predict = cell_scaler.inverse_transform(predict.reshape(-1, 1)).reshape(original_shape)
    
    with h5py.File(path_save, 'w') as f:
        f.create_dataset('OCT', data=OCT)
        f.create_dataset('Cell_base', data=cell) 
        f.create_dataset('Cell_predict', data=predict)
    print(f"[INFO] Zapisano predykcję: {new_filename}")

def make_predictions(model, oct_tensor):
    model.eval()
    with torch.no_grad():
        oct_tensor = oct_tensor.to(config.DEVICE)
        predImg = model(oct_tensor).squeeze()
        if config.DEVICE == 'cuda':
            predImg = predImg.cpu()
    return predImg.numpy()

def main():
    # Sprawdź czy folder z skalerami istnieje
    if not os.path.exists(SCALERS_PATH):
        print(f"[ERROR] Folder ze skalerami '{SCALERS_PATH}' nie istnieje!")
        return

    # Wczytaj scalery
    try:
        oct_scaler, cell_scaler = load_scalers()
    except Exception as e:
        print(f"[ERROR] Błąd wczytywania scalerów: {str(e)}")
        return

    # Wczytaj model
    print("[INFO] Wczytywanie modelu...")
    try:
        unet = UNet3D()
        unet.load_state_dict(torch.load(config.MODEL_PATH))
        unet = unet.to(config.DEVICE)
        print("[INFO] Model załadowany pomyślnie")
    except Exception as e:
        print(f"[ERROR] Błąd wczytywania modelu: {str(e)}")
        return

    # Utwórz folder na wyniki jeśli nie istnieje
    os.makedirs(config.PREDICT_PATHS, exist_ok=True)

    # Przetwarzaj pliki predykcyjne
    for file_name in sorted(f for f in os.listdir(config.DATASET_PATH_PREDICT) if f.endswith('.h5')):
        print(f"\n[INFO] Przetwarzanie pliku: {file_name}")
        
        try:
            # Wczytaj i przeskaluj dane
            file_path = os.path.join(config.DATASET_PATH_PREDICT, file_name)
            cell_matrix, OCT_matrix = read_HDF5(file_path, oct_scaler)
            
            # Przygotuj tensor wejściowy
            oct_tensor = torch.from_numpy(OCT_matrix).unsqueeze(0).unsqueeze(0).float()
            
            # Wykonaj predykcję
            predict_matrix = make_predictions(unet, oct_tensor)
            
            # Zapisz wyniki
            save_to_H5_file(file_name, OCT_matrix, cell_matrix, predict_matrix, cell_scaler)
            
        except Exception as e:
            print(f"[ERROR] Błąd przetwarzania pliku {file_name}: {str(e)}")
            continue

    print("\n[INFO] Zakończono przetwarzanie wszystkich plików")

if __name__ == '__main__':
    main()