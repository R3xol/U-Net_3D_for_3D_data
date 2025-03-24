import os
import numpy as np
import h5py
import joblib
from sklearn.preprocessing import StandardScaler
from pyimagesearch import config
from tqdm import tqdm

class BatchScaler:
    """Przetwarza dane katalog po katalogu"""
    def __init__(self):
        self.oct_scaler = StandardScaler()
        self.cell_scaler = StandardScaler()
        self.oct_scaler.partial_fit(np.zeros((1, 1)))  # Inicjalizacja
        self.cell_scaler.partial_fit(np.zeros((1, 1))) # Inicjalizacja
    
    def process_directory(self, dir_path):
        """Przetwarza wszystkie pliki w jednym katalogu"""
        files = [f for f in os.listdir(dir_path) if f.endswith('.h5')]
        if not files:
            print(f"[WARNING] Brak plików .h5 w {dir_path}")
            return
        
        print(f"[INFO] Przetwarzanie {len(files)} plików w {os.path.basename(dir_path)}...")
        
        for file_name in tqdm(files):
            file_path = os.path.join(dir_path, file_name)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Ładujemy cały plik na raz (60x240x240 ~ 21MB)
                    oct_data = np.float32(f['OCT'][:]).reshape(-1, 1)
                    cell_data = np.float32(f['Cell'][:]).reshape(-1, 1)
                    
                    # Przyrostowe dopasowanie
                    self.oct_scaler.partial_fit(oct_data)
                    self.cell_scaler.partial_fit(cell_data)
                    
            except Exception as e:
                print(f"[ERROR] Błąd przetwarzania {file_name}: {str(e)}")
                continue

def process_all_directories(base_path):
    """Przetwarza wszystkie katalogi po kolei"""
    scaler = BatchScaler()
    
    # Znajdź i posortuj katalogi
    data_dirs = sorted([
        d for d in os.listdir(base_path)
        if d.startswith('Mitochondrium_') and d.endswith('_policzone')
        and os.path.isdir(os.path.join(base_path, d))
    ])
    
    if not data_dirs:
        raise ValueError(f"Nie znaleziono katalogów w {base_path}")
    
    print(f"[INFO] Znaleziono {len(data_dirs)} katalogów do przetworzenia")
    print(data_dirs)
    
    for dir_name in data_dirs:
        dir_path = os.path.join(base_path, dir_name)
        scaler.process_directory(dir_path)
        
        # Informacje o zużyciu pamięci
        print_memory_usage()
    
    return scaler.oct_scaler, scaler.cell_scaler

def print_memory_usage():
    """Wyświetla informacje o zużyciu pamięci"""
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # w MB
    print(f"[MEMORY] Użycie pamięci: {mem:.2f} MB")

def save_scalers(oct_scaler, cell_scaler, output_dir):
    """Zapisuje scalery do plików"""
    os.makedirs(output_dir, exist_ok=True)
    
    oct_path = os.path.join(output_dir, 'oct_scaler.joblib')
    cell_path = os.path.join(output_dir, 'cell_scaler.joblib')
    
    joblib.dump(oct_scaler, oct_path)
    joblib.dump(cell_scaler, cell_path)
    
    print(f"\n[SUKCES] Zapisano scalery:")
    print(f" - OCT: {oct_path}")
    print(f" - Cell: {cell_path}")

if __name__ == '__main__':
    DATA_BASE_PATH = r"D:\Praca_Magisterska_Dane"
    
    try:
        print("[INFO] Rozpoczynanie przetwarzania katalogów...")
        oct_scaler, cell_scaler = process_all_directories(DATA_BASE_PATH)
        
        print("\n[INFO] Zakończono obliczanie skalera. Zapis wyników...")
        save_scalers(oct_scaler, cell_scaler, config.BASE_OUTPUT)
        
    except Exception as e:
        print(f"\n[BŁĄD] {str(e)}")
        print("Przetwarzanie przerwane")