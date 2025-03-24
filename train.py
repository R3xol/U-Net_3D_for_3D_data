# dodać RandomSampler

# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet3D
from pyimagesearch.model import RMSELoss
model = UNet3D() 
from pyimagesearch import config
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np

import signal
import sys

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler 

import h5py
from sklearn.preprocessing import StandardScaler
import joblib

# Wczytanie danyc z katalogów test i train
def load_data_from_train_test():
    testImages = []

    file_names = os.listdir(config.DATASET_PATH_TRAIN)

    trainImages = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_TRAIN, f))]

    file_names = os.listdir(config.DATASET_PATH_TEST)
    testImages = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_TEST, f))]

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, data_Directory=config.DATASET_PATH_TRAIN)
    testDS = SegmentationDataset(imagePaths=testImages, data_Directory=config.DATASET_PATH_TEST)
    return trainDS, testDS

# Wczytanie danyc z katalogu all
def load_data_from_all(split):
    DS = []

    file_names = os.listdir(config.DATASET_PATH_ALL)

    DS = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_ALL, f))]

    DS = SegmentationDataset(imagePaths=DS, data_Directory=config.DATASET_PATH_ALL)

    # Ustawienie ziarna dla powtarzalności
    torch.manual_seed(13)

    # Podział zbioru danych na treningowy 
    train_size = int(split * len(DS))  
    test_size = len(DS) - train_size  
    trainDS, testDS = random_split(DS, [train_size, test_size])
    return trainDS, testDS

# Wczytanie danyc z katalogu all
'''def load_data_from_all_normalized(split):
    DS = []

    file_names = os.listdir(config.DATASET_PATH_ALL)

    DS = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_ALL, f))]

    # Oblicz statystyki skalowania
    oct_scaler, cell_scaler = compute_dataset_stats(config.DATASET_PATH_ALL)

    # Zapisz scalery do plików
    joblib.dump(oct_scaler, os.path.join(config.BASE_OUTPUT, 'oct_scaler.joblib'))
    joblib.dump(cell_scaler, os.path.join(config.BASE_OUTPUT, 'cell_scaler.joblib'))
    print(f"[INFO] Zapisano scalery do: {config.BASE_OUTPUT}")

    # Utwórz dataset z przekazanymi skalerami
    DS = SegmentationDataset(imagePaths=DS, data_Directory=config.DATASET_PATH_ALL,
                           oct_scaler=oct_scaler, cell_scaler=cell_scaler)

    # Ustawienie ziarna dla powtarzalności
    torch.manual_seed(13)

    # Podział zbioru danych na treningowy 
    train_size = int(split * len(DS))  
    test_size = len(DS) - train_size  
    trainDS, testDS = random_split(DS, [train_size, test_size])
    return trainDS, testDS'''

def load_data_from_all_normalized(split, use_existing_scalers=True):
    """Wczytuje i normalizuje dane, używając istniejących lub nowych scalerów."""
    # Utwórz folder wyjściowy jeśli nie istnieje
    os.makedirs(config.BASE_OUTPUT, exist_ok=True)
    
    file_names = [f for f in os.listdir(config.DATASET_PATH_ALL) 
                 if os.path.isfile(os.path.join(config.DATASET_PATH_ALL, f))]
    
    oct_scaler_path = os.path.join(config.BASE_OUTPUT, 'oct_scaler.joblib')
    cell_scaler_path = os.path.join(config.BASE_OUTPUT, 'cell_scaler.joblib')
    
    try:
        if use_existing_scalers and os.path.exists(oct_scaler_path) and os.path.exists(cell_scaler_path):
            oct_scaler, cell_scaler = load_scalers(oct_scaler_path, cell_scaler_path)
            print("[INFO] Użyto istniejących scalerów")
        else:
            oct_scaler, cell_scaler = compute_dataset_stats(config.DATASET_PATH_ALL)
            joblib.dump(oct_scaler, oct_scaler_path)
            joblib.dump(cell_scaler, cell_scaler_path)
            print(f"[INFO] Obliczono i zapisano nowe scalery w: {config.BASE_OUTPUT}")
    except Exception as e:
        print(f"[ERROR] Błąd przetwarzania scalerów: {str(e)}")
        raise
    
    # Utwórz dataset z przekazanymi skalerami
    DS = SegmentationDataset(
        imagePaths=file_names,
        data_Directory=config.DATASET_PATH_ALL,
        oct_scaler=oct_scaler,
        cell_scaler=cell_scaler
    )

    # Podział zbioru danych
    torch.manual_seed(13)  # Dla powtarzalności
    train_size = int(split * len(DS))  
    test_size = len(DS) - train_size  
    trainDS, testDS = random_split(DS, [train_size, test_size])
    
    return trainDS, testDS

def compute_dataset_stats(data_dir):
    oct_data = []
    cell_data = []

    # Przejdź przez wszystkie pliki w katalogu
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        
        with h5py.File(file_path, 'r') as f:
            oct = np.float32(f['OCT'][:])
            cell = np.float32(f['Cell'][:])
            
            oct_data.append(oct.flatten())  # Spłaszczamy, aby obliczyć statystyki globalne
            cell_data.append(cell.flatten())

    # Oblicz średnią i odchylenie standardowe dla OCT i Cell
    oct_scaler = StandardScaler()
    oct_scaler.fit(np.concatenate(oct_data).reshape(-1, 1))  # Musi być kształt (n_samples, 1)

    cell_scaler = StandardScaler()
    cell_scaler.fit(np.concatenate(cell_data).reshape(-1, 1))

    return oct_scaler, cell_scaler

def load_scalers(oct_scaler_path=None, cell_scaler_path=None):
    """Wczytuje zapisane scalery z plików."""
    # Ustaw domyślne ścieżki jeśli nie podano
    if oct_scaler_path is None:
        oct_scaler_path = os.path.join(config.SCALET_INPUT, 'oct_scaler.joblib')
    if cell_scaler_path is None:
        cell_scaler_path = os.path.join(config.SCALET_INPUT, 'cell_scaler.joblib')
    
    # Sprawdź czy pliki istnieją
    if not os.path.exists(oct_scaler_path):
        raise FileNotFoundError(f"Nie znaleziono pliku oct_scaler: {oct_scaler_path}")
    if not os.path.exists(cell_scaler_path):
        raise FileNotFoundError(f"Nie znaleziono pliku cell_scaler: {cell_scaler_path}")
    
    # Wczytaj scalery
    oct_scaler = joblib.load(oct_scaler_path)
    cell_scaler = joblib.load(cell_scaler_path)
    
    print("[INFO] Wczytano scalery")
    return oct_scaler, cell_scaler

if __name__ == '__main__':
    # czy stosować SubsetRandomSampler jesli tak to 1 nie to 0
    SRS = 0

    # Metryki SSIM i PSNR
    ssim_metric = StructuralSimilarityIndexMeasure().to(config.DEVICE)
    psnr_metric = PeakSignalNoiseRatio().to(config.DEVICE)
    mean_squared_error = MeanSquaredError().to(config.DEVICE)

    # Funkcja, która pozwala na przerwanie pętli z zewnątrz (np. przez Ctrl+C)
    def handle_interrupt(signal, frame):
        print("\n[INFO] Trening przerwany przez użytkownika.")
        raise KeyboardInterrupt  # Podnosi wyjątek, aby przerwać pętlę, ale kontynuować kod za pętlą

    try:
        trainDS, testDS = load_data_from_all_normalized(0.7, use_existing_scalers=True)
        print(f"Utworzono zbiór treningowy: {len(trainDS)} próbek")
        print(f"Utworzono zbiór testowy: {len(testDS)} próbek")
    except Exception as e:
        print(f"Nie udało się wczytać danych: {str(e)}")
        sys.exit(1)  # Zakończ program z kodem błędu

    # Ograniczenie liczby danych o połowę
    '''train_indices = list(range(len(trainDS)))[:len(trainDS) // 2]
    test_indices = list(range(len(testDS)))[:len(testDS) // 2]

    trainDS = Subset(trainDS, train_indices)
    testDS = Subset(testDS, test_indices)'''


    # Dodanie SubsetRandomSampler (miesza dane i wybiera tylko część do nauki co epokę inne)
    # Liczba próbek do treningu w każdej epoce
    if SRS == 1:
        print("[INFO] SubsetRandomSampler aktywny")
        sample_size = 1000

        torch.manual_seed(13)
        # Generowanie losowych indeksów dla podzbioru
        random_indices = torch.randperm(len(trainDS))[:sample_size]

        # Tworzenie SubsetRandomSampler
        train_sampler = SubsetRandomSampler(random_indices)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # Number of threads
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        nr_workers = torch.cuda.device_count() * 4
    else:
        nr_workers = 0

    print("Device:            ",device)
    print("Number of threads: ", nr_workers)

    # Utworzenie DataLoaderów
    # Utworzenie DataLoader z SubsetRandomSampler
    if SRS == 1:
        print("[INFO] SubsetRandomSampler aktywny")
        trainLoader = DataLoader(trainDS, sampler=train_sampler,
                                batch_size=config.BATCH_SIZE,
                                pin_memory=config.PIN_MEMORY,
                                num_workers=nr_workers)
    else:
        print("[INFO] Trening bez SubsetRandomSampler")
        trainLoader = DataLoader(trainDS, shuffle=True,
                                    batch_size=config.BATCH_SIZE,
                                    pin_memory=config.PIN_MEMORY,
                                    num_workers=nr_workers)
    
    testLoader = DataLoader(testDS, shuffle=False,
                                batch_size=config.BATCH_SIZE,
                                pin_memory=config.PIN_MEMORY,
                                num_workers=nr_workers)
        
    model = model.to(config.DEVICE)
        
    # Inicjalizacja funkcji straty i optymalizatora
    #lossFunc = MSELoss()
    lossFunc = RMSELoss()

    opt = Adam(model.parameters(), lr=config.INIT_LR, weight_decay=0.0001)
        
    # Obliczenie liczby kroków na epokę
    trainSteps = len(trainLoader)
    testSteps = len(testLoader)

    print(f"[INFO] Liczba kroków na epoke (train): {trainSteps}")
    print(f"[INFO] Liczba kroków na epoke (test): {testSteps}")

    print("\n")

    # Inicjalizacja słownika
    H = {"train_loss": [], "test_loss": [], "ssim": [], "psnr": []}
    Test_loss = []
    Train_loss = []
    SSIM = []
    PSNR = []
    MSE = []
        
    print("[INFO] Rozpoczynam trenowanie modelu...")
    startTime = time.time()

    # Inicjalizacja EarlyStopping
    early_stopping_patience = 30  # Liczba epok bez poprawy, po których zatrzymamy trening
    best_test_loss = float("inf")

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=20, factor=0.5)

    # Rejestracja sygnału przerwania (Ctrl + C)
    signal.signal(signal.SIGINT, handle_interrupt)

    cnt_epoch = 0

    try:
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            totalTrainLoss = 0
            totalTestLoss = 0
            totalSSIM = 0
            totalPSNR = 0
            totalMSE = 0
                
            # Trenowanie modelu
            print(f"[INFO] Epoka {epoch + 1}/{config.NUM_EPOCHS}")
            
            for (x, y) in tqdm(trainLoader):
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))           
                # Obliczanie prognoz i straty
                pred = model(x)
                loss = lossFunc(pred, y)
                    
                # Backpropagation i aktualizacja wag
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                totalTrainLoss += loss.item()
                
            # Walidacja modelu
            model.eval()
            with torch.no_grad():
                for (x, y) in testLoader:
                    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                    pred = model(x)
                    loss = lossFunc(pred, y)
                    totalTestLoss += loss.item()

                    # Obliczanie metryk SSIM i PSNR
                    totalSSIM += ssim_metric(pred, y)
                    totalPSNR += psnr_metric(pred, y)
                    totalMSE += mean_squared_error(pred, y)
                
            # Obliczenie średnich strat dla epoki
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps
            avgSSIM = totalSSIM / testSteps
            avgPSNR = totalPSNR / testSteps
            avgMSE = totalMSE / testSteps
                
            H["train_loss"].append(avgTrainLoss)
            H["test_loss"].append(avgTestLoss)
            H["ssim"].append(avgSSIM.item())
            H["psnr"].append(avgPSNR.item())

            Train_loss.append(avgTrainLoss)
            Test_loss.append(avgTestLoss)
            '''SSIM.append(avgSSIM)
            PSNR.append(avgPSNR)
            MSE.append(avgMSE)'''
            SSIM.append(avgSSIM.cpu().numpy())  # Przeniesienie do CPU przed konwersją
            PSNR.append(avgPSNR.cpu().numpy())
            MSE.append(avgMSE.cpu().numpy())

            cnt_epoch = cnt_epoch + 1
            
            # Informacje o postępie
            print(f"[INFO] EPOKA: {epoch + 1}/{config.NUM_EPOCHS}")
            print(f"Train Loss: {avgTrainLoss:.6f}, Test Loss: {avgTestLoss:.6f}")
            print(f"SSIM: {avgSSIM:.6f}, PSNR: {avgPSNR:.6f}, MSE: {avgMSE:.6f}")

            # Learning Rate Scheduling
            scheduler.step(avgTestLoss)  #tutaj musi być avgTestLoss Train tylko do testów
            print(f"[INFO] Aktualny współczynnik uczenia: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

            # Early Stopping: Sprawdzenie, czy strata na danych testowych się poprawiła
            if avgTestLoss < best_test_loss:
                best_test_loss = avgTestLoss
                early_stopping_counter = 0  # Resetujemy licznik, bo mamy poprawę

                # Zapisanie modelu
                print(f"[INFO] Zapisuję najlepszy model...")
                torch.save(model.state_dict(), config.MODEL_IN_PROGRESS_PATH)

                print(f"[INFO] Najlepszy wynik test loss: {best_test_loss:.6f}")
            else:
                early_stopping_counter += 1
                print(f"[INFO] Brak poprawy test loss przez {early_stopping_counter} epok.")
                
                if early_stopping_counter >= early_stopping_patience:
                    print(f"[INFO] Early stopping aktywowany. Zatrzymujemy trening.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Trening zakończony przez użytkownika.")
        # Zamknięcie DataLoaderów
        if hasattr(trainLoader.dataset, 'close'):
            trainLoader.dataset.close()
        if hasattr(testLoader.dataset, 'close'):
            testLoader.dataset.close()
        # Zapisanie modelu przed wyjściem
        print(f"[INFO] Zapisuję model po zatrzymaniu...")
        torch.save(model.state_dict(), config.MODEL_OUTPUT)
        # Zatrzymanie pętli po naciśnięciu Ctrl+C

    print('\n')

    # Koniec trenowania
    endTime = time.time()
    print(f"[INFO] Całkowity czas trenowania: {endTime - startTime:.2f} sekundy")
        
    # Zapisanie modelu
    print("[INFO] Zapisuję model do pliku...")
    torch.save(model.state_dict(), config.MODEL_PATH)

    Test_loss = np.array(Test_loss)
    Train_loss = np.array(Train_loss)
    SSIM = np.array(SSIM)

    # Rysowanie wykresów strat
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(Train_loss, label="train_loss")
    plt.plot(Test_loss, label="test_loss")
    #plt.plot(SSIM, label="SSMI")
    plt.title("Strata treningowa i walidacyjna")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)

    # Zapis danych procesu uczenia
    PSNR = np.array(PSNR)
    MSE = np.array(MSE)

    combined_matrix = np.array([
        ["MSE"] + MSE.tolist(),
        ["PSNR"] + PSNR.tolist(),
        ["SSIM"] + SSIM.tolist(),
        ["Test_loss"] + Test_loss.tolist(),
        ["Train_loss"] + Train_loss.tolist()
    ], dtype=object)

    # Zapisanie macierzy jako CSV
    np.savetxt(config.LERNING_RATE_PATHS, combined_matrix, fmt='%s', delimiter=',')

    print(f"[INFO] Macierz została zapisana do pliku: {config.LERNING_RATE_PATHS}")
    print("[INFO] Zakończono działanie <3")

