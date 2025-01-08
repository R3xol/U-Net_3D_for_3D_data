# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet3D
model = UNet3D()
from pyimagesearch import config
from torch.nn import MSELoss #BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MeanSquaredError

# Metryki SSIM i PSNR
ssim_metric = StructuralSimilarityIndexMeasure()#.to(config.DEVICE)
psnr_metric = PeakSignalNoiseRatio()#.to(config.DEVICE)
mean_squared_error = MeanSquaredError()

'''# Path to the directory
data_directory_test = './dataset/test'
data_directory = './dataset/train'''

'''index, trainImages = enumerate(sorted(os.listdir(data_directory)))
index_test, testImages = enumerate(sorted(os.listdir(data_directory_test)))'''

testImages = []

file_names = os.listdir(config.DATASET_PATH_TRAIN)

trainImages = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_TRAIN, f))]

file_names = os.listdir(config.DATASET_PATH_TEST)
testImages = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_TEST, f))]

# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, data_Directory=config.DATASET_PATH_TRAIN)
testDS = SegmentationDataset(imagePaths=testImages, data_Directory=config.DATASET_PATH_TEST)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# Number of threads
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    nr_workers = torch.cuda.device_count() * 4
else:
    nr_workers = 0


# Utworzenie DataLoaderów
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
lossFunc = MSELoss() #BCEWithLogitsLoss()
opt = Adam(model.parameters(), lr=config.INIT_LR)
    
# Obliczenie liczby kroków na epokę
trainSteps = len(trainLoader)
testSteps = len(testLoader)

print(trainSteps)
print(testSteps)

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
early_stopping_patience = 10  # Liczba epok bez poprawy, po których zatrzymamy trening
early_stopping_counter = 0
best_test_loss = float("inf")

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5, verbose=True)
    
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
    SSIM.append(avgSSIM)
    PSNR.append(avgPSNR)
    MSE.append(avgMSE)
     
    # Informacje o postępie
    print(f"[INFO] EPOKA: {epoch + 1}/{config.NUM_EPOCHS}")
    print(f"Train Loss: {avgTrainLoss:.6f}, Test Loss: {avgTestLoss:.6f}")
    print(f"SSIM: {avgSSIM:.6f}, PSNR: {avgPSNR:.6f}, MSE: {avgMSE:.6f}")

    # Learning Rate Scheduling
    scheduler.step(avgTestLoss)

    # Early Stopping: Sprawdzenie, czy strata na danych testowych się poprawiła
    if avgTestLoss < best_test_loss:
        best_test_loss = avgTestLoss
        early_stopping_counter = 0  # Resetujemy licznik, bo mamy poprawę

        # Zapisanie modelu
        # print("[INFO] Zapisuję model do pliku...")
        torch.save(model.state_dict(), config.MODEL_IN_PROGRESS_PATH)

        print(f"[INFO] Najlepszy wynik test loss: {best_test_loss:.6f}")
    else:
        early_stopping_counter += 1
        print(f"[INFO] Brak poprawy test loss przez {early_stopping_counter} epok.")
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"[INFO] Early stopping aktywowany. Zatrzymujemy trening.")
            break
    
# Koniec trenowania
endTime = time.time()
print(f"[INFO] Całkowity czas trenowania: {endTime - startTime:.2f} sekundy")
    
# Zapisanie modelu
print("[INFO] Zapisuję model do pliku...")
torch.save(model.state_dict(), config.MODEL_PATH)
    
# Rysowanie wykresów strat
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Strata treningowa i walidacyjna")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.legend(loc="upper right")
plt.savefig(config.PLOT_PATH)

# Zapis danych procesu uczenia
PSNR = np.array(PSNR)
SSIM = np.array(SSIM)
MSE = np.array(MSE)
Test_loss = np.array(Test_loss)
Train_loss = np.array(Train_loss)

combined_matrix = np.array([
    ["MSE"] + MSE.tolist(),
    ["PSNR"] + PSNR.tolist(),
    ["SSIM"] + SSIM.tolist(),
    ["Test_loss"] + Test_loss.tolist(),
    ["Train_loss"] + Train_loss.tolist()
], dtype=object)

# Zapisanie macierzy jako CSV
np.savetxt(config.LERNING_RATE_PATHS, combined_matrix, fmt='%s', delimiter=',')

print(f"Macierz została zapisana do pliku: {config.LERNING_RATE_PATHS}")