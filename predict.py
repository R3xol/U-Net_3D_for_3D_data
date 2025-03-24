# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import os
from pyimagesearch.model import UNet3D
from sklearn.preprocessing import StandardScaler

# odczyt z hdf5
def read_HDF5(file_name):
    with h5py.File(file_name, 'r') as f:
        cell_matrix = f['Cell'][:]
        OCT_matrix = f['OCT'][:]
    return cell_matrix, OCT_matrix

# Zapis do jednego pliku HDF5
def save_to_H5_file(file_name,OCT, cell, predict):
	# Zamiana końcówki
	new_filename = file_name.replace("_converted.h5", "_predict.h5",)
	path_save = os.path.sep.join([config.PREDICT_PATHS, new_filename])
	with h5py.File(path_save, 'w') as f:
		f.create_dataset('OCT', data = OCT)
		f.create_dataset('Cell_base', data = cell) 
		f.create_dataset('Cell_predict', data = predict)
	print("[INFO] save down predict...")
	
def make_predictions(model, oct):


	# set model to evaluation mode
	model.eval()
	oct = torch.from_numpy(oct)#.to(config.DEVICE)
	oct = oct.unsqueeze(0)# Dodaj wymiar kanału: (1, D, H, W)
	oct = oct.unsqueeze(0)# Dodaj wymiar kanału: (1, 1, D, H, W)

	# make the prediction, pass the results through the relu
	# function, and convert the result to a NumPy array
	'''predImg = model(oct).squeeze()
	predImg = torch.relu(predImg)
	predImg = predImg.cpu().numpy()'''

	# Wykonaj predykcję
	with torch.no_grad():  # Wyłączenie obliczania gradientu dla predykcji
		predImg = model(oct).squeeze()  # Przekształć wyjście, usuń dodatkowe wymiary
		#predImg = torch.relu(predImg)  # Zastosuj funkcję aktywacji ReLU
    
    # Przenieś tensor na CPU (jeśli jest na GPU) i konwertuj na numpy
	predImg_numpy = predImg.numpy()  # Zamiana na numpy

	return predImg_numpy


'''file_names = os.listdir(config.DATASET_PATH_TRAIN)
trainImages = [f for f in file_names if os.path.isfile(os.path.join(config.DATASET_PATH_TRAIN, f))]'''

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
#unet = torch.load(config.MODEL_PATH)#.to(config.DEVICE)

unet = UNet3D()  # Stwórz instancję swojego modelu
unet.load_state_dict(torch.load(config.MODEL_PATH))  # Wczytaj state_dict MODEL_PATH/MODEL_IN_PROGRESS_PATH
# unet = unet.to(config.DEVICE)  # Jeśli używasz GPU, odkomentuj to

for i, file_name in enumerate(sorted(os.listdir(config.DATASET_PATH_PREDICT))):
	if file_name.endswith('.h5'):
		print(i, file_name)

		image_path = os.path.join(config.DATASET_PATH_PREDICT, file_name)

		cell_matrix, OCT_matrix = read_HDF5(image_path)

		predict_matrix = make_predictions(unet, OCT_matrix)

		# Powrót do oryginalnego zakresu danych
		'''cell_scaler = StandardScaler()
		cell_scaler.fit(cell_matrix.flatten().reshape(-1, 1))  # Fit on original data
		original_cell_shape = predict_matrix.shape
		predict_matrix = cell_scaler.inverse_transform(predict_matrix.flatten().reshape(-1, 1)).reshape(original_cell_shape)'''

		cell_matrix = np.float32(cell_matrix)

		save_to_H5_file(file_name, OCT_matrix, cell_matrix, predict_matrix)