# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import os

# Zapis do jednego pliku HDF5
def save_to_H5_file(file_name, rescaled_matrix_cell, real_part_inverse_fourier_transform):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('Cell', data = rescaled_matrix_cell) 
        f.create_dataset('OCT', data = real_part_inverse_fourier_transform)
	
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()

	imagePath = os.path.join(self.data_Directory, file_name)

	# load the image from disk
	with h5py.File(imagePath, 'r') as f:
		cell = f['Cell'][:]
		oct = f['OCT'][:]

	cell = np.float32(cell)

	cell = torch.from_numpy(cell)#.to(config.DEVICE)
	oct = torch.from_numpy(oct)#.to(config.DEVICE)

	cell = cell.unsqueeze(0)  # Dodaj wymiar kanału: (1, D, H, W)
	oct = oct.unsqueeze(0)

	# make the prediction, pass the results through the relu
	# function, and convert the result to a NumPy array
	predImg = model(oct).squeeze()
	predImg = torch.relu(predImg)
	predImg = predImg.cpu().numpy()

	save_to_H5_file(predImg)	

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)