# import the necessary packages
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import h5py
import os
import torch
import numpy as np

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, data_Directory):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.data_Directory = data_Directory

		self.oct_scaler = None
		self.cell_scaler = None
		
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		file_name = self.imagePaths[idx]

		imagePath = os.path.join(self.data_Directory, file_name)

		# load the image from disk
		with h5py.File(imagePath, 'r') as f:
			cell = f['Cell'][:]
			oct = f['OCT'][:]

		cell = np.float32(cell)
		oct = np.float32(oct)	

		mean_X, std_X = oct.mean(), oct.std()

		# Avoid division by zero
		std_X = std_X if std_X > 0 else 1.0

		# Scale both OCT (X) and Cell (Y) using OCT's mean and std
		oct_scaled = (oct - mean_X) / std_X
		cell_scaled = (cell - mean_X) / std_X

		#####
		print("\n")
		print("OCT  - mean: {:15.11f}	std: {:15.11f}".format(oct_scaled.mean().item(), oct_scaled.std().item()))
		print("Cell - mean: {:15.11f}	std: {:15.11f}".format(cell_scaled.mean().item(), cell_scaled.std().item()))
		#####

		# Convert to torch tensors
		cell_scaled = torch.from_numpy(cell_scaled)
		oct_scaled = torch.from_numpy(oct_scaled)

		# Add dimension: (1, D, H, W)
		cell_scaled = cell_scaled.unsqueeze(0)  
		oct_scaled = oct_scaled.unsqueeze(0)

		return (oct_scaled, cell_scaled)
	
	def _MinMaxNormalization():
		# Min-Max normalization for cell and oct
		cell_min, cell_max = cell.min(), cell.max()
		oct_min, oct_max = oct.min(), oct.max()
        
        # Avoid division by zero in case of constant arrays
		if cell_max > cell_min:
			cell = (cell - cell_min) / (cell_max - cell_min)
		else:
			cell = np.zeros_like(cell)

		if oct_max > oct_min:
			oct = (oct - oct_min) / (oct_max - oct_min)
		else:
			oct = np.zeros_like(oct)
		
