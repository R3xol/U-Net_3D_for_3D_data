# import the necessary packages
from torch.utils.data import Dataset
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

		'''# Min-Max normalization for cell and oct
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
			oct = np.zeros_like(oct)'''		

		# Convert to torch tensors
		cell = torch.from_numpy(cell)
		oct = torch.from_numpy(oct)

		# Dodaj wymiar kanału: (1, D, H, W)
		cell = cell.unsqueeze(0)  
		oct = oct.unsqueeze(0)

		# return a tuple of the image and its mask
		return (oct, cell)