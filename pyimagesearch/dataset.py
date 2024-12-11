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

		cell = torch.from_numpy(cell)
		oct = torch.from_numpy(oct)

		cell = cell.unsqueeze(0)  # Dodaj wymiar kanału: (1, D, H, W)
		oct = oct.unsqueeze(0)

		# return a tuple of the image and its mask
		return (cell, oct)