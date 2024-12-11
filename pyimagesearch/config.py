# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = os.path.join("dataset", "train")
DATASET_PATH_TEST = os.path.join("dataset", "test")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# define the input image dimensions
INPUT_IMAGE_WIDTH = 240
INPUT_IMAGE_HEIGHT = 240
INPUT_IMAGE_LENGHT = 60
# 60 240 240

# define the path to the base output directory
BASE_OUTPUT = "./output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "unet3d.pth"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "training_plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

BATCH_SIZE = 4  # Rozmiar batcha
INIT_LR = 1e-3  # Początkowa wartość learning rate
NUM_EPOCHS = 10  # Liczba epok

PIN_MEMORY = True  # Wspomaganie pamięci dla GPU

NUM_WORKERS = 1  # Liczba wątków dla DataLoader


