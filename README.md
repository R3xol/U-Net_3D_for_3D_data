# U-Net 3D for 3D Data

The project implements a U-Net architecture for 3D data reconstruction using Python and PyTorch. It includes scripts for training the model, generating predictions, and directory structures storing data and model code.

## Project Structure

- **dataset/**             | Folder containing training, validation, and test data
- **pyimagesearch/**       | Contains the U-Net 3D model implementation
  - `unet_3d.py`           | Main model script containing the U-Net architecture
  - `config.py`            | Configuration file with model parameters and paths
- `train.py`               | Script for training the U-Net model
- `predict.py`             | Script for generating predictions

## Installation

```bash
git clone https://github.com/R3xol/U-Net_3D_for_3D_data.git
```


```bash
py -3.11 -m venv venv

.\venv\Scripts\Activate

py -m pip install --upgrade pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
