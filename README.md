# U-Net_3D_for_3D_data

- **dataset/**             | Folder for training, validation, and test data
- **pyimagesearch/**       | Contains the U-Net 3D model implementation
  - `unet_3d.py`           | Main model script containing U-Net architecture
  - `config.py`            | Configuration file for model parameters and paths
- `train.py`               | Script to train the U-Net model
- `predict.py`             | Script to make predictions

`py -3.11 -m venv venv`
`.\venv\Scripts\Activate`
`py -m pip install --upgrade pip `
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
`pip install -r requirements.txt`

