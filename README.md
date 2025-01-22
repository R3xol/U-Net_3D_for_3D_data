# U-Net 3D for 3D Data

The project implements a U-Net architecture for 3D data reconstruction using Python and PyTorch. It includes scripts for training the model, generating predictions, and directory structures storing data and model code.

## Struktura projektu

- **dataset/**             | Folder przechowujący dane do trenowania, walidacji oraz testów
- **pyimagesearch/**       | Zawiera implementację modelu U-Net 3D
  - `unet_3d.py`           | Główny skrypt modelu zawierający architekturę U-Net
  - `config.py`            | Plik konfiguracyjny z parametrami modelu i ścieżkami
- `train.py`               | Skrypt do trenowania modelu U-Net
- `predict.py`             | Skrypt do generowania prognoz

## Instalacja

Aby pobrać repozytorium na lokalny komputer, użyj poniższego polecenia:

```bash
git clone https://github.com/R3xol/U-Net_3D_for_3D_data.git

```powershell
py -3.11 -m venv venv

.\venv\Scripts\Activate

py -m pip install --upgrade pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt