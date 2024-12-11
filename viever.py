import torch
from torchviz import make_dot
from torch.nn import Module
from pyimagesearch.model import UNet3D
from pyimagesearch import config

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz-12.2.1/bin/'

# Inicjalizacja modelu
model = UNet3D()

# Utworzenie przykładowego wejścia
sample_input = torch.randn(1, 1, 60, 240, 240)

# Przekazanie danych przez model
output = model(sample_input)

# Stworzenie wizualizacji modelu
dot = make_dot(output, params=dict(model.named_parameters()))

# Zapis grafu do pliku PNG
dot.format = 'png'

dot.render(os.path.sep.join([config.BASE_OUTPUT, 'UNet3D_architecture']))

print("Wizualizacja modelu została zapisana jako 'UNet3D_architecture.png'.")