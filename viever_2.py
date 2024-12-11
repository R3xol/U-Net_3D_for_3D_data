import torch
from torchviz import make_dot
from pyimagesearch.model import UNet3D

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz-12.2.1/bin/'

# Inicjalizacja modelu
model = UNet3D()

# Utworzenie przykładowego wejścia (batch size 1, 1 kanał, 64x128x128)
sample_input = torch.randn(1, 1, 64, 128, 128)

# Forward pass przez model
output = model(sample_input)

# Tworzenie wykresu modelu
dot = make_dot(output, params=dict(model.named_parameters()))

# Zapis grafu do pliku PDF
dot.render("UNet3D_architecture", format="pdf")