import torch
#from torchvision.models import resnet18
import segmentation_models_pytorch as smp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import urllib.request
import os

# Definiere die URL und den Dateinamen für das Xception-Modell
url_xception = 'https://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
xception_filename = 'xception-43020ad28.pth'

# Definiere das lokale Verzeichnis, in dem du das Modell speichern möchtest
local_directory = './models/'  # Verwende dein gewünschtes lokales Verzeichnis
local_file_path = os.path.join(local_directory, xception_filename)

# Lade das Modell herunter und speichere es lokal
urllib.request.urlretrieve(url_xception, local_file_path)



"""
model = smp.Unet(
            encoder_name='inceptionv4',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1
        )
"""
"""
# Access the state dictionary containing model weights
model_weights = model.state_dict()

# Save the weights to a file
torch.save(model_weights, './models/inceptionv4_weights.pth')
"""