import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from patchify import patchify, unpatchify
import segmentation_models_pytorch.utils as smp_utils
import csv
import segmentation_models_pytorch as smp
from my_utils import predict_mask, predict_mask_ensemble
import argparse
import os
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'


# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Perform prediction on a large image.')
parser.add_argument('--config', type=str, help='Path to the configuration JSON file.')
args = parser.parse_args()



# Load configuration from JSON file
with open(args.config, 'r') as config_file:
    config = json.load(config_file)
    
output_directory = config['output_directory'].format(**config)

# Create the directory if it doesn't exist
os.makedirs('predicted_treated_data/' + output_directory, exist_ok=True)

# Model architecture configuration
model_config = config["model"]
model_name = model_config["name"]
encoder_name = model_config["encoder_name"]

from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')


# Load the model checkpoint
model_checkpoint_path = f'./fine-tuning/{output_directory}/best_model.pth'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_checkpoint_path, map_location=device) # needs to be loaded with 4 devices if training was done on 4 devices!

# Set the model to evaluation mode
model.eval()


input_directory = "../mitochondria/treated_uniklinik_scaled/images"
for file_name in os.listdir(input_directory):
    input_image_path = os.path.join(input_directory, file_name)

    treated_predicted_mask, treated_image_for_plot = predict_mask(input_image_path, model, preprocess_input, device)


    """
    #----------------------------------------------------------------------------------------------------------------------------------------
    # ENSEMBLE
    models = []
    model_checkpoint_paths = ['./fine-tuning/Adam/DiceFocalLoss/Unet_resnet34/Unet_resnet34_100epochs_bs4_lr_0.0001/best_model.pth',
                              './fine-tuning/Adam/DiceFocalLoss/Unet_xception/Unet_xception_100epochs_bs4_lr_0.0001/best_model.pth',
                              './fine-tuning/Adam/DiceFocalLoss/Unet_vgg16/Unet_vgg16_100epochs_bs4_lr_0.0001/best_model.pth',
                              './fine-tuning/Adam/DiceCELoss/Unet_resnet34/Unet_resnet34_100epochs_bs4_lr_0.0001/best_model.pth',
                              './fine-tuning/Adam/DiceCELoss/Unet_resnet50/Unet_resnet50_100epochs_bs4_lr_0.0001/best_model.pth']
    for model_path in model_checkpoint_paths:
        model = torch.load(model_path, map_location=device) # needs to be loaded with 4 devices if training was done on 4 devices!
        models.append(model)
        
    treated_predicted_mask, treated_image_for_plot = predict_mask_ensemble(input_image_path, models, [1.0, 0.0, 0.0, 0.0, 0.0], preprocess_input, device)
    print(f"predicted {file_name}")

    #-----------------------------------------------------------------------------------------------------------------------------------------
    """

    # Threshold the predicted mask
    treated_predicted_mask_binary = (treated_predicted_mask > 0.5).astype(np.uint8)


    # Plotting the original image and predicted mask
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(treated_image_for_plot)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(treated_predicted_mask_binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout(pad=2)
    plt.savefig(f'./predicted_treated_data/{output_directory}/test_{file_name[:-4]}.png')




