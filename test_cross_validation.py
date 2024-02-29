import torch
from pprint import pprint
import pandas as pd
import numpy as np
#import monai
from monai.data import Dataset, DataLoader
from monai.transforms import ToTensor, Compose
from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
import segmentation_models_pytorch.utils as smp_utils
import monai.losses as monai_losses
from monai.losses import DiceLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'
from my_utils import TrainEpoch, ValidEpoch, NamedLoss, get_cmap
from custom_transform import CustomTransform, ConvertToMultiChannelMask, MaskToRGB
import torchmetrics


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import json
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

# Load configuration from JSON file
with open('finetune-config_multiclass1.json', 'r') as config_file:
    config = json.load(config_file)

"""
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Perform finetuning on a large image.')
parser.add_argument('--config', type=str, help='Path to the configuration JSON file.')
args = parser.parse_args()



# Load configuration from JSON file
with open(args.config, 'r') as config_file:
    config = json.load(config_file)
    
"""
config['model']['encoder_name'] = 'vgg16'
config['loss']['name'] = 'DiceFocalLoss'

# Access configuration parameters
data_dir = config["data_dir"]
images_dir = os.path.join(data_dir, config["images_dir"])
masks_dir = os.path.join(data_dir, config["masks_dir"])
val_images_dir = os.path.join(data_dir, "val/images")
val_masks_dir = os.path.join(data_dir, "val/masks")
train_size = config["train_size"]
val_size = config["val_size"]
test_size = config["test_size"]
batch_size_train = config["batch_size"]["train"]
batch_size_val = config["batch_size"]["val"]
batch_size_test = config["batch_size"]["test"]
num_workers_train = config["num_workers"]["train"]
num_workers_val = config["num_workers"]["val"]
num_workers_test = config["num_workers"]["test"]
spatial_size = config["spatial_size"]
pos = config["pos"]
neg = config["neg"]
num_samples = config["num_samples"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

# Model architecture configuration
model_config = config["model"]
model_name = model_config["name"]
encoder_name = model_config["encoder_name"]
encoder_weights = model_config["encoder_weights"]
in_channels = model_config["in_channels"]
classes = model_config["classes"]

# Loss function configuration
loss_config = config["loss"]
loss_name = loss_config["name"]

# Optimizer configuration
optimizer_config = config["optimizer"]
optimizer_name = optimizer_config["name"]


# Define the output directory based on the configuration
output_directory = config["output_directory"].format(**config)

# Create the directory if it doesn't exist
os.makedirs('multi_class_cross_validation/' + output_directory, exist_ok=True)



from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')

# Set a new temporary directory
new_tmpdir = '/misc/lmbraid21/nasica/tmp'  # Replace this with your desired temporary directory

# Set the environment variable
os.environ['TMPDIR'] = ''
os.environ['TEMP'] = ''

from multiprocessing import Manager
mp = Manager()
mp.shutdown()


#-----------------------------------------------------------------------------------------------------------------------
# Dataloading and Preprocessing
#-----------------------------------------------------------------------------------------------------------------------

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAdjustContrastd,
    AsDiscreted,
    RandZoomd,
    RandAffined,
    Lambdad,
    Resized
)

#data_dir = "../mitochondria/data2"
#images_dir = os.path.join(data_dir, "images")
#masks_dir = os.path.join(data_dir, "masks_png")


val_images = [os.path.join(val_images_dir, img) for img in os.listdir(val_images_dir) if img.endswith(".png")]
val_masks = [os.path.join(val_masks_dir, mask) for mask in os.listdir(val_masks_dir) if mask.endswith(".png")]

#val_images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".png")]
#val_masks = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir) if mask.endswith(".png")]

val_images.sort()
val_masks.sort()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),   
    EnsureChannelFirstd(keys=["image", "mask"]),  
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",
        spatial_size=spatial_size,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
        image_key="image",
    ),
    CustomTransform(keys=["image"], preprocess_input=preprocess_input),
    ConvertToMultiChannelMask(keys=["mask"]),  # Apply the custom transform
])


# Define your validation files
val_files = [{"image": img, "mask": mask} for img, mask in zip(val_images[:], val_masks[:])]

# Use the same transformations as the training set for both validation and test datasets
val_ds = Dataset(data=val_files, transform=val_transforms)

# Create validation and test data loaders
val_loader = DataLoader(val_ds, batch_size=batch_size_val, num_workers=num_workers_val)


# Report split sizes
print('Validation set has {} instances'.format(len(val_ds)))

#-----------------------------------------------------------------------------------------------------------------------
# Sanity Check
#-----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
import numpy as np
  

# Assuming you have a DataLoader named train_loader
batch = next(iter(val_loader))  # Fetching a batch

# Show and save all the accumulated plots
fig, axes = plt.subplots(6, 2, figsize=(10, 20))  # Adjust figsize as needed

# Set the average IoU scores and AP values as the title
plt.suptitle(f"Sanity Check", fontsize=16, y=0.95)

for i in range(6):  # Plotting six samples
    images = batch['image']
    masks = batch['mask']                  # shape: (80, 3, 256, 256)
    images = torch.mean(images, dim=1)
    print(masks.shape)
    masks = masks.permute(0, 2, 3, 1)      # shape: (80, 256, 256, 3) for plotting
    print(masks.shape)
    
    gt_mask = masks[i].cpu().numpy()
    
    print("gt_mask unique values: ", np.unique(gt_mask))
    print("gt_mask shape: ", gt_mask.shape)
    # Convert the NumPy array to a PIL image using the custom colormap
    gt_pil = Image.fromarray(np.argmax(gt_mask, axis=2).astype(np.uint8))
    gt_pil = gt_pil.convert('P')  # Convert the image to 8-bit pixels


    # Apply the colormap
    gt_pil.putpalette([
        0, 0, 0,  # Index 0: Black
        255, 255, 255,  # Index 1: White
        255, 255, 0,  # Index 2: Yellow
    ])
    
   
    # Add the generated plots to the corresponding subplots
    axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
    axes[i, 0].set_title("Image")
    axes[i, 0].axis('off')
    
    # To revert the one-hot encoded mask, just use the class indices as the mask directly
    #axes[i, 1].imshow(np.argmax(masks[i].cpu().numpy(), axis=2).astype(np.uint8), cmap=cmap)  # Use 'gray' colormap for grayscale
    axes[i, 1].imshow(gt_pil, cmap='viridis')  # Use 'gray' colormap for grayscale
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis('off')

plt.savefig(f"./multi_class_cross_validation/{output_directory}/six_samples.png", bbox_inches='tight', pad_inches=0.1)



#-----------------------------------------------------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------------------------------------------------

# Load the pre-trained model
#pretrained_model_path = f'./multi_class_hpbandster/vgg16_DiceFocalLoss_baseline_default/best_model.pth'
pretrained_model_path = f'./multi_class_cross_validation/Adam/DiceFocalLoss/Unet_vgg16/Unet_vgg16_1000epochs_bs4_lr_0.0001/best_model_fold_0.pth'


model = torch.load(pretrained_model_path)

# Unwrap the model if it was wrapped with DataParallel
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# Move the model to the available devices
device = 'cpu'
model = model.to('cpu')  # Move the model to the first GPU
#model = torch.nn.DataParallel(model)  # Wrap with DataParallel


# Loss function definition based on configuration
loss_fn = getattr(monai_losses, loss_name)(softmax=True)

# Wrap the loss function with a name
loss_fn = NamedLoss(loss_fn, loss_name)




from my_utils import TestEpoch
from torchmetrics import Metric


# define metrics
metrics = [
    #smp_utils.metrics.IoU(threshold=0.5),
    #smp_utils.metrics.IoU(threshold=0.5),
    smp_utils.metrics.IoU(threshold=0.5),  # Add per-class IoU
    #JaccardIndex(task='multiclass', num_classes=3, threshold=0.5, average='macro'),
    #smp_utils.metrics.Fscore(threshold=0.5),
    #smp_utils.metrics.Precision(threshold=0.5),
    #smp_utils.metrics.Recall(threshold=0.5),
    #smp_utils.metrics.Accuracy(threshold=0.5),
]

# Optimizer definition based on configuration
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)



#-----------------------------------------------------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------------------------------------------------
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from tqdm import tqdm as tqdm
import sys
import torch.nn.functional as F
from torchmetrics.detection import IntersectionOverUnion

verbose = True

model.eval()
model.to('cpu')  # Move the entire model to CPU, including parameters and buffers

num_classes = 3  # Adjust this based on the number of classes in your segmentation task
iou_metric = IntersectionOverUnion(iou_threshold=0.5)


SMOOTH = 1e-6


def calculate_iou(y_true, y_pred, class_label):
    # Convert inputs to PyTorch tensors if they are NumPy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Flatten the arrays to 1D
    y_true_flat = (y_true == class_label).flatten()
    y_pred_flat = (y_pred == class_label).flatten()

    # Calculate True Positive, False Positive, and False Negative
    true_positive = torch.sum(y_true_flat & y_pred_flat).item()
    false_positive = torch.sum(~y_true_flat & y_pred_flat).item()
    false_negative = torch.sum(y_true_flat & ~y_pred_flat).item()

    # Calculate IoU
    iou = true_positive / (true_positive + false_positive + false_negative)

    return iou

# Example usage:
# Assuming you have a DataLoader named val_loader
batch = next(iter(val_loader))  # Fetching a batch

# Show and save all the accumulated plots
fig, axes = plt.subplots(6, 3, figsize=(15, 20))  # Adjust figsize as needed

# Set the average IoU scores and AP values as the title
plt.suptitle(f"Sanity Check", fontsize=16, y=0.95)

total_class_0_iou = 0.0
total_class_1_iou = 0.0
total_class_2_iou = 0.0
total_iou = 0.0
num_samples = len(val_loader)  # Number of samples to plot

for batch in val_loader:
    images = batch['image']  # shape (batch_size, 3, 256, 256)
    masks = batch['mask']    # shape (batch_size, num_classes, 256, 256)
    
    print(images.shape)
    #images = torch.mean(images, dim=1)
    #print(images.shape) # shape 40,256,256
    print(masks.shape)
    
    # Convert the masks for plotting
    masks_for_plot = masks.permute(0, 2, 3, 1).cpu().numpy()
    
    print("masks_for_plot: ", masks_for_plot.shape)
    # Ground Truth
    gt_mask = np.argmax(masks_for_plot, axis=3).astype(np.uint8)
    print("gt_ shape: ", gt_mask.shape)
    print(np.unique(gt_mask))
    
    
    # Make a prediction using the model
    with torch.no_grad():
        prediction = model(images).cpu()
        

    
    
    # Convert prediction to numpy array and apply argmax
    prediction = torch.argmax(prediction, dim=1).numpy()
    print("prediction shape:", prediction.shape)
    
    # gt_mask shape: (40, 256, 256)
    # prediction shape: (40, 256, 256)
    
    # Calculate IoU for each class
    class_0_iou = calculate_iou(gt_mask, prediction, class_label=0)
    class_1_iou = calculate_iou(gt_mask, prediction, class_label=1)
    class_2_iou = calculate_iou(gt_mask, prediction, class_label=2)

    total_class_0_iou += class_0_iou
    total_class_1_iou += class_1_iou
    total_class_2_iou += class_2_iou
    
    
    # Calculate overall IoU
    total_iou += (class_0_iou + class_1_iou + class_2_iou) / 3


# Calculate average IoU
avg_class_0_iou = total_class_0_iou / num_samples
avg_class_1_iou = total_class_1_iou / num_samples
avg_class_2_iou = total_class_2_iou / num_samples

# Calculate overall average IoU
avg_iou = total_iou / num_samples

print(f"Average IoU for Class 0: {avg_class_0_iou}")
print(f"Average IoU for Class 1: {avg_class_1_iou}")
print(f"Average IoU for Class 2: {avg_class_2_iou}")
print(f"Overall Average IoU: {avg_iou}")


# Plotting
class_labels = ['Background', 'Untreated', 'Treated', 'Overall']
avg_iou_values = [avg_class_0_iou, avg_class_1_iou, avg_class_2_iou, avg_iou]

# Define colors suitable for a thesis
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_labels, avg_iou_values, color=colors)
ax.set_ylabel('Average IoU', fontsize=14)
ax.set_title('Multiclass Pretrained', fontsize=16)
ax.tick_params(axis='both', labelsize=12)

# Adding the numerical values on top of the bars
for bar, value in zip(bars, avg_iou_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{value:.4f}', 
            ha='center', va='bottom', fontsize=12)


print(output_directory)
plt.savefig(f'./multi_class_cross_validation/{output_directory}/stats.png', bbox_inches='tight', pad_inches=0.1)



