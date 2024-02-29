import torch
import pytorch_lightning as pl
from pprint import pprint
import pandas as pd
#import monai
from monai.data import Dataset, DataLoader
from monai.transforms import ToTensor, Compose
from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from sklearn.metrics import confusion_matrix
import monai.losses as monai_losses
from monai.losses import DiceLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'
from my_utils import TrainEpoch, ValidEpoch, NamedLoss, denormalize
import pandas as pd
from custom_transform import CustomTransform
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


import json
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss
from torch.optim import Adam

# Load configuration from JSON file
with open('finetune-config1.json', 'r') as config_file:
    config = json.load(config_file)
    



# Access configuration parameters
data_dir = config["data_dir"]
images_dir = os.path.join(data_dir, config["images_dir"])
masks_dir = os.path.join(data_dir, config["masks_dir"])
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
os.makedirs('./not_pretrained/' + output_directory, exist_ok=True)



from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')


#-----------------------------------------------------------------------------------------------------------------------
# OSError: [Errno 28] No space left on device (Solved!)
#-----------------------------------------------------------------------------------------------------------------------
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
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    NormalizeIntensityd,
    AsDiscreted,
    Lambda
)

#data_dir = "../mitochondria/data2"
#images_dir = os.path.join(data_dir, "images")
#masks_dir = os.path.join(data_dir, "masks_png")

images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".png")]
masks = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir) if mask.endswith(".png")]

images.sort()
masks.sort()


test_transforms = Compose([
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
    AsDiscreted(threshold=0.001, keys=["mask"]),
])

# Define your validation and test files
test_files = [{"image": img, "mask": mask} for img, mask in zip(images[:], masks[:])]

# Use the same transformations as the training set for both validation and test datasets
test_ds = Dataset(data=test_files, transform=test_transforms)


# Create validation and test data loaders
test_loader = DataLoader(test_ds, batch_size=batch_size_test, num_workers=num_workers_test)


# Report split sizes
print('Test set has {} instances'.format(len(test_ds)))


#-----------------------------------------------------------------------------------------------------------------------
# Load model 
#-----------------------------------------------------------------------------------------------------------------------

# Load the pre-trained model
pretrained_model_path = f'./not_pretrained/{output_directory}/best_model.pth'
model = torch.load(pretrained_model_path)

# Unwrap the model if it was wrapped with DataParallel
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# Move the model to the available devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to the first GPU
model = torch.nn.DataParallel(model)  # Wrap with DataParallel


# Loss function definition based on configuration
loss_fn = getattr(monai_losses, loss_name)(sigmoid=True)

# Wrap the loss function with a name
loss_fn = NamedLoss(loss_fn, loss_name)

# define metrics
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
    smp_utils.metrics.Fscore(threshold=0.5),
    smp_utils.metrics.Precision(threshold=0.5),
    smp_utils.metrics.Recall(threshold=0.5),
    smp_utils.metrics.Accuracy(threshold=0.5),
    # Add more metrics as needed
]

# Optimizer definition based on configuration
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)


#-----------------------------------------------------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------------------------------------------------


test_epoch = ValidEpoch(
    model,
    loss=loss_fn, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

test_logs = test_epoch.run(test_loader)
print("Evaluation on Test Data: ")
for metric in metrics:
    print(f"Mean {metric.__class__.__name__}: {test_logs[metric.__name__]:.4f}")
print(f"Mean Dice Loss: {test_logs[loss_fn.name]:.4f}")

# Creating a DataFrame with the single values
data = {
    f"Mean {metric.__class__.__name__}": [test_logs[metric.__name__]] for metric in metrics
}
data["Mean Dice Loss"] = [test_logs[loss_fn.name]]

# Create DataFrame
test_results_df = pd.DataFrame(data)

# Save as CSV
test_results_df.to_csv(f'./not_pretrained/{output_directory}/test_logs.csv', index=False)


num_samples_to_plot = 3  # Change this to the desired number of samples to plot

# Create a subplot for each sample
plt.figure(figsize=(15, 5 * num_samples_to_plot))

for i in range(num_samples_to_plot):
    # Fetch a sample from the test dataset
    sample = next(iter(test_loader))

    # Extract the image and mask from the sample
    image = sample["image"][0].cpu()
    mask = sample["mask"][0].cpu()

    # Make a prediction using the best_model
    with torch.no_grad():
        model.eval()
        prediction = model(image.unsqueeze(0)).cpu() > 0.5

    # Calculate IoU scores
    iou = smp_utils.metrics.IoU(threshold=0.5)(prediction, mask.unsqueeze(0)).cpu()
    
    
    image = np.transpose(image, (1, 2, 0))
    image = denormalize(image, encoder_name)
    
    # Plotting the image, mask, and prediction for each sample
    plt.subplot(num_samples_to_plot, 3, i * 3 + 1)
    plt.title(f"Image {i+1}")
    plt.imshow(image.squeeze().cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(num_samples_to_plot, 3, i * 3 + 2)
    plt.title(f"Mask {i+1}")
    plt.imshow(mask.squeeze().cpu(), cmap='gray')
    plt.axis('off')

    plt.subplot(num_samples_to_plot, 3, i * 3 + 3)
    plt.title(f"Prediction {i+1}\nIoU: {iou:.4f}")
    plt.imshow(prediction.squeeze().cpu(), cmap='gray')
    plt.axis('off')

plt.suptitle(f"Test Data Mean IoU: {test_logs['iou_score']:.4f}, Mean Dice Loss: {test_logs[loss_name]:.4f}", fontsize=18)
plt.tight_layout()
plt.savefig(f'./not_pretrained/{output_directory}/multiple_predictions_scores.png')




exec(open('not_pretrained_predict_large_image.py').read())