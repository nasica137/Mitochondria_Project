import torch
import optuna
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, AsDiscreted
from monai.data import Dataset
import segmentation_models_pytorch as smp
import monai.losses as monai_losses
import json

from my_utils import train_and_validate

# Load configuration from JSON file
with open('config2.json', 'r') as config_file:
    config = json.load(config_file)

# Define the objective function to minimize
def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
    optimizer_name = trial.suggest_categorical('optimizer_name', ["Adam", "RMSprop", "SGD"])  # Add more model options if needed
    loss_name = trial.suggest_categorical('loss_name', ["DiceLoss", "FocalLoss", "DiceCELoss"])  # Add more model options if needed
    encoder_name = trial.suggest_categorical('encoder_name', ['resnet50', 'resnet101'])  # Add more encoder options if needed
    epochs = 50  # Set your desired number of epochs
    
    config['learning_rate'] = learning_rate
    config['optimizer']['name'] = optimizer_name
    config['loss']['name'] = loss_name
    config['model']['encoder_name'] = encoder_name
    config['epochs'] = epochs

    # Train the model with these hyperparameters
    val_iou_score = train_and_validate(config)
    
    # Return the validation IoU score as the objective value to minimize
    return -val_iou_score

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')  # Maximize the validation IoU score
study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)