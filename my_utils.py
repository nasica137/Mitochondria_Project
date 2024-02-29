import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm as tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from matplotlib.colors import ListedColormap
from collections import defaultdict
import torchmetrics
import os
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'
from PIL import Image
from monai.losses import DiceLoss, FocalLoss
import monai.losses as monai_losses
import segmentation_models_pytorch.utils as smp_utils
from torch.optim import Adam, RMSprop, SGD
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    AsDiscreted,
    Compose,
)

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, batch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for batch in iterator:
                x, y = batch["image"].to(self.device), batch["mask"].to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__(): loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
        
import torch.nn.functional as F       

class Epoch2:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = []
        for k, v in logs.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, list):
                formatted_value = "[" + ", ".join("{:.4}".format(item) for item in v) + "]"
            else:
                formatted_value = "{:.4}".format(v)
            str_logs.append("{} - {}".format(k, formatted_value))

        s = ", ".join(str_logs)
        return s

    def batch_update(self, batch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for batch in iterator:
                x, y = batch["image"].to(self.device), batch["mask"].to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__(): loss_meter.mean}
                logs.update(loss_logs)
                
                # apply softmax to y_pred
                y_pred_softmax = F.softmax(y_pred, dim=0)  # Softmax along the class dimension (dim=1)

                # update metrics logs
                for metric_fn in self.metrics:
                    if hasattr(metric_fn, '__name__'):
                        metric_name = metric_fn.__name__
                    else:
                        metric_name = str(metric_fn)

                    metric_value = metric_fn(y_pred_softmax, y).cpu().detach().numpy()

                    metrics_meters[metric_name] = metrics_meters.get(metric_name, AverageValueMeter())
                    metrics_meters[metric_name].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TestEpoch(Epoch2):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


        
        
# Create a wrapper class to assign a name to the loss
class NamedLoss(torch.nn.Module):
    def __init__(self, loss, name):
        super().__init__()
        self.loss = loss
        self.name = name  # Assign a name attribute

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
        
    def __name__(self):
        return self.name
  
        
        
import os
from monai.data import Dataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    AsDiscreted,
    Compose,
)

def get_train_val_dataloader(config):
    # Load your training and validation datasets
    
    # Access configuration parameters
    data_dir = config["data_dir"]
    images_dir = os.path.join(data_dir, config["images_dir"])
    masks_dir = os.path.join(data_dir, config["masks_dir"])
    batch_size_train = config["batch_size"]["train"]
    batch_size_val = config["batch_size"]["val"]
    num_workers_train = config["num_workers"]["train"]
    num_workers_val = config["num_workers"]["val"]
    spatial_size = config["spatial_size"]
    pos = config["pos"]
    neg = config["neg"]
    num_samples = config["num_samples"]

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks_png")

    images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".png")]
    masks = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir) if mask.endswith(".png")]

    images.sort()
    masks.sort()

    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        NormalizeIntensityd(keys="image"),  # Adding normalization for image only
        ScaleIntensityd(keys=["mask"]),
        RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[0, 1]),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_key="image",
        ),
        AsDiscreted(threshold=0.001, keys=["mask"]),
    ])

    train_files = [{"image": img, "mask": mask} for img, mask in zip(images[:300], masks[:300])]
    val_files = [{"image": img, "mask": mask} for img, mask in zip(images[300:400], masks[300:400])]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=train_transforms)
    
    # Create validation and test data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, num_workers=num_workers_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, num_workers=num_workers_val)


    # Report split sizes
    print('Training set has {} instances'.format(len(train_ds)))
    print('Validation set has {} instances'.format(len(val_ds)))

    return train_loader, val_loader


def sanity_check(train_loader):
    #-----------------------------------------------------------------------------------------------------------------------
    # Sanity Check
    #-----------------------------------------------------------------------------------------------------------------------

    # Assuming you have a DataLoader named train_loader
    batch = next(iter(train_loader))  # Fetching a batch

    # Extracting an image and mask from the batch
    image = batch["image"][0]  # Assuming the first image in the batch
    mask = batch["mask"][0]    # Assuming the mask corresponding to the first image

    # Plotting the image and mask
    plt.figure(figsize=(10, 5))

    # Plotting the image
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.squeeze(), cmap='gray')  # Assuming a grayscale image
    plt.axis('off')

    # Plotting the mask
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.squeeze(), cmap='gray')  # Assuming a grayscale mask
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("./sandbox/sample.png")


def train_and_validate(config):
    # Define your training and validation procedures
    # Train the UNet model using the provided hyperparameters
    # Return the validation IoU score
       
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
    os.makedirs(output_directory, exist_ok=True)



    from segmentation_models_pytorch.encoders import get_preprocessing_fn

    preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')


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

    train_loader, val_loader = get_train_val_dataloader(config)
    
    sanity_check(train_loader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model definition based on configuration
    model = getattr(smp, model_name)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    num_gpus = torch.cuda.device_count()
    print(model.name)
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Move the model to the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function definition based on configuration
    if loss_name == 'FocalLoss':
        loss_fn = getattr(monai_losses, loss_name)
    else:
        loss_fn = getattr(monai_losses, loss_name)(sigmoid=True)

    # Wrap the loss function with a name
    loss_fn = NamedLoss(loss_fn, loss_name)

    # define metrics
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    # Optimizer definition based on configuration
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)



    train_epoch = TrainEpoch(
        model, 
        loss=loss_fn, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model, 
        loss=loss_fn, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )
            
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, epochs):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, f'./{output_directory}/best_model.pth')
            print('Model saved!')
            
    return best_iou_score
    
    
    
def denormalize(normalized_image, encoder_name):
    # Assuming the normalization parameters used by ResNet18 for ImageNet
    if encoder_name == 'xception':
        mean = np.array([0.5, 0.5, 0.5])  # Mean values for the RGB channels
        std = np.array([0.5, 0.5, 0.5])   # Standard deviation values for the RGB channels
    if encoder_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg16', 'mobilenet_v2']:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
    return normalized_image * std + mean
    
from torchvision import transforms
from PIL import ImageFilter, ImageEnhance
    
def predict_mask(image_path, model, preprocess_input, device, resize_size=512, patch_size=(256, 256, 3), step=128, mask_path=''):
    # Load and preprocess the image
    image = Image.open(image_path).resize((resize_size, resize_size)).convert('RGB')
    # Adjust brightness of the image tensor
    #brightness_factor = 0.4  # Increase brightness by 50%
    #image = transforms.functional.adjust_brightness(image, brightness_factor)
    #image = image.filter(ImageFilter.SHARPEN)
    # Enhance contrast
    #contrast_factor = 2.0  # Increase contrast factor
    #contrast_enhancer = ImageEnhance.Contrast(image)
    #image = contrast_enhancer.enhance(contrast_factor)
    image_for_plot = np.array(image.copy())
    image = np.array(image)
    image = preprocess_input(image)
    
    
    

    # Define patch size and step for overlapping tiles
    predicted_mask = np.zeros((resize_size, resize_size), dtype=np.float32)
    count_map = np.zeros((resize_size, resize_size), dtype=np.float32)
    
    print("start patching large image...")
    for y in range(0, resize_size - patch_size[0] + 1, step):
        for x in range(0, resize_size - patch_size[1] + 1, step):
            # Extract patch
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patch = np.transpose(patch, (2, 0, 1))  # Change shape to match model input shape

            # Convert patch to tensor
            patch_tensor = torch.from_numpy(patch).to(device).float().unsqueeze(0)

            # Perform prediction on the patch
            with torch.no_grad():
                output = model(patch_tensor).sigmoid()
                output = output.cpu().numpy().squeeze().astype(np.float32)

            # Update predicted_mask and count_map with the patch predictions
            predicted_mask[y:y + patch_size[0], x:x + patch_size[1]] += output
            count_map[y:y + patch_size[0], x:x + patch_size[1]] += 1

    # Take the average of the predictions in overlapping regions
    predicted_mask = predicted_mask / count_map

    if len(mask_path) > 0:
        # Load and preprocess the mask
        ground_truth_mask = Image.open(mask_path).resize((resize_size, resize_size))
        ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)
        
        return predicted_mask, image_for_plot, ground_truth_mask
    else:
        return predicted_mask, image_for_plot
        
        
        
def predict_mask_ensemble(image_path, models, weights, preprocess_input, device, resize_size=512, patch_size=(256, 256, 3), step=128, mask_path=''):
    # Load and preprocess the image
    image = Image.open(image_path).resize((resize_size, resize_size)).convert('RGB')
    # Adjust brightness of the image tensor
    brightness_factor = 0.4  # Increase brightness by 50%
    image = transforms.functional.adjust_brightness(image, brightness_factor)
    image = image.filter(ImageFilter.SHARPEN)
    # Enhance contrast
    contrast_factor = 2.0  # Increase contrast factor
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(contrast_factor)
    image_for_plot = np.array(image.copy())
    image = np.array(image)
    image = preprocess_input(image)

    # Define patch size and step for overlapping tiles
    predicted_masks = []
    count_map = np.zeros((resize_size, resize_size), dtype=np.float32)
    
    print("Start patching large image...")
    for idx, model in enumerate(models):
        predicted_mask = np.zeros((resize_size, resize_size), dtype=np.float32)
        for y in range(0, resize_size - patch_size[0] + 1, step):
            for x in range(0, resize_size - patch_size[1] + 1, step):
                # Extract patch
                patch = image[y:y + patch_size[0], x:x + patch_size[1]]
                patch = np.transpose(patch, (2, 0, 1))  # Change shape to match model input shape

                # Convert patch to tensor
                patch_tensor = torch.from_numpy(patch).to(device).float().unsqueeze(0)

                # Perform prediction on the patch
                with torch.no_grad():
                    output = model(patch_tensor).sigmoid()
                    output = output.cpu().numpy().squeeze().astype(np.float32)

                # Update predicted_mask and count_map with the patch predictions
                predicted_mask[y:y + patch_size[0], x:x + patch_size[1]] += output
                count_map[y:y + patch_size[0], x:x + patch_size[1]] += 1

        # Take the average of the predictions in overlapping regions
        predicted_mask /= count_map
        predicted_masks.append(predicted_mask * weights[idx])  # Multiply by weight

    ensemble_predicted_mask = np.sum(predicted_masks, axis=0) / np.sum(weights)  # Weighted ensemble

    if len(mask_path) > 0:
        # Load and preprocess the mask
        ground_truth_mask = Image.open(mask_path).resize((resize_size, resize_size))
        ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)
        
        return ensemble_predicted_mask, image_for_plot, ground_truth_mask
    else:
        return ensemble_predicted_mask, image_for_plot


def predict_multiclass_mask(image_path, model, preprocess_input, device, resize_size=768, patch_size=(256, 256, 3), step=32, mask_path=''):
    # Load and preprocess the input image
    image = Image.open(image_path).resize((resize_size, resize_size)).convert('RGB')  # Load and resize image
    image_for_plot = np.array(image.copy())  # Create a copy for visualization purposes
    image = np.array(image)  # Convert to numpy array
    image = preprocess_input(image)  # Apply preprocessing function to the image
    
    # Define patch size and step for overlapping tiles
    num_classes = 3  # Number of classes in the multiclass segmentation task
    predicted_mask = np.zeros((num_classes, resize_size, resize_size), dtype=np.float32)  # Initialize predicted mask
    count_map = np.zeros((resize_size, resize_size), dtype=np.float32)  # Initialize count map for averaging
    
    print("Start patching large image...")
    # Iterate through the image in overlapping patches
    for y in range(0, resize_size - patch_size[0] + 1, step):
        for x in range(0, resize_size - patch_size[1] + 1, step):
            # Extract patch and transpose to match model input shape
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patch = np.transpose(patch, (2, 0, 1))  # Change shape to match model input shape

            # Convert patch to tensor and perform prediction
            patch_tensor = torch.from_numpy(patch).to(device).float().unsqueeze(0)
            with torch.no_grad():
                output = model(patch_tensor)
                output = output.cpu().numpy().squeeze().astype(np.float32)
                output = torch.nn.functional.softmax(torch.tensor(output), dim=0).cpu().detach().numpy()

            # Update predicted_mask and count_map with the patch predictions
            predicted_mask[:, y:y + patch_size[0], x:x + patch_size[1]] += output
            count_map[y:y + patch_size[0], x:x + patch_size[1]] += 1

    # Take the average of the predictions in overlapping regions
    predicted_mask /= count_map.reshape(1, resize_size, resize_size)

    if len(mask_path) > 0:
        # Load and preprocess the ground truth mask if provided
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        return predicted_mask, image_for_plot, ground_truth_mask  # Return predicted mask, input image, and ground truth mask
    else:
        return predicted_mask, image_for_plot  # Return predicted mask and input image
        
        
        
def get_cmap(unique_values):
    if np.array_equal(unique_values, [0, 1]):
        # Create a colormap for values 0 (black) and 1 (white)
        cmap = ListedColormap(['black', 'white'])
    elif np.array_equal(unique_values, [0, 2]):
        # Create a colormap for values 0 (black) and 2 (yellow)
        cmap = ListedColormap(['black', 'yellow'])
    else:
        print("taking this one")
        # Handle other cases or set a default colormap
        cmap = ListedColormap(['black', 'white', 'yellow'])
        
        
