import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from patchify import patchify, unpatchify


# Load configuration from JSON file
with open('config1.json', 'r') as config_file:
    config = json.load(config_file)
    
output_directory = config['output_directory'].format(**config)
# Model architecture configuration
model_config = config["model"]
model_name = model_config["name"]
encoder_name = model_config["encoder_name"]

# Load the model checkpoint
model_checkpoint_path = f'./{output_directory}/best_model.pth'  # Update this with your model checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_checkpoint_path, map_location=device)

# Set the model to evaluation mode
model.eval()

resize_size = 256

# Load and preprocess the image
image_path = 'large_image.tif'  # Update this with your image path
image = Image.open(image_path).resize((resize_size,resize_size))  # Open the image in grayscale

from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')


image = preprocess_input(image)

# Convert the image to a NumPy array
image_np = np.array(image_normalized).squeeze()
print(image_np.shape)

# Patchify the image
patch_size = (256, 256)  # Define patch size
patches = patchify(image_np, patch_size, step=256)  # Patchify the image

# Make predictions for each patch
predicted_patches = []
for i in range(patches.shape[0]):
    row_patches = []
    for j in range(patches.shape[1]):
        patch = patches[i, j]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device).float()
        #print(patch_tensor.shape)
        with torch.no_grad():
            output = model(patch_tensor)
            prediction = (output.squeeze().cpu().numpy()).astype(np.uint8)
            row_patches.append(prediction)
    predicted_patches.append(row_patches)

print(np.array(predicted_patches).shape)
# Reconstruct the mask from predicted patches
predicted_mask = unpatchify(np.array(predicted_patches), image_np.shape)

# Plotting the original image and predicted mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(image_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'./{output_directory}/test_large_image.png')
