import os
import numpy as np
import tifffile
from PIL import Image
from patchify import patchify

string = "val"

# Input directories for images and masks
image_input_dir = f'new_data/{string}_images/{string}/'
mask_input_dir = f'new_data/{string}_masks/{string}/'

# Output directories for images and masks
image_output_dir = f'dataset/{string}/images/'
mask_output_dir = f'dataset/{string}/masks/'

# Create the output directories if they don't exist
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Load and process TIFF images and masks
for filename in os.listdir(image_input_dir):
    if filename.endswith('.tif'):
        image_path = os.path.join(image_input_dir, filename)
        mask_path = os.path.join(mask_input_dir, filename.replace('.tif', '_mask.tif'))

        # Load image and mask
        img = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        # Convert mask to binary (0 and 1)
        mask[mask > 0] = 1  # Assuming the foreground class has values greater than 0

        # Convert image to grayscale and resize to 1024x1024 using PIL
        img_pil = Image.fromarray(img).convert('L')
        img_pil = img_pil.resize((1024, 1024), resample=Image.BILINEAR)  # Change resampling method as needed

        img = np.array(img_pil)
        mask = np.array(Image.fromarray(mask).resize((1024, 1024), resample=Image.NEAREST))

        # Patchify the image and mask into 256x256 patches
        img_patches = patchify(img, (256, 256), step=256)
        mask_patches = patchify(mask, (256, 256), step=256)

        # Save the patches as PNG files
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                img_patch = Image.fromarray(img_patches[i, j, :, :])
                mask_patch = Image.fromarray(mask_patches[i, j, :, :])
                base_filename = os.path.splitext(filename)[0]
                img_patch_filename = f'{base_filename}_image_patch_{i}_{j}.png'
                mask_patch_filename = f'{base_filename}_mask_patch_{i}_{j}.png'
                img_patch_path = os.path.join(image_output_dir, img_patch_filename)
                mask_patch_path = os.path.join(mask_output_dir, mask_patch_filename)
                img_patch.save(img_patch_path)
                mask_patch.save(mask_patch_path)
