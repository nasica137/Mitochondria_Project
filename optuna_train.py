from PIL import Image
import numpy as np

# Read the "untreated_ground_truth_mask.png" image
untreated_img = Image.open("untreated_ground_truth_mask.png")

# Create an empty alpha channel
alpha_channel = np.full((untreated_img.size[1], untreated_img.size[0]), 255, dtype=np.uint8)

# Ensure the image is in 'L' (8-bit pixels, black and white) mode
untreated_img = untreated_img.convert('L')

# Convert the image to RGBA by adding the alpha channel
rgba_img = np.dstack((untreated_img, alpha_channel))

# Create an Image object from the numpy array with RGBA mode
rgba_pil_img = Image.fromarray(rgba_img, 'RGBA')

# Save the image with the alpha channel
rgba_pil_img.save("untreated_ground_truth_mask.png")





"""
# Get unique pixel values
unique_values = np.unique(img_array)

# Display the unique pixel values
print("Unique pixel values for untreated:", unique_values)
"""