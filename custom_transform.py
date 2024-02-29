from monai.transforms import MapTransform  # Ensure MapTransform is importe
import numpy as np
import torch

class CustomTransform(MapTransform):
    def __init__(self, keys, preprocess_input):
        super().__init__(keys)
        self.preprocess_input = preprocess_input
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Apply augmentations based on the provided keys
                gray_image = d[key] # shape: (1, 256, 256)
                #print(gray_image.shape)
                # Expand dimensions to make it a 3-channel grayscale image
                rgb_image = np.expand_dims(gray_image[0], axis=-1)  # Expand dimensions to (H, W, 1)

                # Create an RGB image by repeating the expanded grayscale channel along the third axis
                rgb_image = np.repeat(rgb_image, 3, axis=-1) # shape: (256, 256, 3)
                
                rgb_image = self.preprocess_input(rgb_image)
                #print(rgb_image.shape)
                rgb_image = np.transpose(rgb_image, (2, 0, 1))  # shape: (3, 256, 256)
                
                # Update the augmented image in the dictionary
                d[key] = torch.from_numpy(rgb_image).to(torch.float32)
        return d
        
        
        
        
# Define a custom transform to convert single-channel masks to multi-channel masks
class ConvertToMultiChannelMask(MapTransform):
    def __call__(self, data):
        mask = data["mask"]
        num_classes = 3  # Change this to the number of classes in your dataset
        multi_channel_mask = np.zeros((num_classes,) + mask.shape[1:], dtype=np.float32)
        for class_value in range(num_classes):
            class_mask = (mask == class_value).astype(np.float32)
            multi_channel_mask[class_value] = class_mask
        data["mask"] = torch.from_numpy(multi_channel_mask)
        return data
        
    

class MaskToRGB(MapTransform):
    def __call__(self, data):
        mask = data["mask"]
        mask = mask.astype(np.uint8)  # Convert to float for safe manipulation

        #print("mask i unique values: ", np.unique(mask[0]))

        # Assuming the mask is a single-channel NumPy array (H, W)
        # Convert to a 3-channel RGB mask
        mask_rgb = np.repeat(mask, 3, axis=0)  # Repeat the single channel 3 times along the channel dimension

        # Convert the NumPy array to a PyTorch tensor
        data["mask"] = torch.from_numpy(mask_rgb)
        return data