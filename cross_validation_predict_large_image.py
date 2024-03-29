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
from my_utils import predict_mask, predict_mask_ensemble, predict_multiclass_mask, get_cmap
import argparse
import os
from matplotlib.colors import ListedColormap
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'


def run_multiclass_cross_validation(encoder_name, loss_name):
    import matplotlib.pyplot as plt
    # Load configuration from JSON file
    with open('finetune-config_multiclass1.json', 'r') as config_file:
        config = json.load(config_file)


    #config['model']['encoder_name'] = 'mobilenet_v2'
    #config['loss']['name'] = 'DiceLoss'


    config['model']['encoder_name'] = encoder_name
    config['loss']['name'] = loss_name

     
    output_directory = config['output_directory'].format(**config)
    # Model architecture configuration
    model_config = config["model"]
    model_name = model_config["name"]
    encoder_name = model_config["encoder_name"]

    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')

    # Load the pre-trained model
    
    if encoder_name == 'mobilenet_v2':
        base_directory = 'multi_class_cross_validation_not_pretrained'
        pretrained_model_path = f'./{base_directory}/Adam/DiceLoss/Unet_mobilenet_v2/Unet_mobilenet_v2_1000epochs_bs4_lr_0.0001/best_model_fold_1.pth'
    elif encoder_name == 'vgg16':
        base_directory = 'multi_class_cross_validation'
        pretrained_model_path = f'./{base_directory}/Adam/DiceFocalLoss/Unet_vgg16/Unet_vgg16_1000epochs_bs4_lr_0.0001/best_model_fold_0.pth'


    model = torch.load(pretrained_model_path)

    # Unwrap the model if it was wrapped with DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Move the model to the available devices
    device = 'cpu'
    model = model.to('cpu')  # Move the model to the first GPU

    # Set the model to evaluation mode
    model.eval()


    # Perform prediction for multiclass mask
    untreated_predicted_mask, untreated_image_for_plot, untreated_ground_truth_mask = predict_multiclass_mask("untreated_original_image.png", model, preprocess_input, device, mask_path="untreated_ground_truth_mask.png")


    print(np.unique(untreated_ground_truth_mask))

    # Convert the NumPy array to a PIL image using the custom colormap
    untreated_pred_pil = Image.fromarray(np.argmax(untreated_predicted_mask, axis=0).astype(np.uint8))
    untreated_pred_pil = untreated_pred_pil.convert('P')  # Convert the image to 8-bit pixels

    # Convert the NumPy array to a PIL image using the custom colormap
    untreated_gt_pil = Image.fromarray(untreated_ground_truth_mask.astype(np.uint8))
    untreated_gt_pil = untreated_gt_pil.convert('P')  # Convert the image to 8-bit pixels


    # Apply the colormap
    untreated_pred_pil.putpalette([
        0, 0, 0,  # Index 0: Black
        255, 255, 255,  # Index 1: White
        255, 255, 0,  # Index 2: Yellow
    ])

    # Apply the colormap
    untreated_gt_pil.putpalette([
        0, 0, 0,  # Index 0: Black
        255, 255, 255,  # Index 1: White
        255, 255, 0,  # Index 2: Yellow
        
    ])

    # Save the PIL image as a PNG file
    untreated_pred_pil.save(f'./{base_directory}/{output_directory}/untreated_predicted_mask.png')


    # Plotting the original image and predicted multiclass mask (example for visualization)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(untreated_image_for_plot)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(untreated_gt_pil, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Multiclass Mask')
    plt.imshow(untreated_pred_pil, cmap='viridis')  # Adjust the colormap accordingly
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'./{base_directory}/{output_directory}/test_large_untreated_image.png')





    treated_predicted_mask, treated_image_for_plot, treated_ground_truth_mask = predict_multiclass_mask("treated_original_image.png", model, preprocess_input, device, mask_path="treated_ground_truth_mask.png")

    print(np.unique(treated_ground_truth_mask))


    # Convert the NumPy array to a PIL image using the custom colormap
    treated_pred_pil = Image.fromarray(np.argmax(treated_predicted_mask, axis=0).astype(np.uint8))
    treated_pred_pil = treated_pred_pil.convert('P')  # Convert the image to 8-bit pixels

    # Convert the NumPy array to a PIL image using the custom colormap
    treated_gt_pil = Image.fromarray(treated_ground_truth_mask.astype(np.uint8))
    treated_gt_pil = treated_gt_pil.convert('P')  # Convert the image to 8-bit pixels


    # Apply the colormap
    treated_pred_pil.putpalette([
        0, 0, 0,  # Index 0: Black
        255, 255, 255,  # Index 1: White
        255, 255, 0,  # Index 2: Yellow
    ])

    # Apply the colormap
    treated_gt_pil.putpalette([
        0, 0, 0,  # Index 0: Black
        255, 255, 255,  # Index 1: White
        255, 255, 0,  # Index 2: Yellow
    ])

    # Save the PIL image as a PNG file
    treated_pred_pil.save(f'./{base_directory}/{output_directory}/treated_predicted_mask.png')


    # Plotting the original image and predicted multiclass mask (example for visualization)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(treated_image_for_plot)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(treated_gt_pil, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Multiclass Mask')
    plt.imshow(treated_pred_pil, cmap='viridis')  # Adjust the colormap accordingly
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'./{base_directory}/{output_directory}/test_large_treated_image.png')




    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    # Load images
    moving_image1 = sitk.ReadImage(f'untreated_ground_truth_mask.png', sitk.sitkFloat32)
    moving_image2 = sitk.ReadImage(f'./{base_directory}/{output_directory}/untreated_predicted_mask.png', sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(f'untreated_original_image.png', sitk.sitkFloat32)


    # Define the desired output size
    output_size = (512, 512)

    # Resample the ground truth masks to the desired output size
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)  # Set reference image for spacing and origin

    moving_image1 = resampler.Execute(moving_image1)



    # Display the images
    plt.figure(figsize=(10, 5))

    # Display fixed image in gray color
    plt.imshow(sitk.GetArrayViewFromImage(fixed_image), cmap='gray')

    # Overlay moving image1 (ground truth) in 'coolwarm' colormap with higher opacity
    plt.imshow(sitk.GetArrayViewFromImage(moving_image1), cmap='coolwarm', alpha=0.7)

    # Overlay moving image2 (predicted) in 'viridis' colormap with lower opacity
    plt.imshow(sitk.GetArrayViewFromImage(moving_image2), cmap='viridis', alpha=0.3)

    plt.axis('off')
    plt.title('Fixed Image (Gray) & Ground Truth Overlay (Red - Coolwarm) & Predicted Overlay (Blue - Viridis)')
    plt.savefig(f'./{base_directory}/{output_directory}/test_large_untreated_image_overlay.png')



    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    # Load images
    moving_image1 = sitk.ReadImage(f'treated_ground_truth_mask.png', sitk.sitkFloat32)
    moving_image2 = sitk.ReadImage(f'./{base_directory}/{output_directory}/treated_predicted_mask.png', sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(f'treated_original_image.png', sitk.sitkFloat32)


    resampler.SetReferenceImage(fixed_image)  # Set reference image for spacing and origin

    moving_image1 = resampler.Execute(moving_image1)


    # Display the images
    plt.figure(figsize=(10, 5))

    # Display fixed image in gray color
    plt.imshow(sitk.GetArrayViewFromImage(fixed_image), cmap='gray')

    # Overlay moving image1 (ground truth) in 'coolwarm' colormap with higher opacity
    plt.imshow(sitk.GetArrayViewFromImage(moving_image1), cmap='coolwarm', alpha=0.7)

    # Overlay moving image2 (predicted) in 'viridis' colormap with lower opacity
    plt.imshow(sitk.GetArrayViewFromImage(moving_image2), cmap='viridis', alpha=0.3)

    plt.axis('off')
    plt.title('Fixed Image (Gray) & Ground Truth Overlay (Red - Coolwarm) & Predicted Overlay (Blue - Viridis)')
    plt.savefig(f'./{base_directory}/{output_directory}/test_large_treated_image_overlay.png')
    
    
 
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multiclass cross-validation with different encoder and loss combinations.')
    parser.add_argument('--encoder', type=str, help='Encoder name (e.g., mobilenet_v2)')
    parser.add_argument('--loss', type=str, help='Loss name (e.g., DiceLoss)')

    args = parser.parse_args()

    if (args.encoder == 'mobilenet_v2' and args.loss == 'DiceLoss') or (args.encoder == 'vgg16' and args.loss == 'DiceFocalLoss'):
        # Run multiclass cross-validation with specified encoder and loss
        run_multiclass_cross_validation(args.encoder, args.loss)
    else:
        print("Please provide both --encoder and --loss arguments as mentioned in GITHUB.")
        
