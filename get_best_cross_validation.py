import pandas as pd

# Specify the number of folds
num_folds = 5  # You may need to adjust this based on your actual number of folds

# Initialize an empty list to store the maximum validation IoU for each fold
max_val_iou_list = []

# Iterate over each fold
for fold in range(num_folds):
    # Load the validation logs for the current fold
    valid_logs_path = f'./multi_class_cross_validation/Adam/DiceFocalLoss/Unet_vgg16/Unet_vgg16_100epochs_bs4_lr_0.0001/valid_logs_fold_{fold}.csv'
    valid_logs_path = f'./multi_class_cross_validation/Adam/DiceFocalLoss/Unet_vgg16/Unet_vgg16_100epochs_bs4_lr_0.0001/valid_logs_fold_{fold}.csv'
    valid_logs_df = pd.read_csv(valid_logs_path)

    # Find the epoch with the maximum validation IoU
    max_val_iou_epoch = valid_logs_df['iou_score'].idxmax()

    # Get the maximum validation IoU for the current fold
    max_val_iou = valid_logs_df.loc[max_val_iou_epoch, 'iou_score']

    # Append the maximum validation IoU to the list
    max_val_iou_list.append(max_val_iou)

# Calculate the mean of the maximum validation IoU across all folds
mean_max_val_iou = sum(max_val_iou_list) / num_folds

# Print the results
print(f"Maximum Validation IoU for each fold: {max_val_iou_list}")
print(f"Mean Maximum Validation IoU across all folds: {mean_max_val_iou}")
