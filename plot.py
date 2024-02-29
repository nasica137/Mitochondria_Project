import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_all_backbones_for_each_loss(config_file_path):
    # Load the JSON data
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define variations for encoder_name and loss[name]
    encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']
    loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

    for loss_name in loss_variations:
        names = []
        for encoder_name in encoder_variations:
            # Update the values in the config
            config['model']['encoder_name'] = encoder_name
            config['loss']['name'] = loss_name

            # Construct the output directory using updated values
            output_directory = config["output_directory"].format(**config)
            file_name = f'./fine-tuning/{output_directory}/valid_logs.csv'
            names.append((file_name, encoder_name)) 

        file_names = [name[0] for name in names]
        encoder_names = [name[1] for name in names]
        column_indices = [0, 1, 2]

        for idx in column_indices:
            plt.figure(figsize=(10, 6))

            for index, file_name in enumerate(file_names):
                df = pd.read_csv(file_name)
                column_name = df.columns[idx]
                column_data = df.iloc[:, idx]
                plt.plot(df.index, column_data, label=f"{encoder_names[index]}")

            plt.xlabel('Epochs')
            plt.ylabel(f'{column_name}')
            plt.title(f'Finetuning with {loss_name}')
            plt.legend()
            plt.grid(True)
            if idx == 0:
                plt.savefig(f'./plots/finetuning/{loss_name}/loss_compared_by_all_backbones.png')
            else:
                plt.savefig(f'./plots/finetuning/{loss_name}/{column_name}_compared_by_all_backbones.png')
            plt.show()
            plt.close()
            
            
            
def plot_all_losses_for_each_backbone(config_file_path):
    # Load the JSON data
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define variations for encoder_name and loss[name]
    encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']
    loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

    for encoder_name in encoder_variations:
        names = []
        for loss_name in loss_variations:
            # Update the values in the config
            config['model']['encoder_name'] = encoder_name
            config['loss']['name'] = loss_name

            # Construct the output directory using updated values
            output_directory = config["output_directory"].format(**config)
            file_name = f'./fine-tuning/{output_directory}/valid_logs.csv'
            names.append((file_name, loss_name)) 

        file_names = [name[0] for name in names]
        loss_names = [name[1] for name in names]
        encoder_name = 'Unet_' + encoder_name
        column_indices = [0, 1, 2]

        for idx in column_indices:
            plt.figure(figsize=(10, 6))

            for index, file_name in enumerate(file_names):
                df = pd.read_csv(file_name)
                column_name = df.columns[idx]
                column_data = df.iloc[:, idx]
                plt.plot(df.index, column_data, label=f"{loss_names[index]}")

            plt.xlabel('Epochs')
            plt.ylabel(f'Loss')
            plt.title(f'Finetuning for {encoder_name}')
            plt.legend()
            plt.grid(True)
            
            if idx == 0:
                plt.savefig(f'./plots/finetuning/{encoder_name}/loss_compared_by_all_losses.png')
            else:
                plt.savefig(f'./plots/finetuning/{encoder_name}/{column_name}_compared_by_all_losses.png')
            plt.show()
            plt.close()
            
            
            
import pandas as pd
import json

def create_table_from_test_logs(config_file_path):
    # Load the JSON data
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define variations for encoder_name and loss[name]
    encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']
    loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

    dfs = []  # List to store dataframes

    for encoder_name in encoder_variations:
        names = []
        for loss_name in loss_variations:
            # Update the values in the config
            config['model']['encoder_name'] = encoder_name
            config['loss']['name'] = loss_name

            # Construct the output directory using updated values
            output_directory = config["output_directory"].format(**config)
            file_name = f'./pretrained/{output_directory}/test_logs.csv'
            names.append((file_name, loss_name)) 

        file_names = [name[0] for name in names]

        # Concatenate data from all loss variations for each encoder
        dfs_per_encoder = [pd.read_csv(file).assign(encoder_name=encoder_name, loss_name=loss_name) for file, loss_name in names]
        concatenated_df = pd.concat(dfs_per_encoder)
        dfs.append(concatenated_df)

    # Concatenate all dataframes into a single dataframe
    final_df = pd.concat(dfs)

    # Save the concatenated dataframe to a CSV file
    final_df.to_csv('./plots/pretrained/consolidated_test_logs.csv', index=False)
    
    

def bar_all_backbones_for_each_loss(config_file_path):
    # Load the JSON data
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define variations for encoder_name and loss[name]
    encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']
    loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

    for loss_name in loss_variations:
        names = []
        max_values = []

        for encoder_name in encoder_variations:
            # Update the values in the config
            config['model']['encoder_name'] = encoder_name
            config['loss']['name'] = loss_name

            # Construct the output directory using updated values
            output_directory = config["output_directory"].format(**config)
            file_name = f'./fine-tuning/{output_directory}/valid_logs.csv'
            names.append((file_name, encoder_name)) 

        for index, (file_name, encoder_name) in enumerate(names):
            df = pd.read_csv(file_name)
            max_value = df.iloc[:, 1].max()  # Assuming the second column contains validation values
            max_values.append((max_value, encoder_name))

        max_values.sort(reverse=True)
        encoder_names_sorted = [item[1] for item in max_values]
        max_values_sorted = [item[0] for item in max_values]

        encoder_name = 'Unet_' + encoder_name
        
        plt.figure(figsize=(10, 6))
        plt.bar(encoder_names_sorted, max_values_sorted)
        plt.xlabel('Backbones')
        plt.ylabel(f'IoU Score')
        plt.title(f'Max Validation Value for each Backbone with {loss_name}')
        plt.grid(True)
        plt.savefig(f'./plots/finetuning/{loss_name}/max_validation_values_by_all_backbones.png')
        plt.show()
        plt.close()


def bar_all_losses_for_each_backbone(config_file_path):
    # Load the JSON data
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define variations for encoder_name and loss[name]
    encoder_variations = ['mobilenet_v2', 'xception', 'resnet18', 'resnet34', 'resnet50', 'vgg16']
    loss_variations = ['DiceLoss', 'DiceCELoss', 'DiceFocalLoss']

    for encoder_name in encoder_variations:
        names = []
        max_values = []

        for loss_name in loss_variations:
            # Update the values in the config
            config['model']['encoder_name'] = encoder_name
            config['loss']['name'] = loss_name

            # Construct the output directory using updated values
            output_directory = config["output_directory"].format(**config)
            file_name = f'./fine-tuning/{output_directory}/valid_logs.csv'
            names.append((file_name, loss_name))
            
        

        for index, (file_name, loss_name) in enumerate(names):
            df = pd.read_csv(file_name)
            max_value = df.iloc[:, 1].max()  # Assuming the second column contains validation values
            max_values.append((max_value, loss_name))

        max_values.sort(reverse=True)
        loss_names_sorted = [item[1] for item in max_values]
        max_values_sorted = [item[0] for item in max_values]
        encoder_name = 'Unet_' + encoder_name
        
        plt.figure(figsize=(10, 6))
        plt.bar(loss_names_sorted, max_values_sorted)
        plt.xlabel('Loss Functions')
        plt.ylabel('IoU Score')
        plt.title(f'Max Validation Value for each Loss with {encoder_name}')
        plt.grid(True)
        plt.savefig(f'./plots/finetuning/{encoder_name}/max_validation_values_by_all_losses.png')
        plt.show()
        plt.close()

    


# Call the function with the file path to your JSON config file
#create_table_from_test_logs('finetune-config2.json')


# Call the function with the file path to your JSON config file
#create_table_from_test_logs('finetune-config2.json')

# Call the function with the file path to your JSON config file
plot_all_losses_for_each_backbone('finetune-config2.json')

# Call the function with the file path to your JSON config file
plot_all_backbones_for_each_loss('finetune-config2.json')


# Call the new functions with the file path to your JSON config file
#bar_all_backbones_for_each_loss('finetune-config2.json')
bar_all_losses_for_each_backbone('finetune-config2.json')
