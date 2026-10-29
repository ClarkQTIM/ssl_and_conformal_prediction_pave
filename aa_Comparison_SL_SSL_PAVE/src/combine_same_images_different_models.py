import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_image_files(base_dir, models, dataset, class_type, algorithm_type, alpha_value):
    """
    Finds the specified image file for given models, dataset, class type, algorithm type, and alpha value.

    Parameters:
    - base_dir: The base directory where the search begins.
    - models: A list of model names.
    - dataset: A string indicating the dataset (e.g., 'inference_test2').
    - class_type: An integer indicating the class (2 or 3).
    - algorithm_type: A string indicating the algorithm type ('lac' or 'aps').
    - alpha_value: A string indicating the alpha value (e.g., 'alpha_010').

    Returns:
    - A list of full paths to the found image files.
    """
    image_file_name = 'filtered_by_floor_average_roc_auc_class_1.png'
    found_files = []

    for model in models:
        search_path = os.path.join(
            base_dir,
            model,
            dataset,
            f'{class_type}/{algorithm_type}',
            alpha_value
        )
        full_image_path = os.path.join(search_path, image_file_name)

        if os.path.isfile(full_image_path):
            found_files.append(full_image_path)

    return found_files

def plot_images(image_paths, save_path):
    """
    Plots the collected images in a 3-by-2 grid with titles as the model names.

    Parameters:
    - image_paths: A list of full paths to the image files.
    - save_path: The path where the final combined image will be saved.
    """
    num_images = len(image_paths)
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # 3 rows, 2 columns

    for i, img_path in enumerate(image_paths):
        row = i // 2
        col = i % 2
        img = mpimg.imread(img_path)
        axs[row, col].imshow(img)
        # Extract the model name from the path and use it as the title
        model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))))
        axs[row, col].set_title(model_name)
        axs[row, col].axis('off')

    # Hide any unused subplots
    for j in range(num_images, 6):
        row = j // 2
        col = j % 2
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)

# Example usage
base_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms'
models = ['Model36', 'DINO_DN121', 'RN50', 'DINO_RN50', 'EffNetV2S', 'DINO_EffNetV2S']
dataset = 'inference_test2'
class_type = 3
algorithm_type = 'lac'
alpha_value = 'alpha_010'

# Find the image files
found_images = find_image_files(base_dir, models, dataset, class_type, algorithm_type, alpha_value)

# Plot the found images
plot_images(found_images, '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/paper/combined_filter_by_floor_avg_class_1_pred_lengths_dn121_rn50_effnet.png')
