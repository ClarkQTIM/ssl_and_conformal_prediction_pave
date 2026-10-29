import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
3/21/2025: Needed for paper
'''

def find_image_files(base_dir, models, dataset, class_type, algorithm_type, alpha_value, analysis1, analysis2):
    """
    Finds the specified image files for given models, dataset, class type, algorithm type, and alpha value.

    Parameters:
    - base_dir: The base directory where the search begins.
    - models: A list of model names.
    - dataset: A string indicating the dataset (e.g., 'inference_test2').
    - class_type: An integer indicating the class (2 or 3).
    - algorithm_type: A string indicating the algorithm type ('lac' or 'aps').
    - alpha_value: A string indicating the alpha value (e.g., 'alpha_010').
    - analysis1: Filename for the first analysis (e.g., 'analysis1.png').
    - analysis2: Filename for the second analysis (e.g., 'analysis2.png').

    Returns:
    - A dictionary where each key is a model name, and the value is a tuple with the paths to analysis1 and analysis2.
    """
    found_files = {}

    for model in models:
        search_path = os.path.join(
            base_dir,
            model,
            dataset,
            f'{class_type}/{algorithm_type}',
            alpha_value
        )
        analysis1_path = os.path.join(search_path, analysis1)
        analysis2_path = os.path.join(search_path, analysis2)

        if os.path.isfile(analysis1_path) and os.path.isfile(analysis2_path):
            found_files[model] = (analysis1_path, analysis2_path)

    return found_files

def plot_combined_images(image_paths, save_dir):
    """
    Combines the two analyses for each model and saves the result as a single image.

    Parameters:
    - image_paths: A dictionary where each key is a model name, and the value is a tuple with paths to analysis1 and analysis2.
    - save_dir: The directory where the combined images will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model, (analysis1_path, analysis2_path) in image_paths.items():
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns

        # Load and plot analysis1.png
        img1 = mpimg.imread(analysis1_path)
        axs[0].imshow(img1)
        # axs[0].set_title(f'{model} - Analysis 1')
        axs[0].axis('off')

        # Load and plot analysis2.png
        img2 = mpimg.imread(analysis2_path)
        axs[1].imshow(img2)
        # axs[1].set_title(f'{model} - Analysis 2')
        axs[1].axis('off')

        # Save the combined image
        save_path = os.path.join(save_dir, f'{model}_combined_analysis.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":

    base_dir = '/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms'
    models = ['Model36', 'DINO_DN121', 'RN50', 'DINO_RN50', 'EffNetV2S', 'DINO_EffNetV2S']
    dataset = 'inference_test2'
    class_type = 3
    algorithm_type = 'lac'
    alpha_value = 'alpha_010'
    analysis1 = 'Pred Size vs Avg MC Pred Length_scatter.png'
    analysis2 = 'cov_dist_distribution_cov.png'
    save_dir = '/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/paper/'

    # Find the image files
    found_images = find_image_files(base_dir, models, dataset, class_type, algorithm_type, alpha_value, analysis1, analysis2)

    # Plot and save the combined images
    plot_combined_images(found_images, save_dir)
