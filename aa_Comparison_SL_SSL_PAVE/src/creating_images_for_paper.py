import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_combined_image_plot(img_paths_algorithm_1, img_paths_algorithm_2, save_dir, save_name):
    """
    Create a combined plot with images for two algorithms.

    Parameters:
    - img_paths_algorithm_1: Tuple containing (boxplot_image_path, distribution_image_path) for the first algorithm.
    - img_paths_algorithm_2: Tuple containing (boxplot_image_path, distribution_image_path) for the second algorithm.
    - save_dir: Directory to save the resulting image.
    - save_name: Name for the saved image file.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Load and plot images for the first algorithm
    boxplot_img_1 = mpimg.imread(img_paths_algorithm_1[0])
    dist_img_1 = mpimg.imread(img_paths_algorithm_1[1])
    axs[0, 0].imshow(boxplot_img_1)
    axs[0, 1].imshow(dist_img_1)

    # Load and plot images for the second algorithm
    boxplot_img_2 = mpimg.imread(img_paths_algorithm_2[0])
    dist_img_2 = mpimg.imread(img_paths_algorithm_2[1])
    axs[1, 0].imshow(boxplot_img_2)
    axs[1, 1].imshow(dist_img_2)

    # Remove axes for a cleaner look
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name + '_combined_plot.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory
    print(f"Combined image plot saved to: {save_path}")

if __name__ == "__main__":

    create_combined_image_plot(('/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_020/Pred Size Vs CoV_boxplot.png',
                                '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_020/cov_dist_distribution_cov.png'),
                                ('/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_005/Pred Size Vs CoV_boxplot.png',
                                '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_005/cov_dist_distribution_cov.png'),
                               '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE', 'lac_020_005_comparison')
