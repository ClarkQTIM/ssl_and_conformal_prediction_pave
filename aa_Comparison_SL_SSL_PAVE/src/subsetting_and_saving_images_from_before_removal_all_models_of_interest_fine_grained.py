import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load the combined CSV
combined_df = pd.read_csv('/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/before_removal_all_models_of_interest_fine_grain_gt_conf_matrix_slots_with_bb.csv')
base_image_path = '/sddata/projects/Cervical_Cancer_Projects/data/SEED'

for subset in ['cell_3', 'cell_7', 'cell_2', 'cell_8']:
    # Subset the DataFrame based on the condition (either cell_3 == 6 or cell_7 == 6)
    subset_df = combined_df[combined_df[subset].isin([6])]

    # Add a new column for the full image path
    subset_df['image_path'] = base_image_path + subset_df['image']  # Assuming 'image' column contains image filenames

    # Directory to save the images specific to the current subset
    save_dir_cell = os.path.join('/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/images_requiring_analysis_pred_gt_only/', subset)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir_cell, exist_ok=True)

    # Save the images with bounding boxes to the appropriate directory
    for idx, row in subset_df.iterrows():
        image_filename = row['image']  # The image filename
        image_path = os.path.join(base_image_path, image_filename)  # Full path to the image

        # Load the image
        image = Image.open(image_path)

        # Plot the image with bounding boxes
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Draw the bounding box
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Save the image with the bounding box overlaid
        save_image_path = os.path.join(save_dir_cell, image_filename)
        plt.savefig(save_image_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid memory issues

    # Save the subsetted CSV file to the same directory with an appropriate name
    csv_save_path = os.path.join(save_dir_cell, f'subset_{subset}.csv')
    subset_df.to_csv(csv_save_path, index=False)

    print(f"Subset CSV and images saved successfully for {subset}!")
