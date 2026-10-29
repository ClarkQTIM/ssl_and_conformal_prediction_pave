import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
from PIL import Image, ImageDraw
import os
import pandas as pd
from PIL import Image
import shutil

#####################################################################################
# Fine-grain Ground Truth and Prediction with Average Conformal Prediction Set Length
#####################################################################################

def find_csv_files(base_dir):
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def parse_file_path(file_path, base_dir):
    # Remove the base_dir and split the remaining path into components
    relative_path = file_path.replace(base_dir, "")
    path_parts = relative_path.split(os.sep)
    
    # Extract relevant information from the path
    model = path_parts[0]
    test_dataset = path_parts[1]
    num_classes = path_parts[2]
    conformal_algorithm = path_parts[3]
    alpha_value = path_parts[4]
    
    return model, test_dataset, num_classes, conformal_algorithm, alpha_value

def analyze_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Create a confusion matrix using a dictionary
    confusion_matrix = defaultdict(lambda: {"count": 0, "avg_length": 0})
    
    # Calculate the confusion matrix with counts and average pred_set_mc_length
    for _, row in df.iterrows():
        actual = row["CC_ST"]
        predicted = row["pred_class_mc"]
        length = row["pred_set_mc_length"]
        
        key = (actual, predicted)
        confusion_matrix[key]["count"] += 1
        confusion_matrix[key]["avg_length"] += length
    
    # Calculate the average length for each cell
    for key in confusion_matrix:
        confusion_matrix[key]["avg_length"] /= confusion_matrix[key]["count"]
    
    return confusion_matrix

def display_confusion_matrix(confusion_matrix):
    matrix_df = pd.DataFrame([
        {"Fine-grain": actual, "Predicted": predicted, "Count": info["count"], "Avg Length": info["avg_length"]}
        for (actual, predicted), info in confusion_matrix.items()
    ])
    return matrix_df

def save_confusion_matrix(matrix_df, file_name, output_dir):

    plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    })

    csv_path = os.path.join(output_dir, f"{file_name}_confusion_matrix.csv")
    img_path = os.path.join(output_dir, f"{file_name}_confusion_matrix.png")
    
    # Save as CSV
    matrix_df.to_csv(csv_path, index=False)
    
    # Generate pivot tables for counts and average lengths
    count_pivot_df = matrix_df.pivot(index="Fine-grain", columns="Predicted", values="Count")
    avg_length_pivot_df = matrix_df.pivot(index="Fine-grain", columns="Predicted", values="Avg Length")
    
    # Create a combined DataFrame to display avg lengths and counts together
    combined_pivot_df = avg_length_pivot_df.round(2).astype(str) + "\n(n=" + count_pivot_df.astype(str) + ")"
    
    # Generate and save the confusion matrix image with avg length and count
    plt.figure(figsize=(25, 23))
    # plt.title(f"Confusion Matrix: {file_name}")
    
    ax = sns.heatmap(avg_length_pivot_df, annot=combined_pivot_df, fmt="", cmap="Blues", cbar=False)
    
    # Custom x-axis and y-axis labels
    ax.set_xticklabels(['Normal', 'Indeterminate', 'Precancer+'], rotation=45, ha='right')
    ax.set_yticklabels(['Normal', 'Indeterminate-', 'Indeterminate', 'Indeterminate+', 'Precancer', 'Cancer'], rotation=0, va='center')
    
    plt.savefig(img_path)
    plt.close()

# Example usage
base_dir = "/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/"
output_dir = os.path.join(base_dir, "uncertainity_by_fine_grain_gt")

models = ['Model36']
dataset = ['inference_test2']
classes = ['3']
alg = ['lac']
alpha = ['alpha_010']

csv_files = ['/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_DN121/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/RN50/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_RN50/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/EffNetV2S/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv']


# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file
for file in csv_files:
    model, test_dataset, num_classes, conformal_algorithm, alpha_value = parse_file_path(file, base_dir)
    
    # Analyze CSV and create confusion matrix
    confusion_matrix = analyze_csv(file)
    matrix_df = display_confusion_matrix(confusion_matrix)
    
    # Create a unique name for the file based on the model and test dataset
    file_name = f"{model}_{test_dataset}_{num_classes}_{conformal_algorithm}_{alpha_value}"
    
    # Save the confusion matrix and the image
    save_confusion_matrix(matrix_df, file_name, output_dir)

#######################
# Saving Certain Images
#######################

'''
4/7: Needed for paper!
'''

import os
import pandas as pd
from PIL import Image
import shutil

# Define the base output directory for saving images
output_dir = '/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/fine_grain_analysis'
os.makedirs(output_dir, exist_ok=True)

bounding_box_information = pd.read_csv('/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/csvs/SEED_three_class_ground_truths.csv')

# Example CSV files (adjust paths as necessary)
csv_files = ['/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_DN121/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/RN50/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_RN50/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/EffNetV2S/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv',
'/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010/before_removing_any_images_removed_data_with_slots.csv']


# Define base image path (modify based on your actual data structure)
base_image_path = '/sddata/projects/Cervical_Cancer_Projects/data/SEED'

# Process each CSV file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        # Extract ground truth, predicted class, and pred_set_mc_length
        gt = row['CC_ST']
        pred_class = row['pred_class_mc']
        pred_set_mc_length = row['pred_set_mc_length']
        image_filename = row['image']  # Assuming image filenames are stored in the 'image' column
        image_path = os.path.join(base_image_path, image_filename)  # Full path to the image
        
        # Check the different criteria for saving the image
        save_image = False
        criterion_output_dir = None

        # Criterion 1: Ground truth 6, Predicted class 0. Ground truth Cancer predicted Normal
        if gt == 6 and pred_class == 0:
            criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_0')
            save_image = True
        
        # Criterion 2: Ground truth 0, Predicted class 0, pred_set_mc_length = 3. Uncertain but correct Normals
        elif gt == 0 and pred_class == 0 and pred_set_mc_length == 3:
            criterion_output_dir = os.path.join(output_dir, 'gt_0_pred_0_cuq_len_3')
            save_image = True

        # Criterion 3: Ground truth 6, Predicted class 2, pred_set_mc_length = 3. Uncertain but correct Cancer
        elif gt == 6 and pred_class == 2 and pred_set_mc_length == 3:
            criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_2_cuq_len_3')
            save_image = True

        # Criterion 4: Ground truth 6, Predicted class 2, pred_set_mc_length = 1. Certain and correct Cancer
        elif gt == 6 and pred_class == 2 and pred_set_mc_length == 1:
            criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_2_cuq_len_1')
            save_image = True

        # If the image meets one of the criteria, proceed with saving and adding the bounding box
        if save_image:
            os.makedirs(criterion_output_dir, exist_ok=True)
            # Copy the original image to the output directory
            shutil.copy(image_path, os.path.join(criterion_output_dir, image_filename))

            # Now that the image is saved, find the bounding box and add it to the saved image
            bb_row = bounding_box_information[bounding_box_information['MASKED_IMG_ID'] == image_filename]
            
            if not bb_row.empty:
                # Extract bounding box coordinates
                x1, x2, y1, y2 = list(bb_row[['x1', 'x2', 'y1', 'y2']].values[0])
                print(f"Bounding Box: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
                
                # Open the copied image from the output directory
                saved_image_path = os.path.join(criterion_output_dir, image_filename)
                image = Image.open(saved_image_path)
                
                # Create a drawing object to draw on the image
                draw = ImageDraw.Draw(image)
                
                # Draw the bounding box (rectangle)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=5)  # Customize color and width as needed
                
                # Save the image with the bounding box in the same location
                image.save(saved_image_path)
                print(f"Saved image with bounding box at: {saved_image_path}")
            else:
                print(f"No bounding box found for image: {image_filename}")


print(f"Images matching criteria have been saved to {output_dir}")

