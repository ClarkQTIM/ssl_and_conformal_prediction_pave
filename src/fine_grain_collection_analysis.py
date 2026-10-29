"""
Script: fine_grain_analysis.py

Description:
------------
This script performs two main tasks to support fine-grain analysis of model uncertainty and misclassification
patterns in cervical image classification tasks:

1. **Image Selection and Saving**:
   It identifies and copies images that match specific ground truth and prediction criteria, focusing on clinically
   important or interesting cases. The images are saved with bounding boxes overlaid for easy visual inspection.

   The criteria include:
   - Ground truth = Cancer (6), Predicted = Normal (0) — potential false negatives
   - Correct Normal predictions with high uncertainty (length-3 sets)
   - Correct Cancer predictions with low or high uncertainty (length-1 and length-3 sets)

2. **Confusion Matrix Visualization for Fine-Grain Ground Truths**:
   Using precomputed CSVs, the script builds and saves annotated confusion matrices that show:
   - Count of predictions for each (true, predicted) class pair
   - Average conformal prediction set length in each matrix cell
   - Optionally saved as annotated heatmaps and CSV summaries

Typical use cases:
------------------
- To extract visual examples for clinician review.
- To investigate model performance across fine-grained diagnostic labels.
- To better understand patterns of uncertainty and misclassification in high-risk cases.

Requirements:
-------------
- Prediction CSVs that include conformal prediction lengths, predicted classes, and image filenames.
- Ground truth CSVs with bounding box coordinates and fine-grained class labels.
"""


#########
# Imports
#########

import os
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

###########
# Functions
###########

# Selecting different images based on various accuracy and uncertainty criteria

def construct_csv_paths(prediction_csvs, analysis_base_path, num_classes, algo, alpha):

    """
    Constructs full file paths to conformal analysis CSVs for a given model, dataset, and alpha level.

    This function assumes a specific directory structure:
    [base]/[model_name]/[num_classes]/[algorithm]/[alpha]/before_removing_any_images_removed_data_with_slots.csv

    Parameters:
    -----------
    prediction_csvs : list of str
        Paths to model prediction CSVs, used to extract model names.

    analysis_base_path : str
        Root directory where conformal analysis outputs are stored.

    num_classes : int
        Number of prediction classes used in the model.

    algo : str
        Conformal prediction algorithm name (e.g., 'lac').

    alpha : float
        Alpha level used in the conformal prediction.

    Returns:
    --------
    csv_files : list of str
        List of full paths to the expected analysis CSVs.
    """

    alpha_str = f"alpha_{int(alpha * 100):03d}"
    csv_files = []

    for pred_path in prediction_csvs:
        model_name = Path(pred_path).stem.replace("_seed_test2_predictions", "")
        csv_file = (
            Path(analysis_base_path)
            / model_name
            / str(num_classes)
            / algo
            / alpha_str
            / "before_removing_any_images_removed_data_with_slots.csv"
        )
        csv_files.append(str(csv_file))
    
    return csv_files

def save_images_matching_criteria(csv_files, bounding_box_csv, base_image_path, output_dir):

    """
    Selects and saves images that match predefined prediction-ground truth criteria, 
    with optional bounding box overlays.

    The following criteria are used to select and organize images:
    - False Negative: Ground Truth = Cancer (6), Predicted = Normal (0)
    - Correct Normal predictions (GT = 0, Pred = 0) with high uncertainty (CUQ set length = 3)
    - Correct Cancer predictions (GT = 6, Pred = 2) with high (length = 3) and low (length = 1) uncertainty

    Parameters:
    -----------
    csv_files : list of str
        List of CSV file paths containing prediction results, conformal prediction set lengths, and image names.

    bounding_box_csv : str
        Path to a CSV file containing bounding box coordinates. Must include `MASKED_IMG_ID`, `x1`, `x2`, `y1`, `y2`.

    base_image_path : str
        Root directory where the image files are stored.

    output_dir : str
        Directory to save the selected images. Subfolders will be created for each criterion.
    
    Notes:
    ------
    - Bounding boxes are drawn in red and saved over the copied image.
    - If a bounding box is missing for an image, a warning will be printed.
    - File structure: one subdirectory per criterion.
    """

    os.makedirs(output_dir, exist_ok=True)
    bounding_box_information = pd.read_csv(bounding_box_csv)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            gt = row['CC_ST']
            pred_class = row['pred_class_mc']
            pred_set_mc_length = row['pred_set_mc_length']
            image_filename = row['image']
            image_path = os.path.join(base_image_path, image_filename)

            save_image = False
            criterion_output_dir = None

            # Define filtering criteria
            if gt == 6 and pred_class == 0:
                criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_0')
                save_image = True
            elif gt == 0 and pred_class == 0 and pred_set_mc_length == 3:
                criterion_output_dir = os.path.join(output_dir, 'gt_0_pred_0_cuq_len_3')
                save_image = True
            elif gt == 6 and pred_class == 2 and pred_set_mc_length == 3:
                criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_2_cuq_len_3')
                save_image = True
            elif gt == 6 and pred_class == 2 and pred_set_mc_length == 1:
                criterion_output_dir = os.path.join(output_dir, 'gt_6_pred_2_cuq_len_1')
                save_image = True

            # Save image + draw bounding box if applicable
            if save_image:
                os.makedirs(criterion_output_dir, exist_ok=True)
                saved_image_path = os.path.join(criterion_output_dir, image_filename)
                shutil.copy(image_path, saved_image_path)

                bb_row = bounding_box_information[bounding_box_information['MASKED_IMG_ID'] == image_filename]
                if not bb_row.empty:
                    x1, x2, y1, y2 = list(bb_row[['x1', 'x2', 'y1', 'y2']].values[0])
                    image = Image.open(saved_image_path)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                    image.save(saved_image_path)
                    print(f"✔ Saved with bounding box: {saved_image_path}")
                else:
                    print(f"⚠ No bounding box for {image_filename}")

    print(f"\n✅ All matching images saved to: {output_dir}")

# Fine-grain confusion matrix construction

def find_csv_files(base_dir):

    """
    Recursively searches a directory for all `.csv` files.

    Parameters:
    -----------
    base_dir : str
        Root directory to search.

    Returns:
    --------
    csv_files : list of str
        Full paths to all CSV files found.
    """

    csv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def parse_file_path(file_path, base_dir):

    """
    Extracts metadata (model name, dataset, conformal setup) from a structured file path.

    Parameters:
    -----------
    file_path : str
        Full path to a file.

    base_dir : str
        Root portion of the file path to strip before parsing subdirectories.

    Returns:
    --------
    tuple : (model, test_dataset, num_classes, conformal_algorithm, alpha_value)
        Parsed components from the file path.
    """

    relative_path = os.path.relpath(file_path, base_dir)
    path_parts = relative_path.split(os.sep)
    model = path_parts[0]
    test_dataset = path_parts[1]
    num_classes = path_parts[2]
    conformal_algorithm = path_parts[3]
    alpha_value = path_parts[4]
    return model, test_dataset, num_classes, conformal_algorithm, alpha_value

def analyze_csv(file_path):
    
    """
    Parses a CUQ prediction CSV and computes a confusion matrix annotated with 
    prediction counts and average conformal set lengths.

    Parameters:
    -----------
    file_path : str
        Path to the input CSV file.

    Returns:
    --------
    confusion_matrix : dict
        Dictionary mapping (actual, predicted) → {count, avg_length}.
    """

    df = pd.read_csv(file_path)
    confusion_matrix = defaultdict(lambda: {"count": 0, "avg_length": 0})
    for _, row in df.iterrows():
        actual = row["CC_ST"]
        predicted = row["pred_class_mc"]
        length = row["pred_set_mc_length"]
        key = (actual, predicted)
        confusion_matrix[key]["count"] += 1
        confusion_matrix[key]["avg_length"] += length
    for key in confusion_matrix:
        confusion_matrix[key]["avg_length"] /= confusion_matrix[key]["count"]
    return confusion_matrix

def display_confusion_matrix(confusion_matrix):

    """
    Converts a dictionary-based confusion matrix into a tidy DataFrame 
    with actual/predicted class pairs, counts, and average conformal lengths.

    Parameters:
    -----------
    confusion_matrix : dict
        Output from `analyze_csv()`.

    Returns:
    --------
    matrix_df : pandas.DataFrame
        Tabular view of the confusion matrix with useful summary statistics.
    """

    matrix_df = pd.DataFrame([
        {"Fine-grain": actual, "Predicted": predicted, "Count": info["count"], "Avg Length": info["avg_length"]}
        for (actual, predicted), info in confusion_matrix.items()
    ])
    return matrix_df

def save_confusion_matrix(matrix_df, file_name, output_dir):

    """
    Saves a visual and tabular version of the fine-grain confusion matrix.

    Produces:
    - A CSV file summarizing counts and average set lengths.
    - A heatmap (PNG) where each cell shows the average set length and count.

    Parameters:
    -----------
    matrix_df : pandas.DataFrame
        Output from `display_confusion_matrix()`.

    file_name : str
        Prefix for saving the CSV and PNG.

    output_dir : str
        Directory where outputs will be written.
    """
    
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

    # matrix_df.to_csv(csv_path, index=False)

    count_pivot_df = matrix_df.pivot(index="Fine-grain", columns="Predicted", values="Count")
    avg_length_pivot_df = matrix_df.pivot(index="Fine-grain", columns="Predicted", values="Avg Length")

    combined_pivot_df = avg_length_pivot_df.round(2).astype(str) + "\n(n=" + count_pivot_df.astype(str) + ")"

    plt.figure(figsize=(25, 23))
    ax = sns.heatmap(avg_length_pivot_df, annot=combined_pivot_df, fmt="", cmap="Blues", cbar=False)
    ax.set_xticklabels(['Normal', 'Indeterminate', 'Severe'], rotation=45, ha='right')
    ax.set_yticklabels(['Normal', 'Indeterminate-', 'Indeterminate', 'Indeterminate+', 'Precancer', 'Cancer'], rotation=0, va='center')

    plt.savefig(img_path)
    plt.close()

def construct_csv_paths(prediction_csvs, analysis_base_path, num_classes, algo, alpha):
    alpha_str = f"alpha_{int(alpha * 100):03d}"
    csv_files = []
    for pred_path in prediction_csvs:
        model_name = Path(pred_path).stem.replace("_seed_test2_predictions", "")
        csv_file = (
            Path(analysis_base_path)
            / model_name
            / str(num_classes)
            / algo
            / alpha_str
            / "before_removing_any_images_removed_data_with_slots.csv"
        )
        csv_files.append(str(csv_file))
    return csv_files

def run_fine_grain_analysis(prediction_csvs, base_analysis_path, output_dir, num_classes, algo, alpha):

    """
    Constructs and saves annotated confusion matrices showing prediction counts and 
    average conformal set lengths for fine-grain ground truth categories.

    This analysis is particularly useful when comparing how models behave at boundaries 
    between clinical labels, e.g., Indeterminate ↔ Severe.

    Parameters:
    -----------
    prediction_csvs : list of str
        List of model prediction CSV paths (one per model).

    base_analysis_path : str
        Root path where CUQ analysis outputs are stored, organized by model → #classes → algorithm → alpha.

    output_dir : str
        Directory to save the resulting heatmap plots and summary CSVs.

    num_classes : int
        Number of classification categories used during prediction (e.g., 3 for Normal / Indeterminate / Severe).

    algo : str
        Conformal algorithm used (e.g., 'lac').

    alpha : float
        Significance level used to construct conformal prediction sets (e.g., 0.1).

    Output:
    -------
    - Saves .csv tables and .png heatmaps summarizing:
        • Prediction count for each (true, predicted) pair
        • Average conformal prediction set length for each cell
    """

    os.makedirs(output_dir, exist_ok=True)
    csv_files = construct_csv_paths(prediction_csvs, base_analysis_path, num_classes, algo, alpha)
    for file in csv_files:
        model, test_dataset, num_classes, conformal_algorithm, alpha_value = parse_file_path(file, base_analysis_path)
        confusion_matrix = analyze_csv(file)
        matrix_df = display_confusion_matrix(confusion_matrix)
        file_name = f"{model}_{test_dataset}_{num_classes}_{conformal_algorithm}_{alpha_value}"
        save_confusion_matrix(matrix_df, file_name, output_dir)

################
# Implementation
################

if __name__ == "__main__":

    '''
    Getting the images meeting specific accuracy and uncertainty criteria for additional clinician analysis
    '''

    output_dir = '/scratch90/chris/ssl_conformal_pave/analysis/fine_grain_analysis'
    bounding_box_csv = '/scratch90/chris/ssl_conformal_pave/csvs/SEED_three_class_ground_truths.csv'
    base_image_path = '/scratch90/chris/pave_consortium_projects/data/SEED'

    prediction_csvs = [
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_dn121_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_rn50_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_effnetv2s_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/effnetv2s_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/rn50_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dn121_seed_test2_predictions.csv",
    ]

    csv_files = construct_csv_paths(
        prediction_csvs,
        analysis_base_path='/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2',
        num_classes=3,
        algo='lac',
        alpha=0.10
    )

    save_images_matching_criteria(csv_files, bounding_box_csv, base_image_path, output_dir)

    # Fine-grain confusion matrix construction

    base_analysis_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2"
    output_dir = os.path.join(base_analysis_path, "uncertainty_by_fine_grain_gt")
    
    prediction_csvs = [
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_dn121_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_rn50_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_effnetv2s_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/effnetv2s_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/rn50_seed_test2_predictions.csv",
        "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dn121_seed_test2_predictions.csv"
    ]
    
    run_fine_grain_analysis(
        prediction_csvs=prediction_csvs,
        base_analysis_path=base_analysis_path,
        output_dir=output_dir,
        num_classes=3,
        algo="lac",
        alpha=0.10
    )