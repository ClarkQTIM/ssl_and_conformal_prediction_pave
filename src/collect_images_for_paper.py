"""
collect_images_for_paper.py

Consolidates key figures across multiple models into a central folder for paper inclusion.
Also combines specific pairs of images into a single graphic (side-by-side) to save space in LaTeX.

Author: [Your Name]
Date: [Today]
"""

import os
import shutil
from pathlib import Path
from PIL import Image

#####################
# Configurable Setup
#####################

# Root directory where analysis results live
base_dir = "/scratch90/chris/ssl_conformal_pave/analysis"
analysis_set = "six_model_seed_test2"

# List of model names (matching folder names inside analysis_set)
models = [
    "dino_dn121",
    "dino_rn50",
    "dino_effnetv2s",
    "effnetv2s",
    "rn50",
    "dn121"
]

# Class count, conformal method, and alpha string (e.g., "alpha_010")
num_classes = "3"
algo = "lac"
alpha = "alpha_010"

# Filenames to collect
files_to_collect = [
    "roc_auc_whole_dataset.png",
    "after_removal_by_floor_average_all_classes_confusion_matrix.png",
    "filtered_by_floor_average_roc_auc_class_1.png",
    "by_pred_class_and_set_length_cm_heatmap.png",
    "gt_vs_pred_class_w_avg_pred_set_length_confusion_matrix.png",
    "after_removal_by_floor_average_all_classes.png",
    "before_removing_any_images_removed_data_with_slots.png",
    "Pred Size vs Avg MC Pred Length_scatter.png",
    "cov_dist_distribution_cov.png"
]

# Destination folder
output_dir = "/scratch90/chris/ssl_conformal_pave/images_analysis_for_paper"
os.makedirs(output_dir, exist_ok=True)

##########################
# Helper Functions
##########################

def construct_analysis_path(model, num_classes, algo, alpha, analysis_set, base_dir):
    """Builds full path to a model's analysis directory."""
    return os.path.join(base_dir, analysis_set, model, num_classes, algo, alpha)


def copy_and_rename_file(src_path, dst_dir, model, filename):
    """Copies and renames a file with a model-specific prefix."""
    dst_filename = f"{model}_{filename}"
    dst_path = os.path.join(dst_dir, dst_filename)
    shutil.copy(src_path, dst_path)
    print(f"✔ Copied {src_path} → {dst_path}")


def combine_images_side_by_side(image_path1, image_path2, output_path):
    """Combines two images horizontally and saves the result."""
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Resize heights to match (optional but recommended)
    if img1.height != img2.height:
        max_height = max(img1.height, img2.height)
        img1 = img1.resize((img1.width, max_height))
        img2 = img2.resize((img2.width, max_height))

    # Create new image
    new_img = Image.new("RGB", (img1.width + img2.width, img1.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    new_img.save(output_path)
    print(f"📷 Combined image saved to: {output_path}")

##################
# Main Collection
##################

for model in models:
    model_analysis_dir = construct_analysis_path(
        model, num_classes, algo, alpha, analysis_set, base_dir
    )

    # Reset combination paths
    path1 = path2 = None

    for filename in files_to_collect:
        full_path = os.path.join(model_analysis_dir, filename)

        if os.path.exists(full_path):
            copy_and_rename_file(full_path, output_dir, model, filename)

            if filename == "Pred Size vs Avg MC Pred Length_scatter.png":
                path1 = full_path
            elif filename == "cov_dist_distribution_cov.png":
                path2 = full_path
        else:
            print(f"⚠ File not found: {full_path}")

    # Combine the two designated images (if both found)
    if path1 and path2:
        combined_output_path = os.path.join(
            output_dir, f"{model}_combined_mc_pred_len_and_cov_dist.png"
        )
        combine_images_side_by_side(path1, path2, combined_output_path)

print(f"\n✅ Done collecting files! Output folder: {output_dir}")
