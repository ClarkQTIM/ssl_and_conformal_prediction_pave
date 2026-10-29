import os
import glob
import pandas as pd

# Updated models list
models = ['Model36', 'DINO_DN121', 'RN50', 'DINO_RN50', 'EffNetV2S', 'DINO_EffNetV2S']

# Base columns to keep, will add model-specific columns in the loop
base_columns = ['image', 'gt', 'CC_ST', 'pred_mc_quality']

def find_csv_files(base_dir, test_dirs, ssl_dir, models, num_classes, algos, alphas):
    found_files = []

    for test_dir in test_dirs:
        for model in models:
            for cls in num_classes:
                for algo in algos:
                    for alpha in alphas:
                        search_pattern = os.path.join(
                            base_dir, test_dir, ssl_dir, '*', '*', model, '*', 
                            str(cls), algo, alpha, 'before_removing_any_images_removed_data_with_slots.csv'
                        )
                        found_files.extend(glob.glob(search_pattern))
    
    return found_files

def combine_csvs_with_conf_matrix_slot(csv_files, models):
    combined_df = None
    all_columns_to_keep = base_columns.copy()  # Start with the base columns

    # Mapping of conf_matrix_slot values to (gt, pred) pairs
    slot_to_cell_map = {
        1: (0, 0),  # cell_1
        2: (0, 1),  # cell_2
        3: (0, 2),  # cell_3
        4: (1, 0),  # cell_4
        5: (1, 1),  # cell_5
        6: (1, 2),  # cell_6
        7: (2, 0),  # cell_7
        8: (2, 1),  # cell_8
        9: (2, 2)   # cell_9
    }

    for file_path, model in zip(csv_files, models):
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Rename the 'conf_matrix_slot' and 'pred_class_mc' columns to include the model name
        conf_matrix_col_name = f'conf_matrix_slot_{model}'
        pred_class_mc_col_name = f'pred_class_mc_{model}'

        df.rename(columns={'conf_matrix_slot': conf_matrix_col_name,
                           'pred_class_mc': pred_class_mc_col_name}, inplace=True)

        # Update the list of columns to keep
        all_columns_to_keep.append(conf_matrix_col_name)
        all_columns_to_keep.append(pred_class_mc_col_name)

        if combined_df is None:
            # Initialize the combined DataFrame with the first CSV's data
            combined_df = df.copy()

            # Initialize cell count columns to zero
            for i in range(9):
                combined_df[f'cell_{i+1}'] = 0
        else:
            # Merge the new data with the combined DataFrame on 'image'
            combined_df = pd.merge(combined_df, df[['image', conf_matrix_col_name, pred_class_mc_col_name]], on='image', how='inner')

        # Update the cell counts in place based on the conf_matrix_slot
        for index, row in combined_df.iterrows():
            slot = row[conf_matrix_col_name]
            gt, pred = slot_to_cell_map.get(slot, (None, None))

            if gt is not None and pred is not None:
                cell_number = (gt * 3 + pred) + 1  # Calculate cell number (1-9)
                combined_df.at[index, f'cell_{cell_number}'] += 1

    # Return the combined DataFrame with only the required columns
    return combined_df[all_columns_to_keep + [f'cell_{i+1}' for i in range(9)]]

# Example usage
base_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis'
test_dirs = ['test2']
ssl_dir = 'SSL'
num_classes = [3]
algos = ['lac']
alphas = ['alpha_010']  # Add more as needed

csv_files = find_csv_files(base_dir, test_dirs, ssl_dir, models, num_classes, algos, alphas)
combined_df = combine_csvs_with_conf_matrix_slot(csv_files, models)

# Load the bounding box DataFrame
df_with_bb = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/csvs/SEED_three_class_ground_truths.csv'
bb_df = pd.read_csv(df_with_bb)
bb_df.rename(columns={'MASKED_IMG_ID': 'image'}, inplace=True)
# Select only the bounding box columns you need
bb_columns = ['image', 'x1', 'x2', 'y1', 'y2']

# Merge the bounding box columns into the combined DataFrame
final_df = pd.merge(combined_df, bb_df[bb_columns], on='image', how='inner')

# Save the final combined DataFrame to a new CSV file
output_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/before_removal_all_models_of_interest_fine_grain_gt_conf_matrix_slots_with_bb.csv'
final_df.to_csv(output_path, index=False)

print(f"Final CSV columns with bounding boxes: {final_df.columns.tolist()}")
