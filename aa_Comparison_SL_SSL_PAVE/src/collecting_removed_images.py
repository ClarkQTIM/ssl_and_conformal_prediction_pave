import os
import pandas as pd
import argparse
import sys

def find_removed_data_csvs(base_dir):
    removed_data_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("removed_data_by_floor_average_class") and file.endswith(".csv"):
                removed_data_files.append(os.path.join(root, file))
    return removed_data_files

def process_removed_data_csv(file_path):
    df = pd.read_csv(file_path)

    # Extract model and class information from the file path
    parts = file_path.split(os.sep)
    model_name = parts[-6]
    class_info = parts[-1].replace(".csv", "")
    df['file_path'] = file_path
    df['model_name'] = model_name
    df['class_info'] = class_info
    return df

def combine_removed_data_csvs(removed_data_files):
    combined_data = {}
    column_order = []  # To track the order of columns
    
    for file in removed_data_files:
        print(f"Processing file: {file}")  # Debug statement
        df = process_removed_data_csv(file)
        for idx, row in df.iterrows():
            image = row['image']  # Ensure correct column name
            gt = row['gt']
            avg_mc_pred_length = row['avg_mc_pred_length']
            model_name = row['model_name']
            class_info = row['class_info']
            column_name = f'{model_name}_{class_info}'
            
            if column_name not in column_order:
                column_order.append(column_name)
            
            if image not in combined_data:
                combined_data[image] = {'image': image, 'gt': gt, 'avg_mc_pred_length': avg_mc_pred_length,
                                        'total_removed_count': 0, 'removed_class_0_count': 0, 
                                        'removed_class_1_count': 0, 'removed_class_2_count': 0}
            
            combined_data[image][column_name] = True
            combined_data[image]['total_removed_count'] += 1
            class_num = int(class_info.split('_')[-1])
            combined_data[image][f'removed_class_{class_num}_count'] += 1

    # Calculate model overall removals
    for image, data in combined_data.items():
        for model in set([col.split('_')[0] for col in column_order]):
            model_total = sum(data.get(f'{model}_removed_data_by_floor_average_class_{i}', False) for i in range(3))
            data[f'{model}_removed_overall'] = model_total
    
    combined_df = pd.DataFrame.from_dict(combined_data, orient='index')
    combined_df.fillna(False, inplace=True)  # Fill missing values with False
    # Reorder columns according to column_order
    column_order_with_overall = []
    for model_class in column_order:
        model = model_class.split('_')[0]
        if f'{model}_removed_overall' not in column_order_with_overall:
            column_order_with_overall.append(f'{model}_removed_overall')
        column_order_with_overall.append(model_class)

    combined_df = combined_df[['image', 'gt', 'avg_mc_pred_length'] + column_order_with_overall + 
                              ['removed_class_0_count', 'removed_class_1_count', 'removed_class_2_count', 'total_removed_count']]
    return combined_df

def main(base_dirs, output_csv):
    all_removed_data_files = []
    for base_dir in base_dirs:
        print(f"Searching in base directory: {base_dir}")  # Debug statement
        removed_data_files = find_removed_data_csvs(base_dir)
        all_removed_data_files.extend(removed_data_files)
    
    print(f"Found {len(all_removed_data_files)} removed data files")  # Debug statement
    combined_df = combine_removed_data_csvs(all_removed_data_files)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved to {output_csv}")

if __name__ == "__main__":
    model_paths = [
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/3/lac/alpha_010",
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_DN121/inference_test2/3/lac/alpha_010",
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/RN50/inference_test2/3/lac/alpha_010",
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_RN50/inference_test2/3/lac/alpha_010",
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/EffNetV2S/inference_test2/3/lac/alpha_010",
        "/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010",
    ]
    output_csv = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/removed_images_best_models_test2.csv'

    main(model_paths, output_csv)
