import os
import pandas as pd

def find_and_combine_csvs(base_dir, models, dataset, num_classes, conformal_algo, alpha_value, output_file):
    all_data = []

    for model in models:
        model_dir = os.path.join(base_dir, model, 'inference_' + dataset, str(num_classes), conformal_algo, alpha_value)
        model_csv = os.path.join(model_dir, 'analysis_alpha_010')
        df = pd.read_csv(model_csv)
        all_data.append(df)
    
    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Model'] = models
    combined_df = combined_df.round(3)
    combined_df = combined_df[['avg_pred_set_length_whole_df', 'ci_avg_pred_set_length_whole_df',
    'avg_pred_set_length_normal_only', 'ci_avg_pred_set_length_normal_only',
    'avg_pred_set_length_gz_only', 'ci_avg_pred_set_length_gz_only',
    'avg_pred_set_length_pcplus_only', 'ci_avg_pred_set_length_pcplus_only',
    'Model']]
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(combined_df.columns)
    print(f"Combined CSV saved to {output_file}")
    print(combined_df[['avg_pred_set_length_whole_df', 'ci_avg_pred_set_length_whole_df',
    'avg_pred_set_length_normal_only', 'ci_avg_pred_set_length_normal_only',
    'avg_pred_set_length_gz_only', 'ci_avg_pred_set_length_gz_only',
    'avg_pred_set_length_pcplus_only', 'ci_avg_pred_set_length_pcplus_only',
    'Model']])

# Example usage
base_directory = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/'
models_list = ['Model36', 'DINO_DN121', 'RN50', 'DINO_RN50', 'EffNetV2S', 'DINO_EffNetV2S']  # Add other models as needed
dataset_name = 'test2'
num_classes = 3
conformal_algorithm = 'lac'
alpha_value = 'alpha_010'
output_csv = os.path.join(base_directory, 'overall_averages_six_models.csv')

find_and_combine_csvs(base_directory, models_list, dataset_name, num_classes, conformal_algorithm, alpha_value, output_csv)
