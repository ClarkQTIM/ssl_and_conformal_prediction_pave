import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import os
import sys

# Prepare dataframe

def prepare_df(removed_images_df, df_with_info, brian_df, save_dir):

    removed_images_df = pd.read_csv(removed_images_df)
    info_df = pd.read_csv(df_with_info)
    merged_df = pd.merge(removed_images_df, info_df[['MASKED_IMG_ID', 'ID']], 
                     left_on='image', right_on='MASKED_IMG_ID', how='left')

    # Drop the MASKED_IMG_ID column if not needed
    merged_df.drop(columns=['MASKED_IMG_ID'], inplace=True)
    brian_df = pd.read_csv(brian_df)
    # Merge the DataFrames on the ID column
    final_merged_df = pd.merge(merged_df, brian_df, 
                            on='ID', how='left')

    final_merged_df.to_csv(os.path.join(save_dir, 'removed_images_with_all_info.csv'))

    return final_merged_df

# Function to calculate and add correlation text to plots
def add_correlation_text(ax, x, y, data, title, ordinal=False):
    if ordinal:
        corr, p_value = spearmanr(data[x], data[y])
        corr_type = "Spearman's ρ"
    else:
        corr, p_value = pearsonr(data[x], data[y])
        corr_type = "Pearson's r"

    ax.text(0.05, 0.95, f'{corr_type} = {corr:.2f}\np = {p_value:.2g}', transform=ax.transAxes, 
            fontsize=14, verticalalignment='top')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)

# Analysis function for various correlations
def run_analysis(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    models = ['Model36', 'DINO_DN121', 'RN50', 'DINO_RN50', 'EffNetV2S', 'DINO_EffNetV2S']
    classes = [0, 1, 2]

    # Correlation Analysis: removed_class_X_count vs avg_mc_pred_length
    for y_col in ['removed_class_0_count', 'removed_class_1_count', 'removed_class_2_count']:
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(x='avg_mc_pred_length', y=y_col, data=data)
        add_correlation_text(ax, 'avg_mc_pred_length', y_col, data, 
                             f'{y_col.replace("_", " ").title()} vs Avg MC Pred Length', ordinal=False)
        plot_path = os.path.join(save_dir, f'{y_col}_vs_avg_mc_pred_length.png')
        plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close()

    # Correlation Analysis: HPV_STRAT vs avg_mc_pred_length
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='HPV_STRAT', y='avg_mc_pred_length', data=data)
    add_correlation_text(ax, 'HPV_STRAT', 'avg_mc_pred_length', data, 
                         'HPV STRAT vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'HPV_STRAT_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    # Correlation Analysis: CC_ST vs avg_mc_pred_length
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='CC_ST', y='avg_mc_pred_length', data=data)
    add_correlation_text(ax, 'CC_ST', 'avg_mc_pred_length', data, 
                         'CC ST vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'CC_ST_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    # Subset 1: total_removed_count is 5 or 6
    subset1 = data[(data['total_removed_count'].isin([5, 6]))]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='HPV_STRAT', y='avg_mc_pred_length', data=subset1)
    add_correlation_text(ax, 'HPV_STRAT', 'avg_mc_pred_length', subset1, 
                         'Subset (Total Removed Count 5 or 6): HPV STRAT vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'subset1_HPV_STRAT_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='CC_ST', y='avg_mc_pred_length', data=subset1)
    add_correlation_text(ax, 'CC_ST', 'avg_mc_pred_length', subset1, 
                         'Subset (Total Removed Count 5 or 6): CC ST vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'subset1_CC_ST_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    # Subset 2: removed_class_1_count is 5 or 6, CC_ST 2,3,4 only
    subset2 = data[(data['removed_class_1_count'].isin([5, 6])) & (data['CC_ST'].isin([2, 3, 4]))]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='CC_ST', y='avg_mc_pred_length', data=subset2)
    add_correlation_text(ax, 'CC_ST', 'avg_mc_pred_length', subset2, 
                         'Subset (Removed Class 1 Count 5 or 6, CC_ST 2,3,4): CC ST vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'subset2_CC_ST_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    # Subset 3: removed_class_2_count is 5 or 6, CC_ST 5,6 only
    subset3 = data[(data['removed_class_2_count'].isin([1, 2, 3, 4, 5, 6])) & (data['CC_ST'].isin([5, 6]))]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='CC_ST', y='avg_mc_pred_length', data=subset3)
    add_correlation_text(ax, 'CC_ST', 'avg_mc_pred_length', subset3, 
                         'Subset (Removed Class 2 Count 5 or 6, CC_ST 5,6): CC ST vs Avg MC Pred Length', ordinal=True)
    plot_path = os.path.join(save_dir, 'subset3_CC_ST_vs_avg_mc_pred_length.png')
    plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')
    plt.close()

    # Print out correlation results to a text file
    with open(os.path.join(save_dir, 'correlation_results.txt'), 'w') as f:
        sys.stdout = f
        
        for y_col in ['removed_class_0_count', 'removed_class_1_count', 'removed_class_2_count']:
            pearson_corr, pearson_p = pearsonr(data['avg_mc_pred_length'], data[y_col])
            print(f"Pearson's r ({y_col} vs avg_mc_pred_length): {pearson_corr}, p-value: {pearson_p}")

        spearman_corr, spearman_p = spearmanr(data['HPV_STRAT'], data['avg_mc_pred_length'])
        print(f"Spearman's ρ (HPV_STRAT vs avg_mc_pred_length): {spearman_corr}, p-value: {spearman_p}")
        
        spearman_corr, spearman_p = spearmanr(data['CC_ST'], data['avg_mc_pred_length'])
        print(f"Spearman's ρ (CC_ST vs avg_mc_pred_length): {spearman_corr}, p-value: {spearman_p}")
        
        print("\n--- Subset Analyses ---")
        
        # Subset 1 correlations
        subset1_spearman_corr, subset1_spearman_p = spearmanr(subset1['HPV_STRAT'], subset1['avg_mc_pred_length'])
        print(f"Subset 1: Spearman's ρ (HPV_STRAT vs avg_mc_pred_length): {subset1_spearman_corr}, p-value: {subset1_spearman_p}")

        subset1_spearman_corr, subset1_spearman_p = spearmanr(subset1['CC_ST'], subset1['avg_mc_pred_length'])
        print(f"Subset 1: Spearman's ρ (CC_ST vs avg_mc_pred_length): {subset1_spearman_corr}, p-value: {subset1_spearman_p}")

        # Subset 2 correlations
        subset2_spearman_corr, subset2_spearman_p = spearmanr(subset2['CC_ST'], subset2['avg_mc_pred_length'])
        print(f"Subset 2: Spearman's ρ (CC_ST 2,3,4 vs avg_mc_pred_length): {subset2_spearman_corr}, p-value: {subset2_spearman_p}")

        # Subset 3 correlations
        subset3_spearman_corr, subset3_spearman_p = spearmanr(subset3['CC_ST'], subset3['avg_mc_pred_length'])
        print(f"Subset 3: Spearman's ρ (CC_ST 5,6 vs avg_mc_pred_length): {subset3_spearman_corr}, p-value: {subset3_spearman_p}")
        
        # New Subset Analyses for each model and class
        for model in models:
            for class_idx in classes:
                subset = data[data[f'{model}_removed_data_by_floor_average_class_{class_idx}'] == 1]

                if not subset.empty:
                    # Spearman correlation: HPV_STRAT vs avg_mc_pred_length
                    subset_spearman_corr, subset_spearman_p = spearmanr(subset['HPV_STRAT'], subset['avg_mc_pred_length'])
                    print(f"{model} - Class {class_idx}: Spearman's ρ (HPV_STRAT vs avg_mc_pred_length): {subset_spearman_corr}, p-value: {subset_spearman_p}")

                    # Spearman correlation: CC_ST vs avg_mc_pred_length (all CC_ST values)
                    subset_spearman_corr, subset_spearman_p = spearmanr(subset['CC_ST'], subset['avg_mc_pred_length'])
                    print(f"{model} - Class {class_idx}: Spearman's ρ (CC_ST vs avg_mc_pred_length): {subset_spearman_corr}, p-value: {subset_spearman_p}")


                    # Additional analysis for specific CC_ST values
                    if class_idx == 1:
                        subset_specific = subset[subset['CC_ST'].isin([2, 3, 4])]
                        if not subset_specific.empty:
                            subset_spearman_corr, subset_spearman_p = spearmanr(subset_specific['CC_ST'], subset_specific['avg_mc_pred_length'])
                            print(f"{model} - Class {class_idx} (CC_ST 2,3,4): Spearman's ρ (CC_ST vs avg_mc_pred_length): {subset_spearman_corr}, p-value: {subset_spearman_p}")

                    elif class_idx == 2:
                        subset_specific = subset[subset['CC_ST'].isin([5, 6])]
                        if not subset_specific.empty:
                            subset_spearman_corr, subset_spearman_p = spearmanr(subset_specific['CC_ST'], subset_specific['avg_mc_pred_length'])
                            print(f"{model} - Class {class_idx} (CC_ST 5,6): Spearman's ρ (CC_ST vs avg_mc_pred_length): {subset_spearman_corr}, p-value: {subset_spearman_p}")

    # Reset stdout
    sys.stdout = sys.__stdout__

# Example usage:
# data = pd.read_csv('your_data.csv')  # Load your data here
# save_dir = 'output_directory'
# run_analysis(data, save_dir)



if __name__ == "__main__":

    # Getting the prepped dataframe
    removed_images_df_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/removed_images_best_models_test2.csv'
    df_with_info_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/csvs/SEED_three_class_ground_truths.csv'
    brian_df_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/csvs/testset_endpoint_20240716.csv'
    # prepped_df = prepare_df(removed_images_df_path, df_with_info_path, brian_df_path, overall_analysis_save_dir)

    # Assuming the dataframe has already been prepared as `prepped_df`
    analysis_save_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/lac_alpha010_fine_grain_gt_analysis/correlation_analysis'
    os.makedirs(analysis_save_dir, exist_ok=True)
    prepped_df = prepare_df(removed_images_df_path, df_with_info_path, brian_df_path, analysis_save_dir)
    run_analysis(prepped_df, analysis_save_dir)
