#########
# Imports
#########

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import os
import sys

###########
# Functions
###########

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

# Function to add correlation coefficient to plots
def add_correlation_text(ax, x, y, data, title):
    
    corr, p_value = pointbiserialr(data[x], data[y])
    ax.text(0.05, 0.95, f'r = {corr:.2f}\np = {p_value:.2g}', transform=ax.transAxes, 
            fontsize=14, verticalalignment='top')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x, fontsize=14)
    ax.set_ylabel(y, fontsize=14)

# Analysis with Brian's ground truth
def run_brian_analysis(data, hpv_plot_save_path, cc_st_plot_save_path, save_dir):

    # Ensure directories exist
    os.makedirs(hpv_plot_dir, exist_ok=True)
    os.makedirs(cc_st_plot_dir, exist_ok=True)

    # Analysis for total_removed_count
    for y_col in ['total_removed_count', 'removed_class_0_count', 'removed_class_1_count', 'removed_class_2_count']:
        # Distribution Analysis: HPV_STRAT vs y_col
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='HPV_STRAT', y=y_col, data=data)
        add_correlation_text(ax, 'HPV_STRAT', y_col, data, f'Distribution of {y_col.replace("_", " ").title()} by HPV_STRAT')
        plot_path = os.path.join(hpv_plot_dir, f'HPV_STRAT_vs_{y_col}_boxplot.png')
        plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')

        # Distribution Analysis: CC_ST vs y_col
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='CC_ST', y=y_col, data=data)
        add_correlation_text(ax, 'CC_ST', y_col, data, f'Distribution of {y_col.replace("_", " ").title()} by CC_ST')
        plot_path = os.path.join(cc_st_plot_dir, f'CC_ST_vs_{y_col}_boxplot.png')
        plt.savefig(plot_path, facecolor='w', edgecolor='w', bbox_inches='tight')

        # Filter data for CC_ST values 5 and 6
        data_cc_56 = data[data['CC_ST'].isin([5, 6])]

        # Boxplot for CC_ST 5 and 6 vs y_col
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='CC_ST', y=y_col, data=data_cc_56)
        add_correlation_text(ax, 'CC_ST', y_col, data_cc_56, f'Distribution of {y_col.replace("_", " ").title()} by CC_ST (5 and 6)')
        plot_path_56 = os.path.join(cc_st_plot_dir, f'CC_ST_5_6_vs_{y_col}_boxplot.png')
        plt.savefig(plot_path_56, facecolor='w', edgecolor='w', bbox_inches='tight')

        # Filter data for CC_ST values 2, 3, and 4
        data_cc_234 = data[data['CC_ST'].isin([2, 3, 4])]

        # Boxplot for CC_ST 2, 3, and 4 vs y_col
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='CC_ST', y=y_col, data=data_cc_234)
        add_correlation_text(ax, 'CC_ST', y_col, data_cc_234, f'Distribution of {y_col.replace("_", " ").title()} by CC_ST (2, 3, and 4)')
        plot_path_234 = os.path.join(cc_st_plot_dir, f'CC_ST_2_3_4_vs_{y_col}_boxplot.png')
        plt.savefig(plot_path_234, facecolor='w', edgecolor='w', bbox_inches='tight')

    with open(os.path.join(save_dir, 'printout_correlations.txt'), 'w') as f:
        sys.stdout = f
        # Correlation Analysis
        for y_col in ['total_removed_count', 'removed_class_0_count', 'removed_class_1_count', 'removed_class_2_count']:
            hpv_strat_corr, hpv_strat_p = pointbiserialr(data['HPV_STRAT'], data[y_col])
            cc_st_corr, cc_st_p = pointbiserialr(data['CC_ST'], data[y_col])

            print(f"HPV_STRAT vs {y_col.replace('_', ' ').title()} Correlation: {hpv_strat_corr}, p-value: {hpv_strat_p}")
            print(f"CC_ST vs {y_col.replace('_', ' ').title()} Correlation: {cc_st_corr}, p-value: {cc_st_p}")

        sys.stdout = sys.__stdout__

if __name__ == "__main__":

    # Getting the prepped dataframe
    removed_images_df_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/removed_images_best_models_test2.csv'
    df_with_info_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/csvs/SEED_three_class_ground_truths.csv'
    brian_df_path = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/csvs/testset_endpoint_20240716.csv'
    overall_analysis_save_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/lac_alpha010_fine_grain_gt_analysis'
    os.makedirs(overall_analysis_save_dir, exist_ok=True)
    prepped_df = prepare_df(removed_images_df_path, df_with_info_path, brian_df_path, overall_analysis_save_dir)
    print(prepped_df.columns)

    # Analysis
    ## Define directories for saving plots
    overall_analysis_save_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/lac_alpha010_fine_grain_gt_analysis'
    hpv_plot_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/lac_alpha010_fine_grain_gt_analysis/HPV_Relationships'
    cc_st_plot_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL_PAVE/analysis/test2/SSL/M36_Pipeline/M36_Norms/lac_alpha010_fine_grain_gt_analysis/CC_ST_Relationships'

    run_brian_analysis(prepped_df, hpv_plot_dir, cc_st_plot_dir, overall_analysis_save_dir)
