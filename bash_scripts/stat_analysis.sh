#!/bin/bash

# # ===========================================
# # Run DeLong Statistical Test for AUCs
# # ===========================================

# PYTHON_SCRIPT="/scratch90/chris/ssl_conformal_pave/src/delong.py"
# PROBS_COL="pred_probs_mc"
# GT_COL="gt"
# CLASS_INDICES="0 2"
# CLASS_NAMES="Normal Severe"

# ===== Model Pairs =====
declare -a PAIRS=(
  # Format: "name1|path1|name2|path2|output_dir_name"
  "dino_dn121|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|dn121|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|dino_vs_dn121"
  "dino_effnetv2s|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_effnetv2s/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|effnetv2s|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/effnetv2s/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|dino_vs_effnetv2s"
  "dino_rn50|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_rn50/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|rn50|/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/rn50/3/lac/alpha_010/mc_lac_run_alpha0.1.csv|dino_vs_rn50"
)

# # ===== Output Base Directory =====
# OUT_BASE="/scratch90/chris/ssl_conformal_pave/analysis/delong"
# mkdir -p "$OUT_BASE"

# # ===== Loop through model pairs =====
# for entry in "${PAIRS[@]}"; do
#     IFS="|" read -r NAME1 PATH1 NAME2 PATH2 OUTNAME <<< "$entry"

#     echo ""
#     echo "🔬 DeLong Test: $NAME1 vs $NAME2"
#     echo "→ Output: $OUT_BASE/$OUTNAME"

#     python "$PYTHON_SCRIPT" \
#         --model1_csv "$PATH1" \
#         --model2_csv "$PATH2" \
#         --output_dir "$OUT_BASE/$OUTNAME" \
#         --probs_col "$PROBS_COL" \
#         --gt_col "$GT_COL" \
#         --class_indices $CLASS_INDICES \
#         --class_names $CLASS_NAMES
# done

# echo ""
# echo "✅ DeLong testing complete."

# ===========================================
# Run Two-Sided T-Test on Prediction Set Lengths
# ===========================================

TTEST_SCRIPT="/scratch90/chris/ssl_conformal_pave/src/t_test.py"
LENGTH_COL="pred_set_mc_length"
TTEST_OUT_BASE="/scratch90/chris/ssl_conformal_pave/analysis/t_test"
SUBSET_COL="gt"
CLASS_INDICES="0 1 2"
CLASS_NAMES="Normal Indeterminate Severe"

mkdir -p "$TTEST_OUT_BASE"

echo ""
echo "📏 Running two-sided t-tests on prediction set lengths..."

for entry in "${PAIRS[@]}"; do
    IFS="|" read -r NAME1 PATH1 NAME2 PATH2 OUTNAME <<< "$entry"

    echo ""
    echo "🧪 T-Test: $NAME1 vs $NAME2"
    echo "→ Output: $TTEST_OUT_BASE/$OUTNAME"

    python "$TTEST_SCRIPT" \
        --model1_csv "$PATH1" \
        --model2_csv "$PATH2" \
        --output_dir "$TTEST_OUT_BASE/$OUTNAME" \
        --length_col "$LENGTH_COL" \
        --subset_col "$SUBSET_COL" \
        --class_indices $CLASS_INDICES \
        --class_names $CLASS_NAMES
done

echo ""
echo "✅ Two-sided t-tests complete."
