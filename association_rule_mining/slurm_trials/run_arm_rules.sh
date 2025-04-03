#!/bin/bash
#SBATCH --job-name=arm_rules
#SBATCH --output=arm_rules_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4         # e.g. 4 cores for local Spark
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --mail-user=kw3215@cumc.columbia.edu
#SBATCH --mail-type=ALL

# 1) Load conda
module load conda/3

# 2) Activate your user environment ("arm_env") in your home directory
conda activate ~/arm_env

# 3) Provide arguments to your Python script as needed.
SUBJECT_LIST="INS-W_166#INS-W_1 INS-W_023#INS-W_1 INS-W_116#INS-W_1 INS-W_137#INS-W_1 INS-W_072#INS-W_1"
INPUT_DATA_PATH="/groups/xx2489_gp/kw3215/Datasets/globem/INS-W_1/FeatureData/rapids.csv"
OUTPUT_FILE="arm_behavior_rules.csv"

# 4) Run the Python script in local Spark mode (no spark-submit needed).
python arm_rules_spark.py \
    --subjects "$SUBJECT_LIST" \
    --data_path "$INPUT_DATA_PATH" \
    --output "$OUTPUT_FILE" \
    --min_support 0.3 \
    --min_confidence 0.7
