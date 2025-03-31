#!/bin/bash
#SBATCH --job-name=arm_rules                  # Job name
#SBATCH --output=arm_rules_%j.log             # Standard output and error log
#SBATCH --nodes=1                             # Run on a single node
#SBATCH --cpus-per-task=8                     # Number of CPU cores to use
#SBATCH --mem=128G                            # Total memory for the job
#SBATCH --time=0-03:00:00                     # Time limit (HH:MM:SS)
#SBATCH --mail-user=kw3215@cumc.columbia.edu  # use only Columbia address
#SBATCH --mail-type=ALL                       # send email alert on all events

# Load necessary modules (if applicable)
module load spark
module load python

# Activate virtual environment if needed (optional)
# source ~/envs/pyspark-env/bin/activate

# Run the Spark FPGrowth ARM script for the specified users
SUBJECT_LIST="INS-W_166#INS-W_1 INS-W_023#INS-W_1 INS-W_116#INS-W_1 INS-W_137#INS-W_1 INS-W_072#INS-W_1"
INPUT_DATA_PATH="/groups/xx2489_gp/kw3215/Datasets/globem/INS-W_1/FeatureData/rapids.csv"
OUTPUT_FILE="arm_behavior_rules.csv"

# Use spark-submit to run the PySpark application
spark-submit --master local[$SLURM_CPUS_PER_TASK] \
    arm_rules_spark.py --subjects $SUBJECT_LIST \
    --data_path $INPUT_DATA_PATH --output $OUTPUT_FILE \
    --min_support 0.3 --min_confidence 0.7
