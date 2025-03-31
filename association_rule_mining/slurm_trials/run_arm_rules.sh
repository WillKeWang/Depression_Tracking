#!/bin/bash
#SBATCH --job-name=arm_rules                  # Job name
#SBATCH --output=arm_rules_%j.log             # Standard output & error log
#SBATCH --nodes=2                             # Allocate at least 2 nodes
#SBATCH --cpus-per-task=8                     # Number of CPU cores per node
#SBATCH --mem=128G                            # Total memory per node
#SBATCH --time=0-03:00:00                     # Time limit (HH:MM:SS)
#SBATCH --mail-user=kw3215@cumc.columbia.edu  # Columbia address
#SBATCH --mail-type=ALL                       # Send email on all events

# 1) Load Spark (required for spark-slurm). 
#    If you also need Anaconda for custom Python env, you can load it as well,
#    but ensure 'spark' is definitely loaded.
module load spark
module load anaconda

# 2) Set JAVA_HOME as required by Spark on this cluster
export JAVA_HOME=/usr

# 3) Start Spark in standalone cluster mode via spark-slurm
SPARK_LOG=~/.spark/spark-${SLURM_JOB_ID}.log
spark-slurm > "$SPARK_LOG" &
sleep 20   # Give the Spark master & workers time to start

# 4) Extract the Spark master URL from the log
sparkmaster=$(awk '/master:/ {print $NF}' "$SPARK_LOG")
echo "sparkmaster=$sparkmaster"

# ------------------------------------------------------------------------------
# Optional: Test your Spark cluster with the WordCount example
# ------------------------------------------------------------------------------
spark-submit --master "$sparkmaster" \
    "$SPARK_HOME/examples/src/main/python/wordcount.py" \
    "$SPARK_HOME/README.md"

# ------------------------------------------------------------------------------
# Now run your ARM (Association Rule Mining) job
# ------------------------------------------------------------------------------
SUBJECT_LIST="INS-W_166#INS-W_1 INS-W_023#INS-W_1 INS-W_116#INS-W_1 INS-W_137#INS-W_1 INS-W_072#INS-W_1"
INPUT_DATA_PATH="/groups/xx2489_gp/kw3215/Datasets/globem/INS-W_1/FeatureData/rapids.csv"
OUTPUT_FILE="arm_behavior_rules.csv"

spark-submit --master "$sparkmaster" \
    arm_rules_spark.py \
    --subjects "$SUBJECT_LIST" \
    --data_path "$INPUT_DATA_PATH" \
    --output "$OUTPUT_FILE" \
    --min_support 0.3 \
    --min_confidence 0.7
