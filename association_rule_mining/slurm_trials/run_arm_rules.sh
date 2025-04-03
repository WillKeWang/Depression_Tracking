#!/bin/bash
#SBATCH --job-name=arm_rules                  # Job name
#SBATCH --output=arm_rules_%j.log            # Standard output & error log
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --cpus-per-task=1                     # Number of CPU cores per node
#SBATCH --mem=32G                             # Total memory per node
#SBATCH --time=0-03:00:00                     # Time limit (HH:MM:SS)
#SBATCH --mail-user=kw3215@cumc.columbia.edu  # Email for notifications
#SBATCH --mail-type=ALL                       # Receive email on all events

# 1) Load conda & create/activate environment (if needed).
module load conda/3

# This portion is optional if you already have an environment set up.
# You might prefer to "conda activate some_preexisting_env" instead.
srun --pty /bin/bash
conda create -n spark-env python=3.8 -y
conda activate spark-env
conda install -c conda-forge pyspark apache-spark -y

# 2) Set JAVA_HOME if required for Spark
export JAVA_HOME=/usr

# 3) Start a Spark cluster via spark-slurm (adjust if your cluster uses a different Spark launcher).
SPARK_LOG=~/.spark/spark-${SLURM_JOB_ID}.log
spark-slurm > "$SPARK_LOG" &
sleep 20  # Give the Spark master/worker time to start

# 4) Extract the Spark master URL from the log
sparkmaster=$(awk '/master:/ {print $NF}' "$SPARK_LOG")
echo "sparkmaster=$sparkmaster"

# 5) Example test job (optional): WordCount
# spark-submit --master "$sparkmaster" \
#     "$SPARK_HOME/examples/src/main/python/wordcount.py" \
#     "$SPARK_HOME/README.md"

# 6) Now run your ARM (Association Rule Mining) job
#    Modify these as needed for your data, subject IDs, etc.
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
