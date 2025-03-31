#!/usr/bin/env python3
import sys
import argparse
import math
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Association Rule Mining with Spark FPGrowth on 4-week user data.")
parser.add_argument("--subjects", nargs="+", required=True, help="List of subject IDs to include (space-separated).")
parser.add_argument("--data_path", default="data/INS-W/1/rapids.csv", 
                    help="Path to the input features CSV (default assumes original directory structure).")
parser.add_argument("--output", default="arm_behavior_rules.csv", help="Output file name for rules (CSV).")
parser.add_argument("--min_support", type=float, default=0.5, help="Minimum support threshold (fraction of transactions).")
parser.add_argument("--min_confidence", type=float, default=0.7, help="Minimum confidence threshold.")
args = parser.parse_args()

subjects = args.subjects
data_path = args.data_path
out_file = args.output
min_support = args.min_support
min_confidence = args.min_confidence

# Initialize SparkSession (use all available cores on one node by default)
spark = SparkSession.builder \
    .appName("ARM_Spark_FPGrowth") \
    .master("local[*]") \
    .getOrCreate()

# Read the daily features dataset
# Infers schema to get correct data types; adjust path as needed for actual data location
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Filter to the specified subjects
df_sub = df.filter(F.col("pid").isin(subjects))

# Ensure date is in proper date format (if not inferred, cast explicitly)
if dict(df_sub.dtypes).get("date") not in ("date", "timestamp"):
    df_sub = df_sub.withColumn("date", F.to_date("date"))

# Keep only Xu's 33 unimodal features (columns starting with f_slp, f_steps, f_screen, f_loc), plus pid and date
feature_prefixes = ["f_slp", "f_steps", "f_screen", "f_loc"]
cols_to_keep = [c for c in df_sub.columns if any(c.startswith(pref) for pref in feature_prefixes)]
# Include pid and date for window filtering
cols_to_keep = ["pid", "date"] + cols_to_keep
df_sub = df_sub.select(*cols_to_keep)

# Identify 4-week (28-day) window for each user: [max_date - 27 days, max_date] per user
# Get max date per user
max_dates = df_sub.groupBy("pid").agg(F.max("date").alias("max_date"))
# Join to have max_date available on each record, then filter records within 28 days of max_date
df_joined = df_sub.join(max_dates, on="pid")
df_window = df_joined.filter(
    (F.col("date") >= F.date_sub(F.col("max_date"), 27)) & (F.col("date") <= F.col("max_date"))
)

# Compute 33rd and 66th percentiles for each feature *within each user's 4-week window*
# We will collect these thresholds to driver for simplicity (since number of users is small)
user_thresholds = {}
numeric_features = [c for c in df_window.columns if c not in ("pid", "date", "max_date")]
# Collect the filtered 4-week data to Pandas for percentile calculation (safe for small number of records)
data_pd = df_window.select("pid", "date", *numeric_features).toPandas()
for pid, group in data_pd.groupby("pid"):
    # Compute per-feature thresholds for this user's 4-week data
    thresholds = {}
    for feat in numeric_features:
        # Drop NaNs for percentile calculation
        vals = group[feat].dropna().values
        if len(vals) == 0:
            continue  # no data for this feature in window
        # 33rd and 66th percentiles
        p33 = np.percentile(vals, 33)
        p66 = np.percentile(vals, 66)
        thresholds[feat] = (p33, p66)
    user_thresholds[pid] = thresholds

# UDF to discretize one row’s features into a list of items (Low/Med/High), skipping missing
def row_to_items(pid, row):
    """Convert a pandas Series (one day of features) to a list of categorical items."""
    items = []
    thr = user_thresholds.get(pid, {})
    for feat, value in row.items():
        if feat in ("pid", "date") or pd.isna(value):
            continue
        # Determine category based on thresholds
        if feat in thr:
            p33, p66 = thr[feat]
            if value <= p33:
                items.append(f"{feat}_low")
            elif value <= p66:
                items.append(f"{feat}_medium")
            else:
                items.append(f"{feat}_high")
        else:
            # If feature threshold not computed (all NaN in window), skip it
            continue
    return items

# Create transactions: list of items per day for each user’s window
transactions = []
for pid, group in data_pd.groupby("pid"):
    for _, row in group.iterrows():
        items = row_to_items(pid, row)
        if items:
            transactions.append((f"{pid}_{row['date']}", items))  # use pid_date as transaction ID

# Convert transactions to Spark DataFrame (columns: id, items)
transactions_df = spark.createDataFrame(transactions, schema=["id", "items"])

# Run FPGrowth on the transactions
fp = F.FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
model = fp.fit(transactions_df)

# Extract association rules
rules_df = model.associationRules
# Convert antecedent & consequent arrays to string for output
rules_pd = rules_df.toPandas()
if not rules_pd.empty:
    rules_pd["antecedent"] = rules_pd["antecedent"].apply(lambda x: ";".join(sorted(map(str, x))))
    rules_pd["consequent"] = rules_pd["consequent"].apply(lambda x: ";".join(sorted(map(str, x))))
# Save rules to CSV (include header)
rules_pd.to_csv(out_file, index=False)

print(f"Association rules mining complete. Found {len(rules_pd)} rules. Saved to {out_file}")
spark.stop()
