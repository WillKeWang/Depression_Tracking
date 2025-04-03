#!/usr/bin/env python3
"""
arm_rules_spark.py

A Python script to run Spark-based Association Rule Mining 
on 4-week user data with 33rd/66th percentile discretization.

Example usage:
  spark-submit --master local[*] arm_rules_spark.py \
      --subjects "INS-W_166#INS-W_1 INS-W_023#INS-W_1" \
      --data_path /path/to/rapids.csv \
      --output my_rules.csv \
      --min_support 0.3 \
      --min_confidence 0.7
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Association Rule Mining with Spark FPGrowth on 4-week user data.")
    parser.add_argument("--subjects", nargs="+", required=True, 
                        help="List of subject IDs to include (space-separated).")
    parser.add_argument("--data_path", default="data/INS-W/1/rapids.csv", 
                        help="Path to the input features CSV.")
    parser.add_argument("--output", default="arm_behavior_rules.csv", 
                        help="Output file name for rules (CSV).")
    parser.add_argument("--min_support", type=float, default=0.5, 
                        help="Minimum support threshold (fraction of transactions).")
    parser.add_argument("--min_confidence", type=float, default=0.7, 
                        help="Minimum confidence threshold.")
    args = parser.parse_args()

    subjects = args.subjects
    data_path = args.data_path
    out_file = args.output
    min_support = args.min_support
    min_confidence = args.min_confidence

    print("=== Spark ARM Script ===")
    print("Subjects:", subjects)
    print("Data path:", data_path)
    print("Output file:", out_file)
    print(f"minSupport={min_support}, minConfidence={min_confidence}")

    # 1) Initialize SparkSession
    #    If you run in local mode, you'd do .master("local[*]")
    #    But we'll let the Slurm script handle spark-slurm and set master dynamically.
    spark = (SparkSession.builder
             .appName("ARM_Spark_FPGrowth")
             .getOrCreate())

    # 2) Load data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df_sub = df.filter(F.col("pid").isin(subjects))

    # Ensure date is date/timestamp
    if dict(df_sub.dtypes).get("date") not in ("date", "timestamp"):
        df_sub = df_sub.withColumn("date", F.to_date("date"))

    # Keep only subset of features (Xu's four categories). 
    # Adjust if your dataset has different naming.
    feature_prefixes = ["f_slp", "f_steps", "f_screen", "f_loc"]
    cols_to_keep = [c for c in df_sub.columns if any(c.startswith(pref) for pref in feature_prefixes)]
    # Include pid and date
    cols_to_keep = ["pid", "date"] + cols_to_keep
    df_sub = df_sub.select(*cols_to_keep)

    # 3) Identify 4-week (28-day) window per user
    max_dates = df_sub.groupBy("pid").agg(F.max("date").alias("max_date"))
    df_joined = df_sub.join(max_dates, on="pid")
    df_window = df_joined.filter(
        (F.col("date") >= F.date_sub(F.col("max_date"), 27)) &
        (F.col("date") <= F.col("max_date"))
    )

    # 4) Compute 33rd and 66th percentile thresholds for each user
    data_pd = df_window.select("pid", "date",
                               *[c for c in df_window.columns if c not in ("pid","date","max_date")]).toPandas()
    numeric_features = [c for c in data_pd.columns if c not in ("pid","date")]
    user_thresholds = {}

    for pid, group in data_pd.groupby("pid"):
        thresholds = {}
        for feat in numeric_features:
            vals = group[feat].dropna().values
            if len(vals) == 0:
                continue
            p33 = np.percentile(vals, 33)
            p66 = np.percentile(vals, 66)
            thresholds[feat] = (p33, p66)
        user_thresholds[pid] = thresholds

    def row_to_items(pid, row):
        """Convert a row of features to Low/Med/High items based on user-specific thresholds."""
        items = []
        thr = user_thresholds.get(pid, {})
        for feat, value in row.items():
            if feat in ("pid","date") or pd.isna(value):
                continue
            if feat in thr:
                p33, p66 = thr[feat]
                if value <= p33:
                    items.append(f"{feat}_low")
                elif value <= p66:
                    items.append(f"{feat}_medium")
                else:
                    items.append(f"{feat}_high")
        return items

    # 5) Create transactions
    transactions = []
    for pid, group in data_pd.groupby("pid"):
        for _, row in group.iterrows():
            items = row_to_items(pid, row)
            if items:
                tx_id = f"{pid}_{row['date']}"
                transactions.append((tx_id, items))

    # 6) Build Spark DataFrame of transactions
    transactions_df = spark.createDataFrame(transactions, ["id", "items"])

    # 7) Run FPGrowth
    fpGrowth = FPGrowth(itemsCol="items", 
                        minSupport=min_support, 
                        minConfidence=min_confidence)
    model = fpGrowth.fit(transactions_df)

    # 8) Extract association rules
    rules_df = model.associationRules
    rules_pd = rules_df.toPandas()

    # Convert antecedent/consequent arrays to strings
    if not rules_pd.empty:
        rules_pd["antecedent"] = rules_pd["antecedent"].apply(
            lambda arr: ";".join(sorted(map(str, arr))))
        rules_pd["consequent"] = rules_pd["consequent"].apply(
            lambda arr: ";".join(sorted(map(str, arr))))

    rules_pd.to_csv(out_file, index=False)
    print(f"Association rule mining complete. Found {len(rules_pd)} rules. Saved to {out_file}")

    spark.stop()

if __name__ == "__main__":
    main()
