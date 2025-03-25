import os
import argparse
from data_loader.data_loader_ml import data_loader_single
from algorithm.ml_xu_interpretable import DepressionDetectionAlgorithm_ML_xu_interpretable
from algorithm.ml_xu_personalized import DepressionDetectionAlgorithm_ML_xu_personalized

# Define a helper load_dataset to wrap data_loader_single with default institution and phase.
def load_dataset(prediction_target, institution="INS-W", phase=1, flag_more_feat_types=False):
    return data_loader_single(prediction_target, institution, phase, flag_more_feat_types)

def main(prediction_target):
    print("Loading dataset...")
    dataset = load_dataset(prediction_target)
    
    # Run the interpretable model pipeline
    print("\n--- Running Interpretable Model Pipeline ---")
    interp_algo = DepressionDetectionAlgorithm_ML_xu_interpretable({"verbose": 1})
    data_repo_interp = interp_algo.prep_data_repo(dataset, flag_train=True)
    clf_interp = interp_algo.prep_model(data_repo_interp)
    clf_interp.fit(data_repo_interp.X, data_repo_interp.y)
    y_pred_interp = clf_interp.predict(data_repo_interp.X)
    print("Interpretable model predictions:", y_pred_interp)
    
    # Run the personalized model pipeline
    print("\n--- Running Personalized Model Pipeline ---")
    pers_algo = DepressionDetectionAlgorithm_ML_xu_personalized({"verbose": 1})
    data_repo_pers = pers_algo.prep_data_repo(dataset, flag_train=True)
    clf_pers = pers_algo.prep_model(data_repo_pers)
    clf_pers.fit(data_repo_pers.X, data_repo_pers.y)
    y_pred_pers = clf_pers.predict(data_repo_pers.X)
    print("Personalized model predictions:", y_pred_pers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run depression detection pipeline on GLOBEM data")
    parser.add_argument("--target", type=str, default="dep_weekly", help="Prediction target (e.g., dep_weekly)")
    args = parser.parse_args()
    main(args.target)
