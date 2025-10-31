import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

for item_type in config.ITEM_TYPES:
    for data_set in config.DATA_SETS:
        print(f"Combining original and core for {data_set} ({item_type})")
        fold_path_original = f"data/feature_folds/{data_set}/original/{item_type}"
        fold_path_core = f"data/feature_folds/{data_set}/core/{item_type}"
        fold_path_core_original = f"data/feature_folds/{data_set}/core_original/{item_type}"

        for fold in range(10):
            test_original = pd.read_csv(f"{fold_path_original}/test{fold}.csv")
            test_core = pd.read_csv(f"{fold_path_core}/test{fold}.csv")
            test = pd.merge(test_original, test_core.drop(columns=["targets"]), on=["item_id", "tag"], how="left")
            test.to_csv(f"{fold_path_core_original}/test{fold}.csv", index=False)

            train_original = pd.read_csv(f"{fold_path_original}/train{fold}.csv")
            train_core = pd.read_csv(f"{fold_path_core}/train{fold}.csv")
            train = pd.merge(train_original, train_core.drop(columns=["targets"]), on=["item_id", "tag"], how="left")
            train.to_csv(f"{fold_path_core_original}/train{fold}.csv", index=False)
        print("Done")