import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config


def evaluate_avg(in_path, out_path):
    all_mae = []
    for fold in range(10):
        train = pd.read_csv(f"{in_path}/train{fold}.csv")
        test = pd.read_csv(f"{in_path}/test{fold}.csv")

        test["pred"] = train.targets.mean()

        mae = mean_absolute_error(test.targets, test.pred)
        all_mae.append(mae)
    mean_mae = np.mean(all_mae)
    with open(f"{out_path}/log.txt", "w") as file:
        file.write(f"All MAE: {all_mae}\nAverage MAE: {mean_mae}\n")

for item_type in config.ITEM_TYPES:
    print(f"Calculating average for {item_type}...")
    in_path = f"data/feature_folds/item_tag_target_only/{item_type}"
    out_path = f"data/evaluation_results/average/{item_type}"
    evaluate_avg(in_path, out_path)
    print("Done.")