import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

def print_results(path, test_path):
    all_mae = []
    for fold in range(10):
        test = pd.read_csv(f"{test_path}/test{fold}.csv")
        pred = pd.read_csv(f"{path}/predictions_fold_{fold}.txt", header=None)
        pred.columns = ["pred"]

        mae = mean_absolute_error(test.targets, pred.pred)
        all_mae.append(mae)
    mean_mae = np.mean(all_mae)
    with open(f"{path}/log.txt", "w") as file:
        file.write(f"All MAE: {all_mae}\nAverage MAE: {mean_mae}\n")

for item_type in config.ITEM_TYPES:
    print(f"Preparing results for GLMER ({item_type})")
    path = f"data/evaluation_results/glmer/tg/{item_type}"
    test_path = f"data/feature_folds/item_tag_target_only/{item_type}"

    print_results(path, test_path)