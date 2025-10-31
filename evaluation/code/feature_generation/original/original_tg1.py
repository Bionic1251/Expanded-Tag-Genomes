import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config
import original_util

NUMERICAL_COLS = ["rating_similarity", "log_IMDB", "log_IMDB_nostem", "lsi_tags_75", "lsi_imdb_175", "avg_rating"]

def save_scaled(in_path, out_path, features, numerical_cols):
    print("Saving scaled features to", out_path)
    for fold in range(10):
        train_df = pd.read_csv(f"{in_path}/train{fold}.csv")
        test_df = pd.read_csv(f"{in_path}/test{fold}.csv")

        train_df = pd.merge(train_df, features, on=["item_id", "tag"], how="left")
        test_df = pd.merge(test_df, features, on=["item_id", "tag"], how="left")

        scaler = StandardScaler()
        train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
        test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

        train_df.to_csv(f"{out_path}/train{fold}.csv", index=False)
        test_df.to_csv(f"{out_path}/test{fold}.csv", index=False)

print("Preparing original features to calculate tag probability")

for item_type in config.ITEM_TYPES:
    print(item_type)
    data_set = "tg"
    preprocessing_path = f"data/preprocessed/{data_set}/original/{item_type}"
    raw_path = f"data/raw_selected/{data_set}/original/{item_type}"
    fold_path = f"data/feature_folds/item_tag_target_only/{item_type}"

    tagdl_features = pd.read_csv(preprocessing_path + "/features.txt", sep="\t")
    tagdl_features = original_util.transform(tagdl_features.copy())
    tagdl_features.drop(columns=["tag_count"], inplace=True)

    save_scaled(fold_path, preprocessing_path + "/folds", tagdl_features, NUMERICAL_COLS)

print("The next step is to run the R script to generate the tag probability feature")