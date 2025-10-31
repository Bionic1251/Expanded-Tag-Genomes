import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config
import original_util

def save_scaled(in_fold_path, out_fold_path, tag_prob_path, features, numerical_cols):
    features.drop_duplicates(subset=["item_id", "tag"], inplace=True)
    print("Saving scaled features")
    for fold in range(10):
        train_df = pd.read_csv(f"{in_fold_path}/train{fold}.csv")
        test_df = pd.read_csv(f"{in_fold_path}/test{fold}.csv")

        tag_prob_train = pd.read_csv(f"{tag_prob_path}/train_tag_prob{fold}.csv")
        tag_prob_test = pd.read_csv(f"{tag_prob_path}/test_tag_prob{fold}.csv")

        train_df = pd.merge(train_df, tag_prob_train, on=["item_id", "tag"], how="left")
        test_df = pd.merge(test_df, tag_prob_test, on=["item_id", "tag"], how="left")

        train_df = pd.merge(train_df, features, on=["item_id", "tag"], how="left")
        test_df = pd.merge(test_df, features, on=["item_id", "tag"], how="left")

        scaler = StandardScaler()
        train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
        test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

        train_df.drop_duplicates(subset=["item_id", "tag"], inplace=True)
        test_df.drop_duplicates(subset=["item_id", "tag"], inplace=True)

        train_df.to_csv(f"{out_fold_path}/train{fold}.csv", index=False)
        test_df.to_csv(f"{out_fold_path}/test{fold}.csv", index=False)

for item_type in config.ITEM_TYPES:
    print(item_type)
    data_set = "tg"
    preprocessing_path = f"data/preprocessed/{data_set}/original/{item_type}"
    raw_path = f"data/raw_selected/{data_set}/original/{item_type}"
    in_fold_path = f"data/feature_folds/item_tag_target_only/{item_type}"
    out_fold_path = f"data/feature_folds/{data_set}/original/{item_type}"

    tagdl_features = pd.read_csv(preprocessing_path + "/features.txt", sep="\t")

    tagdl_features = original_util.transform(tagdl_features.copy())
    tagdl_features.drop(columns=["tag_count", "lsi_imdb_25"], inplace=True)
    NUMERICAL_COLS = ["rating_similarity", "log_IMDB", "log_IMDB_nostem", "lsi_tags_75", "lsi_imdb_175", "avg_rating",
                      "tag_prob", "tag_exists"]
    save_scaled(in_fold_path, out_fold_path, preprocessing_path + "/tag_prob", tagdl_features, NUMERICAL_COLS)

print("Done")