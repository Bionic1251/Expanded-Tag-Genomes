import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

NUMERICAL_COLS = ["log_IMDB", "log_IMDB_nostem", "lsi_imdb_175", "avg_rating"]

def transform(movie_df):
    movie_df = movie_df.rename(columns={"movieId": "item_id",
                             "term_freq_IMDB_log" : "log_IMDB",
                             "term_freq_IMDB_log_nostem" : "log_IMDB_nostem",
                             "avg_movie_rating" : "avg_rating"})
    movie_df = movie_df[["item_id", "tag", "log_IMDB", "log_IMDB_nostem", "lsi_imdb_175", "avg_rating"]].copy()
    print("Feature record num", len(movie_df))
    movie_df = movie_df.drop_duplicates(subset=['item_id', "tag"])
    print("Feature record num after duplicate removal", len(movie_df))
    return movie_df

def save_scaled(in_path, out_path, features, numerical_cols):
    features.drop_duplicates(subset=["item_id", "tag"], inplace=True)
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

for item_type in config.ITEM_TYPES:
    print(item_type)
    data_set = "amazon"
    preprocessing_path = f"data/preprocessed/{data_set}/original/{item_type}"
    out_fold_path = f"data/feature_folds/{data_set}/original/{item_type}"
    in_fold_path = f"data/feature_folds/item_tag_target_only/{item_type}"

    tagdl_features = pd.read_csv(preprocessing_path + "/features.txt", sep="\t")
    tagdl_features = transform(tagdl_features.copy())
    save_scaled(in_fold_path, out_fold_path, tagdl_features, NUMERICAL_COLS)
    print("Done")
