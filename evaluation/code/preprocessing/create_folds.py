import pandas as pd
from sklearn.model_selection import KFold
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

FOLDS = 10

for item_type in config.ITEM_TYPES:

    if item_type == "movies":
        TG_PATH = config.MOVIE_TAG_GENOME_PATH
        AMAZON_METADATA_PATH = config.AMAZON_MOVIE_METADATA_PATH
    else:
        TG_PATH = config.BOOK_TAG_GENOME_PATH
        AMAZON_METADATA_PATH = config.AMAZON_BOOK_METADATA_PATH

    paths_to_save = [f"data/feature_folds/item_tag_target_only/{item_type}"]

    def save_n_fold_split_by_item(folds, df, item_type, paths_to_save):
        # Get unique items
        unique_items = df["item_id"].unique()

        # Set up KFold
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        # Iterate over folds
        for fold, (train_idx, test_idx) in enumerate(kf.split(unique_items)):
            train_items = unique_items[train_idx]
            test_items = unique_items[test_idx]

            train = df[df["item_id"].isin(train_items)].reset_index(drop=True)
            test = df[df["item_id"].isin(test_items)].reset_index(drop=True)

            # Save to CSV
            for path in paths_to_save:
                train.to_csv(f"{path}/train{fold}.csv", index=False)
                test.to_csv(f"{path}/test{fold}.csv", index=False)

            print(f"Fold {fold}: Train items = {len(train_items)}, Test items = {len(test_items)}")
            print(f"Train rows = {len(train)}, Test rows = {len(test)}\n")
            print(f"Train tags = {train.tag.nunique()}, Test tags = {test.tag.nunique()}\n")


    evaluation_data = pd.read_csv(TG_PATH + "/processed/features_r.csv")

    # Limiting data to only matched items
    matched_data = pd.read_json(f"data/matches/{item_type}.json", lines=True)
    print(f"Matched items. TG items: {matched_data.item_id.nunique()}, "
          f"Amazon items: {matched_data.parent_asin.nunique()}, "
          f"Data frame length {len(matched_data)}")

    print(f"Items used in evaluated before limiting them to the matched ones: {evaluation_data.item_id.nunique()}")

    evaluation_data = evaluation_data[evaluation_data.item_id.isin(matched_data.item_id.unique())]

    # Grouping rows: multiple users can rate the same item-tag pair

    item_tag_grouped = evaluation_data.groupby(["tag", "item_id"]).targets.mean().reset_index()
    print(f"Items used in evaluated after limiting them to the matched ones: {item_tag_grouped.item_id.nunique()}")
    print("Overall rows:", len(item_tag_grouped))
    print("Data example:")
    print(item_tag_grouped.head())
    print()

    save_n_fold_split_by_item(FOLDS, item_tag_grouped, item_type, paths_to_save)