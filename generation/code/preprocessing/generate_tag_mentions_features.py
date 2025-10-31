import pandas as pd
import numpy as np
import util
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config


def get_log_mentions(tag_counts, word_counts, field_name, item_id_field_name):
    tag_counts = pd.merge(tag_counts, word_counts, on=["review_id", item_id_field_name])
    tag_counts["mention_count"] = tag_counts["tag_count"] / tag_counts["word_count"]
    tag_counts = tag_counts.groupby([item_id_field_name, "tag"]).mention_count.mean().reset_index()
    tag_counts["mention_count"] = np.log(tag_counts["mention_count"] + 1)
    return tag_counts.rename(columns={"mention_count": field_name})

# Get folders to process
folders = util.get_folders(config.OUTPUT_PREPROCESSED)

for folder in folders:
    print(f"Processing {config.OUTPUT_PREPROCESSED}/{folder}")

    path = f"{config.OUTPUT_PREPROCESSED}/{folder}"

    raw = pd.read_csv(f"{path}/raw.csv")
    lemma = pd.read_csv(f"{path}/lemmatized.csv")
    words = pd.read_csv(f"{path}/word_count.csv")

    items_raw_review_mentions = get_log_mentions(raw, words, "raw_review_mentions", "parent_asin")
    items_lemma_review_mentions = get_log_mentions(lemma, words, "lemma_review_mentions", "parent_asin")

    raw_review_mentions_dict = util.convert_to_nested_dict(items_raw_review_mentions, "raw_review_mentions")
    lemma_review_mentions_dict = util.convert_to_nested_dict(items_lemma_review_mentions, "lemma_review_mentions")

    pairs = [(raw_review_mentions_dict, "raw_review_mentions_dict"),
             (lemma_review_mentions_dict, "lemma_review_mentions_dict")]

    util.save_pickle_files(pairs, path)