import pandas as pd
import numpy as np
import util
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

# TFIDF per review
def prepare_review_df(tag_count, word_count, item_id_field_name):
    return pd.merge(tag_count, word_count, on=[item_id_field_name, "review_id"], how="left")

def get_tf(counts):
    counts["tf"] = counts.tag_count / (counts.word_count + 1)
    return counts

def get_idf(counts):
    # calculating how many times a tag appeared in reviews
    review_count = counts.tag.value_counts().reset_index()
    review_count.columns = ["tag", "review_count"]

    review_count["idf"] = np.log(counts.review_id.nunique() / review_count["review_count"] + 1)
    return review_count

def get_log_tfidf(tf, idf):
    tfidf = pd.merge(tf, idf, on="tag", how="left")
    tfidf.fillna(0, inplace=True)
    tfidf["tfidf"] = tfidf.tf * tfidf.idf
    tfidf["tfidf"] = np.log(tfidf["tfidf"] + 1)
    return tfidf

def get_tfidf(count_type, raw, word, training_raw, training_word):
    training_raw_counts = prepare_review_df(training_raw, training_word, "item_id")
    items_raw_counts = prepare_review_df(raw, word, "parent_asin")
    # calculating tf
    items_raw_tf = get_tf(items_raw_counts)
    # calculating idf
    items_raw_idf = get_idf(training_raw_counts)
    #items_raw_idf = get_idf(items_raw_counts)
    # calculating tfidf
    items_raw_tfidf = get_log_tfidf(items_raw_tf, items_raw_idf)
    # grouping by item-tag
    items_raw_tfidf_max = items_raw_tfidf.groupby(["parent_asin", "tag"]).tfidf.max().reset_index()
    items_raw_tfidf_max.columns = ["parent_asin", "tag", f"{count_type}_max_tfidf"]
    items_raw_tfidf_mean = items_raw_tfidf.groupby(["parent_asin", "tag"]).tfidf.mean().reset_index()
    items_raw_tfidf_mean.columns = ["parent_asin", "tag", f"{count_type}_mean_tfidf"]
    # results
    result = pd.merge(items_raw_tfidf_max, items_raw_tfidf_mean, on=["parent_asin", "tag"], how="inner")
    result.fillna(0, inplace=True)
    return result


# TFIDF per item
def get_idf_concat(counts, item_id_field_name):
    # calculating how many times a tag appeared in reviews
    tag_count = counts.tag.value_counts().reset_index()
    tag_count.columns = ["tag", "item_count"]

    tag_count["idf"] = np.log(counts[item_id_field_name].nunique() / tag_count["item_count"] + 1)
    return tag_count

def add_concat_tfidf(count_type, raw, word, training_raw, training_word):
    training_raw_counts = prepare_review_df(training_raw, training_word, "item_id")
    items_raw_counts = prepare_review_df(raw, word, "parent_asin")
    # tf
    raw_counts_grouped = items_raw_counts.groupby(["parent_asin", "tag"])[["tag_count", "word_count"]].sum().reset_index()
    raw_counts_grouped = get_tf(raw_counts_grouped)
    # idf
    idf = get_idf_concat(training_raw_counts, "item_id")
    # tfidf
    items_raw_tfidf = get_log_tfidf(raw_counts_grouped, idf)
    field_name = f"concat_{count_type}_tfidf"
    items_raw_tfidf = items_raw_tfidf.rename(columns={"tfidf": field_name})
    items_raw_tfidf.fillna(0, inplace=True)
    return items_raw_tfidf[["parent_asin", "tag", field_name]]


# main

training_raw = pd.read_csv(f"{config.REVIEW_COUNT_FOLDER}/raw_tag_count.csv")
training_lemma = pd.read_csv(f"{config.REVIEW_COUNT_FOLDER}/lemma_tag_count.csv")
training_word = pd.read_csv(f"{config.REVIEW_COUNT_FOLDER}/word_count.csv")

# Get folders to process
folders = util.get_folders(config.OUTPUT_PREPROCESSED)

for folder in folders:
    print(f"Processing {config.OUTPUT_PREPROCESSED}/{folder}")

    path = f"{config.OUTPUT_PREPROCESSED}/{folder}"

    raw = pd.read_csv(f"{path}/raw.csv")
    lemma = pd.read_csv(f"{path}/lemmatized.csv")
    word = pd.read_csv(f"{path}/word_count.csv")

    raw_tfidf = get_tfidf("raw", raw, word, training_raw, training_word)
    lemma_tfidf = get_tfidf("lemma", lemma, word, training_lemma, training_word)

    raw_concat_tfidf = add_concat_tfidf("raw", raw, word, training_raw, training_word)
    lemma_concat_tfidf = add_concat_tfidf("lemma", lemma, word, training_lemma, training_word)

    raw_max_tfidf_dict = util.convert_to_nested_dict(raw_tfidf, "raw_max_tfidf")
    raw_mean_tfidf_dict = util.convert_to_nested_dict(raw_tfidf, "raw_mean_tfidf")
    lemma_max_tfidf_dict = util.convert_to_nested_dict(lemma_tfidf, "lemma_max_tfidf")
    lemma_mean_tfidf_dict = util.convert_to_nested_dict(lemma_tfidf, "lemma_mean_tfidf")
    raw_concat_tfidf_dict = util.convert_to_nested_dict(raw_concat_tfidf, "concat_raw_tfidf")
    lemma_concat_tfidf_dict = util.convert_to_nested_dict(lemma_concat_tfidf, "concat_lemma_tfidf")

    pairs = [(raw_max_tfidf_dict, "raw_max_tfidf_dict"),
             (raw_mean_tfidf_dict, "raw_mean_tfidf_dict"),
             (lemma_max_tfidf_dict, "lemma_max_tfidf_dict"),
             (lemma_mean_tfidf_dict, "lemma_mean_tfidf_dict"),
             (raw_concat_tfidf_dict, "raw_concat_tfidf_dict"),
             (lemma_concat_tfidf_dict, "lemma_concat_tfidf_dict")]

    util.save_pickle_files(pairs, path)