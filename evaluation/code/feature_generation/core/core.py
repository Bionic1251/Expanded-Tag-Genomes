import pandas as pd
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

for item_type in config.ITEM_TYPES:
    for data_set in config.DATA_SETS:
        preprocessing_path = f"data/preprocessed/{data_set}/core/{item_type}"
        raw_path = f"data/raw_selected/{data_set}/{item_type}"
        fold_path = f"data/feature_folds/{data_set}/core/{item_type}"
        all_data_path = f"data/all_data/{data_set}/core/{item_type}.csv"
        item_tag_target_only_path = f"data/feature_folds/item_tag_target_only/{item_type}"

        print(f"Generating features for the {data_set} dataset ({item_type})")

        def add_item_tag_pairs(in_path, out_path):
            for fold in range(10):
                train_df = pd.read_csv(f"{in_path}/train{fold}.csv")
                test_df = pd.read_csv(f"{in_path}/test{fold}.csv")

                train_df.to_csv(f"{out_path}/train{fold}.csv", index=False)
                test_df.to_csv(f"{out_path}/test{fold}.csv", index=False)


        add_item_tag_pairs(item_tag_target_only_path, fold_path)

        # adding a feature
        def add_feature(feature_df, path, merge_columns=["item_id", "tag"]):
            feature_df.drop_duplicates(subset=merge_columns, inplace=True)
            for fold in range(10):
                train_df = pd.read_csv(f"{path}/train{fold}.csv")
                test_df = pd.read_csv(f"{path}/test{fold}.csv")

                train_df = pd.merge(train_df, feature_df, on=merge_columns, how="left")
                test_df = pd.merge(test_df, feature_df, on=merge_columns, how="left")

                train_df.fillna(0, inplace=True)
                test_df.fillna(0, inplace=True)

                train_df.to_csv(f"{path}/train{fold}.csv", index=False)
                test_df.to_csv(f"{path}/test{fold}.csv", index=False)

        # rating features
        print("Loading ratings")
        ratings = pd.read_csv(raw_path + "/ratings.csv")
        print(f"{len(ratings)} ratings loaded")

        # average rating
        print("Calculating average ratings")
        avg_ratings = ratings.groupby("item_id").rating.mean().reset_index()
        avg_ratings.columns = ["item_id", "avg_rating"]
        add_feature(avg_ratings, fold_path, ["item_id"])
        print(f"{len(avg_ratings)} items calculated")

        # popularity
        print("Calculating popularity")
        item_pop = ratings.groupby("item_id").rating.count().reset_index()
        item_pop.columns = ["item_id", "pop"]
        item_pop["pop"] = np.log(item_pop["pop"] + 1)
        add_feature(item_pop, fold_path, ["item_id"])
        print(f"{len(item_pop)} items calculated")

        # features based on word counts in reviews
        word_review_counts = pd.read_csv(preprocessing_path + "/word_count.csv")
        raw_review_counts = pd.read_csv(preprocessing_path + "/raw_tag_count.csv")
        lemma_review_counts = pd.read_csv(preprocessing_path + "/lemma_tag_count.csv")

        def get_log_mentions(tag_counts, word_counts, field_name):
            tag_counts = pd.merge(tag_counts, word_counts, on=["review_id", "item_id"])
            tag_counts["mention_count"] = tag_counts["tag_count"] / tag_counts["word_count"]
            tag_counts = tag_counts.groupby(["item_id", "tag"]).mention_count.mean().reset_index()
            tag_counts["mention_count"] = np.log(tag_counts["mention_count"] + 1)
            return tag_counts.rename(columns={"mention_count": field_name})

        print("Calculating review mentions")
        raw_review_mentions = get_log_mentions(raw_review_counts, word_review_counts, "raw_review_mentions")
        add_feature(raw_review_mentions, fold_path, ["item_id", "tag"])
        print(f"{len(raw_review_mentions)} raw review mentions calculated")

        lemma_review_mentions = get_log_mentions(lemma_review_counts, word_review_counts, "lemma_review_mentions")
        add_feature(lemma_review_mentions, fold_path, ["item_id", "tag"])
        print(f"{len(lemma_review_mentions)} lemma review mentions calculated")

        # TFIDF
        print("Calculating TFIDF")
        def prepare_review_df(tag_count, word_count):
            return pd.merge(tag_count, word_count, on=["item_id", "review_id"], how="left")

        def get_tf(counts):
            counts["tf"] = counts.tag_count / (counts.word_count + 1)
            return counts

        def get_idf(counts, ids):
            counts = counts[counts.item_id.isin(ids)]
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

        def add_tfidf(path, count_type, processed_words, word_count):
            for fold in range(10):
                train = pd.read_csv(f"{path}/train{fold}.csv")
                test = pd.read_csv(f"{path}/test{fold}.csv")
                movies_raw_counts = prepare_review_df(processed_words, word_count)
                # calculating tf
                movies_raw_tf = get_tf(movies_raw_counts)
                # calculating idf
                movies_raw_idf = get_idf(movies_raw_counts, train.item_id.unique())
                # calculating tfidf
                movies_raw_tfidf = get_log_tfidf(movies_raw_tf, movies_raw_idf)
                # grouping by item-tag
                movies_raw_tfidf_max = movies_raw_tfidf.groupby(["item_id", "tag"]).tfidf.max().reset_index()
                movies_raw_tfidf_max.columns = ["item_id", "tag", f"{count_type}_max_tfidf"]
                movies_raw_tfidf_mean = movies_raw_tfidf.groupby(["item_id", "tag"]).tfidf.mean().reset_index()
                movies_raw_tfidf_mean.columns = ["item_id", "tag", f"{count_type}_mean_tfidf"]
                # merging with training
                train = pd.merge(train, movies_raw_tfidf_max[["item_id", "tag", f"{count_type}_max_tfidf"]], on=["item_id", "tag"], how="left")
                train = pd.merge(train, movies_raw_tfidf_mean[["item_id", "tag", f"{count_type}_mean_tfidf"]], on=["item_id", "tag"], how="left")
                train.fillna(0, inplace=True)
                # merging with test
                test = pd.merge(test, movies_raw_tfidf_max[["item_id", "tag", f"{count_type}_max_tfidf"]], on=["item_id", "tag"], how="left")
                test = pd.merge(test, movies_raw_tfidf_mean[["item_id", "tag", f"{count_type}_mean_tfidf"]], on=["item_id", "tag"], how="left")
                test.fillna(0, inplace=True)
                # saving results
                train.to_csv(f"{path}/train{fold}.csv", index=False)
                test.to_csv(f"{path}/test{fold}.csv", index=False)

        add_tfidf(fold_path, "raw", raw_review_counts, word_review_counts)
        print(f"{len(raw_review_counts)} raw_review_counts")
        add_tfidf(fold_path, "lemma", lemma_review_counts, word_review_counts)
        print(f"{len(lemma_review_counts)} lemma_review_counts")

        # bert features
        print(f"BERT reviews")
        bert_reviews = pd.read_csv(preprocessing_path + "/bert_reviews.csv")
        bert_reviews.rename(columns={"sim_mean": "bert_avg_sim", "sim_max": "bert_max_sim"}, inplace=True)
        add_feature(bert_reviews, fold_path, ["item_id", "tag"])

        if item_type == "books" and data_set == "tg" or data_set == "amazon":
            print(f"BERT descriptions")
            bert_descriptions = pd.read_csv(preprocessing_path + "/bert_descriptions.csv")
            bert_descriptions.rename(columns={"similarity": "bert_description"}, inplace=True)
            add_feature(bert_descriptions, fold_path, ["item_id", "tag"])

        #features specific to the domain
        if item_type == "books" and data_set == "amazon":
            print(f"bert_highlights for books")
            bert_features = pd.read_csv(preprocessing_path + "/bert_highlights.csv")
            bert_features.rename(columns={'similarity': 'bert_highlights'}, inplace=True)
            add_feature(bert_features, fold_path, ["item_id", "tag"])

        def add_unscaled_tfidf(df, count_type, processed_words, word_count):
            movies_raw_counts = prepare_review_df(processed_words, word_count)
            # calculating tf
            movies_raw_tf = get_tf(movies_raw_counts)
            # calculating idf
            movies_raw_idf = get_idf(movies_raw_counts, df.item_id.unique())
            # calculating tfidf
            movies_raw_tfidf = get_log_tfidf(movies_raw_tf, movies_raw_idf)
            # grouping by item-tag
            movies_raw_tfidf_max = movies_raw_tfidf.groupby(["item_id", "tag"]).tfidf.max().reset_index()
            movies_raw_tfidf_max.columns = ["item_id", "tag", f"{count_type}_max_tfidf"]
            movies_raw_tfidf_mean = movies_raw_tfidf.groupby(["item_id", "tag"]).tfidf.mean().reset_index()
            movies_raw_tfidf_mean.columns = ["item_id", "tag", f"{count_type}_mean_tfidf"]
            # merging with training
            df = pd.merge(df, movies_raw_tfidf_max[["item_id", "tag", f"{count_type}_max_tfidf"]], on=["item_id", "tag"], how="left")
            df = pd.merge(df, movies_raw_tfidf_mean[["item_id", "tag", f"{count_type}_mean_tfidf"]], on=["item_id", "tag"], how="left")
            df.fillna(0, inplace=True)
            return df

        def save_unscaled(in_path, out_path, raw_review_counts, lemma_review_counts, word_review_counts):
            fold = 0
            train_df = pd.read_csv(f"{in_path}/train{fold}.csv")
            test_df = pd.read_csv(f"{in_path}/test{fold}.csv")
            df = pd.concat([train_df, test_df])
            df = df.drop(["raw_max_tfidf", "raw_mean_tfidf", "lemma_max_tfidf", "lemma_mean_tfidf"], axis=1)

            df = add_unscaled_tfidf(df, "raw", raw_review_counts, word_review_counts)
            df = add_unscaled_tfidf(df, "lemma", lemma_review_counts, word_review_counts)

            numerical_cols = df.drop(columns=["item_id", "tag", "targets"]).columns
            df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], 0)
            df[numerical_cols] = df[numerical_cols].fillna(0)
            df.to_csv(out_path, index=False)

        # Scaling
        def save_complete(path):
            for fold in range(10):
                train_df = pd.read_csv(f"{path}/train{fold}.csv")
                test_df = pd.read_csv(f"{path}/test{fold}.csv")
                numerical_cols = train_df.drop(columns=["item_id", "tag", "targets"]).columns


                train_df[numerical_cols] = train_df[numerical_cols].replace([np.inf, -np.inf], 0)
                train_df[numerical_cols] = train_df[numerical_cols].fillna(0)
                test_df[numerical_cols] = test_df[numerical_cols].replace([np.inf, -np.inf], 0)
                test_df[numerical_cols] = test_df[numerical_cols].fillna(0)

                scaler = StandardScaler()
                train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
                test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

                train_df.drop_duplicates(subset=["item_id", "tag"], inplace=True)
                test_df.drop_duplicates(subset=["item_id", "tag"], inplace=True)

                train_df.to_csv(f"{path}/train{fold}.csv", index=False)
                test_df.to_csv(f"{path}/test{fold}.csv", index=False)


        print("Save unscaled for score generation to", all_data_path)
        save_unscaled(fold_path, all_data_path, raw_review_counts, lemma_review_counts, word_review_counts)
        print("Scaling")
        save_complete(fold_path)
        print("Saved to", fold_path)
        print()