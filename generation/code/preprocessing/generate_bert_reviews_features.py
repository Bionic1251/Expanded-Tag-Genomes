import pandas as pd
import numpy as np
import util
import sys
import os
from torch import cuda
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

feature_path = config.TRAINING_DATA
CHUNK_SIZE = 10000

print("preparing")

BERT_MODEL = "msmarco-distilbert-base-v4"

# Read data
movie_tag_pair_df = pd.read_csv(feature_path)
tags = movie_tag_pair_df.tag.unique()

print("Encoding tags...")
device = "cuda" if cuda.is_available() else "cpu"
model = SentenceTransformer(BERT_MODEL)
print("model built")

tag_embeddings = model.encode(tags, show_progress_bar=True, device=device)
print("Tags encoded")
tag_df = pd.DataFrame({"tag": tags, "embedding": tag_embeddings.tolist()})
tag_df["parsed"] = tag_df["embedding"].map(np.array)
print("Tag df is built")


def process_review_chunk(texts, item_ids, tag_df, model, device, agg_data):
    """Encodes reviews, computes cosine similarity with tags, and updates aggregates."""
    review_embs = model.encode(texts, show_progress_bar=False, device=device)
    sim_matrix = cosine_similarity(np.stack(tag_df["parsed"]), np.array(review_embs))

    for tag_idx, tag in enumerate(tag_df["tag"]):
        for rev_idx, item_id in enumerate(item_ids):
            sim = sim_matrix[tag_idx, rev_idx]
            key = (item_id, tag)
            agg_data[key]["sum"] += sim
            agg_data[key]["count"] += 1
            agg_data[key]["max"] = max(agg_data[key]["max"], sim)

# Get folders to process
folders = util.get_folders(config.OUTPUT_RAW)
print(f"folders: {folders}")

for folder in folders:
    print(f"Processing {config.OUTPUT_PREPROCESSED}/{folder}")
    os.makedirs(f"{config.OUTPUT_PREPROCESSED}/{folder}", exist_ok=True)
    agg_data = defaultdict(lambda: {"sum": 0.0, "count": 0, "max": -1.0})

    with open(f"{config.OUTPUT_RAW}/{folder}/reviews.json", "r") as infile:
        chunk_texts = []
        chunk_item_ids = []
        total_count = 0
        chunk_count = 0

        for line in infile:
            obj = json.loads(line)

            chunk_texts.append(obj["text"])
            chunk_item_ids.append(obj["parent_asin"])
            total_count += 1

            if len(chunk_texts) >= CHUNK_SIZE:
                print(f"Processing chunk {chunk_count}, total reviews so far: {total_count}")
                process_review_chunk(chunk_texts, chunk_item_ids, tag_df, model, device, agg_data)
                chunk_texts, chunk_item_ids = [], []
                chunk_count += 1

        # Final chunk
        if chunk_texts:
            print(f"Processing final chunk {chunk_count}, total reviews: {total_count}")
            process_review_chunk(chunk_texts, chunk_item_ids, tag_df, model, device, agg_data)

    # ------------------------- STEP 4: SAVE RESULTS -------------------
    print("Saving aggregated similarity results...")

    summary_rows = []
    for (item_id, tag), data in agg_data.items():
        sim_mean = data["sum"] / data["count"]
        sim_max = data["max"]
        summary_rows.append({"parent_asin": item_id, "tag": tag, "bert_avg_sim": sim_mean, "bert_max_sim": sim_max})
    items_bert_sim = pd.DataFrame(summary_rows)

    bert_avg_sim_dict = util.convert_to_nested_dict(items_bert_sim, "bert_avg_sim")
    bert_max_sim_dict = util.convert_to_nested_dict(items_bert_sim, "bert_max_sim")

    pairs = [(bert_avg_sim_dict, "bert_avg_sim_dict"),
             (bert_max_sim_dict, "bert_max_sim_dict")]

    util.save_pickle_files(pairs, f"{config.OUTPUT_PREPROCESSED}/{folder}")