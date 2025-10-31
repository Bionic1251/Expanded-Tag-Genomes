import pandas as pd
import numpy as np
import util
import sys
import os
from torch import cuda
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

if config.INCLUDE_BERT_PRODUCT_HIGHLIGHTS:
    field_names = ["description", "highlights"]
else:
    field_names = ["description"]

feature_path = config.TRAINING_DATA

id_field = "parent_asin"

BERT_MODEL = "msmarco-distilbert-base-v4"

print("preparing")
# Read data
movie_tag_pair_df = pd.read_csv(feature_path)
tags = movie_tag_pair_df.tag.unique()

print("Encoding tags...")
device = "cuda" if cuda.is_available() else "cpu"
model = SentenceTransformer(BERT_MODEL)

tag_embeddings = model.encode(tags, show_progress_bar=True, device=device)
tag_df = pd.DataFrame({"tag": tags, "embedding": tag_embeddings.tolist()})
tag_df["parsed"] = tag_df["embedding"].map(np.array)


# Get folders to process
folders = util.get_folders(config.OUTPUT_RAW)

for field_name in field_names:
    for folder in folders:
        print(f"Processing {config.OUTPUT_RAW}/{folder}")
        os.makedirs(f"{config.OUTPUT_PREPROCESSED}/{folder}", exist_ok=True)

        print(f"Loading item {field_name}...")
        descriptions = []
        item_ids = []

        with open(f"{config.OUTPUT_RAW}/{folder}/metadata.json", "r") as file:
            for line in file:
                obj = json.loads(line)
                desc = obj.get(field_name)
                if not isinstance(desc, str) or not desc.strip():
                    continue
                descriptions.append(obj[field_name])
                item_ids.append(obj[id_field])

        print(f"Loaded {len(descriptions)} fields {field_name}.")
        print(f"Encoding {field_name}...")
        item_embeddings = model.encode(descriptions, show_progress_bar=True, device=device)
        print("Computing similarities...")
        sim_matrix = cosine_similarity(np.stack(tag_df["parsed"]), np.array(item_embeddings))
        print("Formatting results...")
        rows = []
        for tag_idx, tag in enumerate(tag_df["tag"]):
            for book_idx, item_id in enumerate(item_ids):
                sim = sim_matrix[tag_idx, book_idx]
                rows.append({id_field: item_id, "tag": tag, "similarity": sim})

        df = pd.DataFrame(rows)
        df.rename(columns={"similarity": f"bert_{field_name}"}, inplace=True)
        bert_description_dict = util.convert_to_nested_dict(df, f"bert_{field_name}")

        pairs = [(bert_description_dict, f"bert_{field_name}_dict")]
        util.save_pickle_files(pairs, f"{config.OUTPUT_PREPROCESSED}/{folder}")