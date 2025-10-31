import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import cuda
from collections import defaultdict

def compute_review_tag_similarities(input_file_path, output_file_path, tags, chunk_size=1000):
    """
    Compute review-tag similarities using BERT embeddings.

    Parameters:
    - input_file_path (str): Path to the input review file (JSON lines).
    - output_file_path (str): Path to save the output CSV file.
    - tags (list of str): List of tags to compare reviews with.
    - chunk_size (int): Number of reviews per processing chunk.
    """

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

    # Step 1: Set up model and device
    bert_model_name = "msmarco-distilbert-base-v4"
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(bert_model_name)

    # Step 2: Encode tags
    print("Encoding tags...")
    tag_embeddings = model.encode(tags, show_progress_bar=True, device=device)
    tag_df = pd.DataFrame({"tag": tags, "embedding": tag_embeddings.tolist()})
    tag_df["parsed"] = tag_df["embedding"].map(np.array)

    # Step 3: Process reviews
    print("Processing reviews and aggregating similarities...")
    agg_data = defaultdict(lambda: {"sum": 0.0, "count": 0, "max": -1.0})

    with open(input_file_path, "r") as infile:
        chunk_texts = []
        chunk_item_ids = []
        total_count = 0
        chunk_count = 0

        for line in infile:
            obj = json.loads(line)
            chunk_texts.append(obj["txt"])
            chunk_item_ids.append(obj["item_id"])
            total_count += 1

            if len(chunk_texts) >= chunk_size:
                print(f"Processing chunk {chunk_count}, total reviews so far: {total_count}")
                process_review_chunk(chunk_texts, chunk_item_ids, tag_df, model, device, agg_data)
                chunk_texts, chunk_item_ids = [], []
                chunk_count += 1

        if chunk_texts:
            print(f"Processing final chunk {chunk_count}, total reviews: {total_count}")
            process_review_chunk(chunk_texts, chunk_item_ids, tag_df, model, device, agg_data)

    # Step 4: Save results
    print("Saving aggregated similarity results...")
    summary_rows = []
    for (item_id, tag), data in agg_data.items():
        sim_mean = data["sum"] / data["count"]
        sim_max = data["max"]
        summary_rows.append({"item_id": item_id, "tag": tag, "sim_mean": sim_mean, "sim_max": sim_max})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file_path, index=False)
    print(f"Saved summary to {output_file_path}")
