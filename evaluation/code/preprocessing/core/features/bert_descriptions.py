import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import cuda

def compute_item_tag_similarities(tags, item_file_path, output_file_path, chunk_size):
    """
    Encode tags and item descriptions, then compute and save similarity scores.

    Parameters:
    - tags (list of str): List of tags to encode and compare.
    - item_file_path (str): Path to JSONL file with item descriptions. Each line is JSON with fields "item_id" and "txt".
    - output_file_path (str): Path to save the CSV similarity output.
    """
    bert_model_name = "msmarco-distilbert-base-v4"
    print("Encoding tags...")
    device = "cuda" if cuda.is_available() else "cpu"
    model = SentenceTransformer(bert_model_name)

    tag_embeddings = model.encode(tags, show_progress_bar=True, device=device)
    tag_df = pd.DataFrame({"tag": tags, "embedding": tag_embeddings.tolist()})
    tag_df["parsed"] = tag_df["embedding"].map(np.array)

    print("Loading item descriptions...")
    texts = []
    item_ids = []

    for chunk in pd.read_json(item_file_path, lines=True, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["txt"])
        chunk = chunk[chunk["txt"].str.strip() != ""]
        texts.extend(chunk["txt"].tolist())
        item_ids.extend(chunk["item_id"].tolist())

    print(f"Loaded {len(texts)} items.")

    print("Encoding texts...")
    item_embeddings = model.encode(texts, show_progress_bar=True, device=device)

    print("Computing similarities...")
    sim_matrix = cosine_similarity(np.stack(tag_df["parsed"]), np.array(item_embeddings))

    print("Formatting results...")
    rows = []
    for tag_idx, tag in enumerate(tag_df["tag"]):
        for item_idx, item_id in enumerate(item_ids):
            sim = sim_matrix[tag_idx, item_idx]
            rows.append({"item_id": item_id, "tag": tag, "similarity": sim})

    df = pd.DataFrame(rows)
    df.to_csv(output_file_path, index=False)
    print(f"Saved similarity results to {output_file_path}")

