import pandas as pd
import sys
import os
import features.bert_descriptions as bert_descriptions
import features.bert_reviews as bert_reviews

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

# BERT preprocessing only (CUDA is recommended)

def process_bert_embeddings(item_type, data_set, chunk_size):
    print(f"Processing {item_type} ({data_set})...")
    if item_type == "movies":
        tg_path = config.MOVIE_TAG_GENOME_PATH
    else:
        tg_path = config.BOOK_TAG_GENOME_PATH

    print("Loading tags...")
    evaluation_data = pd.read_csv(f"{tg_path}/processed/features_r.csv")
    tags = evaluation_data["tag"].unique()

    # Process reviews with BERT
    reviews_input = f"data/raw_selected/{data_set}/{item_type}/reviews.json"
    reviews_output = f"data/preprocessed/{data_set}/core/{item_type}/bert_reviews.csv"
    print("Processing reviews...")
    bert_reviews.compute_review_tag_similarities(reviews_input, reviews_output, tags, chunk_size)

    if item_type == "books" and data_set == "tg" or data_set == "amazon":
        # Process descriptions
        desc_input = f"data/raw_selected/{data_set}/{item_type}/descriptions.json"
        desc_output = f"data/preprocessed/{data_set}/core/{item_type}/bert_descriptions.csv"
        print("Processing descriptions...")
        bert_descriptions.compute_item_tag_similarities(tags, desc_input, desc_output, chunk_size)

    if item_type == "books" and data_set == "amazon":
        # Process "highlights"
        desc_input = f"data/raw_selected/{data_set}/{item_type}/highlights.json"
        desc_output = f"data/preprocessed/{data_set}/core/{item_type}/bert_highlights.csv"
        print(f"Processing highlights ({item_type})...")
        bert_descriptions.compute_item_tag_similarities(tags, desc_input, desc_output, chunk_size)

for item_type in config.ITEM_TYPES:
    for data_set in config.DATA_SETS:
        process_bert_embeddings(item_type, data_set, config.CHUNK_SIZE)
        print()
