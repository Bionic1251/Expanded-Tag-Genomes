import pandas as pd
import sys
import os
import features.count_words as count_words

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

def process_reviews(item_type, data_set, chunk_size):
    print(f"Processing {item_type} ({data_set})...")
    if item_type == "movies":
        tg_path = config.MOVIE_TAG_GENOME_PATH
    else:
        tg_path = config.BOOK_TAG_GENOME_PATH

    print("Loading tags...")
    evaluation_data = pd.read_csv(f"{tg_path}/processed/features_r.csv")
    tags = evaluation_data["tag"].unique()

    # Process reviews with a counter
    reviews_input = f"data/raw_selected/{data_set}/{item_type}/reviews.json"
    reviews_output = f"data/preprocessed/{data_set}/core/{item_type}/"
    print("Processing reviews (counter)...")
    count_words.count_words(tags, reviews_input, reviews_output, chunk_size)

for item_type in config.ITEM_TYPES:
    for data_set in config.DATA_SETS:
        process_reviews(item_type, data_set, config.CHUNK_SIZE)
        print()
