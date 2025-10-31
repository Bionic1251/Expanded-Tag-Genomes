import pandas as pd
import json
import re
from collections import Counter
import spacy
import util
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

file_name = "reviews.json"
feature_path = config.TRAINING_DATA
raw_out_file_name = "raw.csv"
lemma_out_file_name = "lemmatized.csv"
word_count_file_name = "word_count.csv"

item_id_file_name = "parent_asin"
batch_size = 100000

print("preparing")
# Read data
movie_tag_pair_df = pd.read_csv(feature_path)
tags = movie_tag_pair_df.tag.unique()

# Lemmatization function
def lemmatize_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

# Lemmatize tags
lemmatized_tag_map = {tag: lemmatize_text(tag) for tag in tags}

# Compile regex patterns
pattern_lemma = re.compile(r'\b(?:' + '|'.join(re.escape(lemma) for lemma in lemmatized_tag_map.values()) + r')\b', flags=re.IGNORECASE)
pattern_raw = re.compile(r'\b(?:' + '|'.join(re.escape(tag) for tag in tags) + r')\b', flags=re.IGNORECASE)

def save_batch(data, columns, path, is_first):
    if is_first:
        pd.DataFrame(data, columns=columns).to_csv(path, index=False)
    else:
        pd.DataFrame(data, columns=columns).to_csv(path, index=False, mode='a', header=False)


def process_batch(batch, is_first):
    global count
    lemma_rows, raw_rows, word_count_rows = [], [], []

    docs = list(nlp.pipe((obj["text"] for obj in batch)))

    for obj, doc in zip(batch, docs):
        count += 1
        text = obj["text"]
        word_count = sum(1 for token in doc if token.is_alpha)

        lemmatized_review = ' '.join(token.lemma_ for token in doc if not token.is_punct and not token.is_space)

        # Lemmatized tag counts
        lemma_frequencies = Counter(
            match.lower() for match in pattern_lemma.findall(lemmatized_review)
        )
        for tag, lemmatized_tag in lemmatized_tag_map.items():
            freq = lemma_frequencies.get(lemmatized_tag.lower(), 0)
            if freq:
                lemma_rows.append([obj[item_id_file_name], tag, count, freq])

        # Raw tag counts
        raw_frequencies = Counter(
            match.lower() for match in pattern_raw.findall(text)
        )
        for tag in tags:
            freq = raw_frequencies.get(tag.lower(), 0)
            if freq:
                raw_rows.append([obj[item_id_file_name], tag, count, freq])

        word_count_rows.append([obj[item_id_file_name], count, word_count])

    save_batch(lemma_rows, [item_id_file_name, "tag", "review_id", "tag_count"],
               f"{config.OUTPUT_PREPROCESSED}/{folder}/{lemma_out_file_name}", is_first)
    save_batch(raw_rows, [item_id_file_name, "tag", "review_id", "tag_count"],
               f"{config.OUTPUT_PREPROCESSED}/{folder}/{raw_out_file_name}", is_first)
    save_batch(word_count_rows, [item_id_file_name, "review_id", "word_count"],
               f"{config.OUTPUT_PREPROCESSED}/{folder}/{word_count_file_name}", is_first)

    print(f"Processed {count} reviews")

# Get folders to process
folders = util.get_folders(config.OUTPUT_RAW)

for folder in folders:
    print(f"Processing {config.OUTPUT_RAW}/{folder}")
    os.makedirs(f"{config.OUTPUT_PREPROCESSED}/{folder}", exist_ok=True)

    # Initialize counters and buffers
    count = 0
    is_first = True

    file = f"{config.OUTPUT_RAW}/{folder}/{file_name}"

    # Read and process file in batches
    batch = []
    with open(file, "r") as infile:
        for line in infile:
            obj = json.loads(line)
            batch.append(obj)

            if len(batch) >= batch_size:
                process_batch(batch, is_first)
                is_first = False
                batch = []
        # Process any remaining items
        if batch:
            process_batch(batch, is_first)

print("Finished processing.")
