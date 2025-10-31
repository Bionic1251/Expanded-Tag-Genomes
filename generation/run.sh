#!/bin/bash

# The script processes large Amazon review and metadata files by counting item popularity and
# splitting the data into balanced chunks, saving reviews, metadata and item statistics for
# each chunk into separate folders.
python code/raw.py

# The script processes review files folder by folder, lemmatizes text with spaCy, counts raw
# and lemmatized tag occurrences and word counts per review and saves the results into CSV
# files (raw tags, lemmatized tags and word counts).
python code/preprocessing/generate_count_raw_lemma.py

# The script computes TF-IDF features for raw and lemmatized tags in reviews, both per
# review and aggregated per item, then saves the results as pickled nested dictionaries for
# later use.
python code/preprocessing/generate_tfidf_features.py

# The script calculates average normalized tag mention frequencies (log-scaled) for raw and
# lemmatized tags per item, converts the results into nested dictionaries, and saves them
# as pickled files.
python code/preprocessing/generate_tag_mentions_features.py

# The script encodes tags and reviews with a BERT model, computes cosine similarities between
# them, aggregates mean and max similarities per item–tag pair and saves the results as
# pickled nested dictionaries.
python code/preprocessing/generate_bert_reviews_features.py

# The script encodes item descriptions (and optionally highlights) with a BERT model,
# computes cosine similarities with tag embeddings and saves the per-item–tag similarity
# scores as pickled nested dictionaries.
python code/preprocessing/generate_bert_metadata_features.py

# The script reads review ratings, computes per-item average ratings and log-scaled popularity counts,
# converts them into dictionaries and saves them as pickled files.
python code/preprocessing/generate_avg_pop_features.py

# The script trains (or loads) a neural network to predict tag relevance scores for items using
# precomputed features, then scales features, builds prediction dataframes in chunks and
# saves the resulting scores to CSV files.
python code/generate_scores.py