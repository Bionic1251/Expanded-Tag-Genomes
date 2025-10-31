import pandas as pd
import sys
import os
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config


def get_item_pop(path, chunk_size):
    print(f"Reading {path}, chunk size: {chunk_size}")
    item_counts = Counter()

    for chunk in pd.read_json(path, chunksize=chunk_size, lines=True):
        # Count occurrences of parent_asin in the current chunk
        counts = chunk['parent_asin'].value_counts()
        # Update the total counts
        item_counts.update(counts.to_dict())

    # If you want to convert the result to a DataFrame:
    result_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['count'])
    result_df.index.name = 'parent_asin'
    result_df = result_df.reset_index()
    result_df = result_df.sort_values(by='count', ascending=False)
    print(f"Aggregated {len(result_df)} items")
    return result_df


def save_filtered_json(input_path, filter_asins, output_file, columns, chunk_size):
    """Read JSONL file in chunks and save filtered rows with selected columns to output_file."""
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for chunk in pd.read_json(input_path, chunksize=chunk_size, lines=True):
            matched = chunk[chunk['parent_asin'].isin(filter_asins)].copy()
            if not matched.empty:
                # Process list-valued columns (e.g., description and features)
                if "features" in matched.columns:
                    matched = matched.rename(columns={"features": "highlights"}).copy()

                for col in ["description", "highlights"]:
                    if col in columns and col in matched.columns:
                        matched[col] = matched[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

                matched[columns].to_json(f_out, orient='records', lines=True, force_ascii=False)

def split_item_pop(item_pop, max_item, max_chunks):
    # Sort by number of reviews, descending
    item_pop = item_pop.sort_values("count", ascending=True).reset_index(drop=True)

    total_reviews = item_pop["count"].sum()
    target_reviews_per_chunk = total_reviews / max_chunks

    chunks = []
    current_chunk = []
    current_review_sum = 0

    for _, row in item_pop.iterrows():
        item = row["parent_asin"]
        count = row["count"]

        # Start new chunk if needed
        if (len(current_chunk) >= max_item or
            (current_review_sum + count > target_reviews_per_chunk and len(chunks) + 1 < max_chunks)):
            chunks.append(current_chunk)
            current_chunk = []
            current_review_sum = 0

        current_chunk.append(item)
        current_review_sum += count

    # Add last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def save_raw_data(in_review_path, in_metadata_path, out_path, chunk_size, highlights, item_pop):
    print(f"Saving raw data to {out_path}")
    chunks = split_item_pop(item_pop, config.MAX_CHUNK_SIZE, config.MAX_CHUNKS)
    i = 0
    for chunk_ids in chunks:
        print(f"Writing chunk {i}/{len(item_pop)}")
        chunk_df = item_pop[item_pop.parent_asin.isin(chunk_ids)].copy()
        folder_name = Path(out_path) / f"{i}-{i + len(chunk_df) - 1}"
        folder_name.mkdir(parents=True, exist_ok=True)

        # Save the items.csv for this chunk
        chunk_df.to_csv(folder_name / 'items.csv', index=False)

        asins = set(chunk_df['parent_asin'])

        # Save filtered reviews
        save_filtered_json(
            in_review_path,
            asins,
            folder_name / 'reviews.json',
            columns=['parent_asin', 'user_id', 'rating', 'text'],
            chunk_size=chunk_size
        )

        # Save filtered metadata
        metadata_columns = ['parent_asin', 'description', 'highlights'] if highlights else ['parent_asin', 'description']
        save_filtered_json(
            in_metadata_path,
            asins,
            folder_name / 'metadata.json',
            columns=metadata_columns,
            chunk_size=chunk_size
        )
        i += len(chunk_df)


item_pop = get_item_pop(config.REVIEW_PATH, config.CHUNK_SIZE)

save_raw_data(config.REVIEW_PATH, config.METADATA_PATH, config.OUTPUT_RAW, config.CHUNK_SIZE,
              config.INCLUDE_BERT_PRODUCT_HIGHLIGHTS, item_pop)