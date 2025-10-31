import pandas as pd
import json
import re
from collections import Counter
import spacy
import os


def count_words(tags, review_path, output_folder, chunk_size):
    # Load spaCy with only tokenizer and lemmatizer
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print(f"Reading reviews from {review_path}, and saving results to {output_folder}")
    print(f"Chunk size: {chunk_size}")

    def lemmatize_tag(tag):
        doc = nlp(tag.lower())
        return ' '.join([t.lemma_ for t in doc if not t.is_punct and not t.is_space])

    lemmatized_tag_map = {tag: lemmatize_tag(tag) for tag in tags}

    # Precompile patterns
    pattern_raw = re.compile(r'\b(?:' + '|'.join(re.escape(tag) for tag in tags) + r')\b', flags=re.IGNORECASE)
    pattern_lemma = re.compile(r'\b(?:' + '|'.join(re.escape(lemma) for lemma in lemmatized_tag_map.values()) + r')\b', flags=re.IGNORECASE)

    # Buffers
    count = 0

    lemma_path = os.path.join(output_folder, "lemma_tag_count.csv")
    raw_path = os.path.join(output_folder, "raw_tag_count.csv")
    wc_path = os.path.join(output_folder, "word_count.csv")

    pd.DataFrame(columns=["item_id", "tag", "review_id", "tag_count"]).to_csv(lemma_path, index=False)
    pd.DataFrame(columns=["item_id", "tag", "review_id", "tag_count"]).to_csv(raw_path, index=False)
    pd.DataFrame(columns=["item_id", "review_id", "word_count"]).to_csv(wc_path, index=False)

    # Batch processing function
    def process_batch(batch):
        nonlocal count
        texts = [obj["txt"] for obj in batch]
        item_ids = [obj["item_id"] for obj in batch]
        docs = list(nlp.pipe(texts, batch_size=32))

        lemma_rows = []
        raw_rows = []
        word_count_rows = []

        for i, doc in enumerate(docs):
            count += 1
            item_id = item_ids[i]
            raw_text = texts[i]

            # Word count
            word_count = sum(1 for token in doc if token.is_alpha)

            # Create lemmatized review string exactly like original
            lemmatized_review = ' '.join(
                token.lemma_ for token in doc if not token.is_punct and not token.is_space
            )

            # Find raw tag matches (regex on raw text)
            raw_freqs = Counter(
                match.lower() for match in pattern_raw.findall(raw_text)
            )
            for tag in tags:
                freq = raw_freqs.get(tag.lower(), 0)
                if freq:
                    raw_rows.append([item_id, tag, count, freq])

            # Find lemma tag matches (regex on lemmatized string)
            lemma_freqs = Counter(
                match.lower() for match in pattern_lemma.findall(lemmatized_review)
            )
            for tag, lemma in lemmatized_tag_map.items():
                freq = lemma_freqs.get(lemma.lower(), 0)
                if freq:
                    lemma_rows.append([item_id, tag, count, freq])

            # Save word count
            word_count_rows.append([item_id, count, word_count])

        pd.DataFrame(lemma_rows, columns=["item_id", "tag", "review_id", "tag_count"]).to_csv(lemma_path, mode="a",
                                                                                              header=False, index=False)
        pd.DataFrame(raw_rows, columns=["item_id", "tag", "review_id", "tag_count"]).to_csv(raw_path, mode="a",
                                                                                            header=False, index=False)
        pd.DataFrame(word_count_rows, columns=["item_id", "review_id", "word_count"]).to_csv(wc_path, mode="a",
                                                                                             header=False, index=False)
        print(f"Processed {count} reviews")

    # Read file in batches
    batch = []
    with open(review_path, "r") as infile:
        for line in infile:
            obj = json.loads(line)
            batch.append(obj)
            if len(batch) >= chunk_size:
                process_batch(batch)
                batch = []
        if batch:
            process_batch(batch)

    print("Finished processing.")
