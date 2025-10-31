import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config


class AmazonDataExtractor:
    def __init__(self, item_type="movies", dataset_name="amazon"):
        self.item_type = item_type
        print(f"Processing {item_type}")
        self.dataset_name = dataset_name
        self.chunk_size = config.CHUNK_SIZE

        self.review_path = (
            config.AMAZON_MOVIE_REVIEWS_PATH
            if item_type == "movies"
            else config.AMAZON_BOOK_REVIEWS_PATH
        )
        self.metadata_path = (
            config.AMAZON_MOVIE_METADATA_PATH
            if item_type == "movies"
            else config.AMAZON_BOOK_METADATA_PATH
        )
        self.tg_path = (
            config.MOVIE_TAG_GENOME_PATH
            if item_type == "movies"
            else config.BOOK_TAG_GENOME_PATH
        )

        self.output_dir = f"data/raw_selected/{self.dataset_name}/{self.item_type}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.matched_data = pd.read_json(f"data/matches/{self.item_type}.json", lines=True)
        self.parent_asins = self.matched_data.parent_asin.unique()

    def process_ratings(self):
        print("Processing ratings")
        out_file = os.path.join(self.output_dir, "ratings.csv")
        first_chunk = True
        count = 0
        for chunk in pd.read_json(self.review_path, chunksize=self.chunk_size, lines=True):
            filtered = chunk[chunk.parent_asin.isin(self.parent_asins)]
            filtered = pd.merge(filtered, self.matched_data[["parent_asin", "item_id"]], on="parent_asin", how="left")
            if not filtered.empty:
                filtered[["item_id", "user_id", "rating"]].to_csv(out_file, index=False, mode="a", header=first_chunk)
            count += len(filtered)
            first_chunk = False
        print(f"Processed {count} ratings")

    def process_reviews(self):
        print("Processing reviews")
        out_file = os.path.join(self.output_dir, "reviews.json")
        count = 0
        for chunk in pd.read_json(self.review_path, chunksize=self.chunk_size, lines=True):
            filtered = chunk[chunk.parent_asin.isin(self.parent_asins)]
            filtered = pd.merge(filtered, self.matched_data[["parent_asin", "item_id"]], on="parent_asin", how="left")
            filtered = filtered[["item_id", "user_id", "text"]].rename(columns={"text": "txt"})
            if not filtered.empty:
                filtered.to_json(out_file, orient="records", lines=True, mode="a")
            count += len(filtered)
        print(f"Processed {count} reviews")

    def process_descriptions(self):
        print("Processing descriptions")
        out_file = os.path.join(self.output_dir, "descriptions.json")
        count = 0
        for chunk in pd.read_json(self.metadata_path, chunksize=self.chunk_size, lines=True):
            filtered = chunk[chunk.parent_asin.isin(self.parent_asins)]
            filtered = pd.merge(filtered, self.matched_data[["parent_asin", "item_id"]], on="parent_asin", how="left")
            filtered = filtered[["item_id", "description"]]
            filtered["description"] = filtered["description"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
            filtered = filtered[filtered["description"].str.strip() != ""]
            filtered = filtered.rename(columns={"description": "txt"})
            if not filtered.empty:
                filtered.to_json(out_file, orient="records", lines=True, mode="a")
            count += len(filtered)
        print(f"Processed {count} reviews")

    def process_features(self):
        print("Processing the highlights field")
        out_file = os.path.join(self.output_dir, "highlights.json")
        count = 0
        for chunk in pd.read_json(self.metadata_path, chunksize=self.chunk_size, lines=True):
            filtered = chunk[chunk.parent_asin.isin(self.parent_asins)]
            filtered = pd.merge(filtered, self.matched_data[["parent_asin", "item_id"]], on="parent_asin", how="left")
            filtered = filtered[["item_id", "features"]]
            filtered["features"] = filtered["features"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
            filtered = filtered[filtered["features"].str.strip() != ""]
            filtered = filtered.rename(columns={"features": "txt"})
            if not filtered.empty:
                filtered.to_json(out_file, orient="records", lines=True, mode="a")
            count += len(filtered)
        print(f"Processed {count} feature values")


    def run(self):
        self.process_ratings()
        self.process_reviews()
        self.process_descriptions()
        if self.item_type == "books":
            self.process_features()

for item_type in config.ITEM_TYPES:
    extractor = AmazonDataExtractor(item_type=item_type)
    extractor.run()
    print()