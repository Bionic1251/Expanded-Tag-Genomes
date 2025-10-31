import os
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config


class TGDataFilter:
    def __init__(self, item_type: str = "movies", dataset_name: str = "tg"):
        self.item_type = item_type
        print(f"Processing {item_type}")
        self.dataset_name = dataset_name
        self.chunk_size = config.CHUNK_SIZE
        self.path = (
            config.MOVIE_TAG_GENOME_PATH
            if item_type == "movies"
            else config.BOOK_TAG_GENOME_PATH
        )
        self.output_dir = f"data/raw_selected/{self.dataset_name}/{self.item_type}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.item_ids = self._load_matched_item_ids()
        self.tag_mapping = self._load_tag_mapping()

    def _load_matched_item_ids(self):
        path = f"data/matches/{self.item_type}.json"
        df = pd.read_json(path, lines=True)
        return df.item_id.unique()

    def _load_tag_mapping(self):
        path = os.path.join(self.path, "raw/tags.json")
        return pd.read_json(path, lines=True)

    def _filter_and_save(
        self,
        in_file_path: str,
        out_file_path: str,
        file_format: str,
        transform_fn=None,
        columns=None
    ):
        print(f"Reading {in_file_path}... Saving {out_file_path}...")
        first_chunk = True
        for chunk in pd.read_json(in_file_path, chunksize=self.chunk_size, lines=True):
            filtered = chunk[chunk.item_id.isin(self.item_ids)]
            print(f"Processing chunk {len(filtered)}")
            if transform_fn:
                filtered = transform_fn(filtered)
            if columns:
                filtered = filtered[columns]

            if file_format == "csv":
                if not filtered.empty:
                    filtered.to_csv(out_file_path, index=False, mode="a", header=first_chunk)
                first_chunk = False
            elif file_format == "json":
                if not filtered.empty:
                    filtered.to_json(out_file_path, orient="records", lines=True, mode="a")
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    def process_tag_applications(self):
        print("Processing tag applications")
        def merge_tags(df):
            return pd.merge(df, self.tag_mapping, left_on="tag_id", right_on="id", how="left")

        self._filter_and_save(
            in_file_path=os.path.join(self.path, "raw/tag_count.json"),
            out_file_path=os.path.join(self.output_dir, "tag_applications.csv"),
            file_format="csv",
            transform_fn=merge_tags,
            columns=["item_id", "tag", "num"]
        )

    def process_ratings(self):
        print("Processing ratings")
        self._filter_and_save(
            in_file_path=os.path.join(self.path, "raw/ratings.json"),
            out_file_path=os.path.join(self.output_dir, "ratings.csv"),
            file_format="csv"
        )

    def process_reviews(self):
        print("Processing reviews")
        self._filter_and_save(
            in_file_path=os.path.join(self.path, "raw/reviews.json"),
            out_file_path=os.path.join(self.output_dir, "reviews.json"),
            file_format="json"
        )

    def process_metadata(self):
        if self.item_type == "movies":
            return
        print("Processing metadata")

        def prepare_descriptions(df):
            return df.rename(columns={"description": "txt"})[["item_id", "txt"]]

        self._filter_and_save(
            in_file_path=os.path.join(self.path, "raw/metadata.json"),
            out_file_path=os.path.join(self.output_dir, "descriptions.json"),
            file_format="json",
            transform_fn=prepare_descriptions
        )



for item_type in config.ITEM_TYPES:
    processor = TGDataFilter(item_type=item_type)
    processor.process_tag_applications()
    processor.process_ratings()
    processor.process_reviews()
    processor.process_metadata()
    print()