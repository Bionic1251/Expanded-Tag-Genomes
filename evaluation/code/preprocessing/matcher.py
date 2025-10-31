import pandas as pd
import tqdm
from abc import ABC, abstractmethod

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config


class BaseMatcher(ABC):
    def __init__(self):
        self.TG_PATH = self.get_tg_path()
        self.AMAZON_METADATA_PATH = self.get_amazon_path()

    @abstractmethod
    def get_tg_path(self):
        pass

    @abstractmethod
    def get_amazon_path(self):
        pass

    @abstractmethod
    def add_tg_context(self, record):
        pass

    @abstractmethod
    def add_amazon_context(self, record):
        pass

    @abstractmethod
    def prepare_tg_df(self, df):
        pass

    @abstractmethod
    def prepare_a_df(self, df):
        pass

    def clean_title(self, title):
        return (title.str.lower().fillna("").str
                .replace(r'\s*[\(\[\<][^()\[\]<>]*[\)\]\>]', '', regex=True).str.strip())


    def load_tg_metadata(self):
        evaluation_data = pd.read_csv(f"{self.TG_PATH}/processed/features_r.csv")
        evaluated_items = evaluation_data.item_id.unique()
        print(f"Trying to match {len(evaluated_items)} tg items")

        tg_metadata = pd.read_json(f"{self.TG_PATH}/raw/metadata.json", lines=True)
        tg_metadata = tg_metadata[tg_metadata.item_id.isin(evaluated_items)]
        tg_metadata["title_clean"] = self.clean_title(tg_metadata["title"])
        tg_metadata = self.prepare_tg_df(tg_metadata)
        tg_metadata["context"] = tg_metadata.apply(self.add_tg_context, axis=1)
        return tg_metadata

    def load_amazon_metadata(self, titles):
        array = []
        for chunk in tqdm.tqdm(pd.read_json(self.AMAZON_METADATA_PATH, lines=True, chunksize=config.CHUNK_SIZE), desc="Reading chunks"):
            chunk["title_clean"] = self.clean_title(chunk["title"])
            array.append(chunk[chunk["title_clean"].isin(titles)])
        a_metadata = pd.concat(array)
        a_metadata = self.clean_amazon_metadata(a_metadata)
        a_metadata = self.prepare_a_df(a_metadata)
        a_metadata["context"] = a_metadata.apply(self.add_amazon_context, axis=1)
        return a_metadata

    def clean_amazon_metadata(self, df):
        return df

    def match(self, tg_metadata, a_metadata):
        def jaccard_similarity(set1, set2):
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        merged = pd.merge(tg_metadata, a_metadata, on='title_clean', suffixes=('_tg', '_a'))
        merged = merged[merged.apply(lambda row: bool(row['context_tg'] & row['context_a']), axis=1)]
        grouped = merged.groupby('item_id')

        used_a_ids = set()
        matched_rows = []

        for item_id, group in tqdm.tqdm(grouped, desc="Matching"):
            group = group[~group['parent_asin'].isin(used_a_ids)]
            if group.empty:
                continue
            if len(group) == 1:
                best_row = group.iloc[0]
            else:
                group = group.copy()
                group['jaccard'] = group.apply(
                    lambda row: jaccard_similarity(row['context_tg'], row['context_a']), axis=1
                )
                best_row = group.sort_values(by='jaccard', ascending=False).iloc[0]
            used_a_ids.add(best_row['parent_asin'])
            matched_rows.append({
                'item_id': best_row['item_id'],
                'title_clean': best_row['title_clean'],
                'title_tg': best_row['title_tg'],
                'context_tg': best_row['context_tg'],
                'parent_asin': best_row['parent_asin'],
                'title_a': best_row['title_a'],
                'context_a': best_row['context_a'],
                'context_jaccard': jaccard_similarity(best_row['context_tg'], best_row['context_a']),
            })

        return pd.DataFrame(matched_rows)


class MovieMatcher(BaseMatcher):
    def get_tg_path(self):
        return config.MOVIE_TAG_GENOME_PATH

    def get_amazon_path(self):
        return config.AMAZON_MOVIE_METADATA_PATH

    def prepare_tg_df(self, df):
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')
        return df

    def prepare_a_df(self, df):
        df = df[df.main_category.isin(["Movies & TV", "Prime Video"])].copy()
        df['year'] = df['title'].str.extract(r'\(( *\d{4} *)\)')
        return df

    def add_tg_context(self, record):
        context = set()
        self.add_names_from_string_field(record["directedBy"], context)
        self.add_names_from_string_field(record["starring"], context)
        self.add_names_from_string_field(record["year"], context)
        return context

    def add_amazon_context(self, record):
        people = set()
        self.add_names_from_dict_field(people, record["details"], 'Producers')
        self.add_names_from_dict_field(people, record["details"], 'Directors')
        self.add_names_from_dict_field(people, record["details"], 'Starring')
        self.add_names_from_dict_field(people, record["details"], 'Actors')
        self.add_names_from_dict_field(people, record["details"], 'Director')
        self.add_names_from_dict_field(people, record["details"], 'Contributor')
        self.add_names_from_dict_field(people, record["details"], 'Year')
        self.add_names_from_string_field(record["year"], people)
        return people

    def add_names_from_string_field(self, field, context):
        if field and isinstance(field, str):
            for val in field.split(','):
                if val:
                    context.add(val.lower().strip())

    def add_names_from_dict_field(self, people, field, field_name):
        if field and field_name in field:
            value = field[field_name]
            if isinstance(value, (list, set)):
                for name in value:
                    people.add(name.lower().strip())
            elif isinstance(value, str):
                for name in value.split(","):
                    people.add(name.lower().strip())


class BookMatcher(BaseMatcher):
    def get_tg_path(self):
        return config.BOOK_TAG_GENOME_PATH

    def get_amazon_path(self):
        return config.AMAZON_BOOK_METADATA_PATH

    def prepare_tg_df(self, df):
        df["subtitle"] = df["title"].str.extract(r"\(([^)]+)\)")
        df["subtitle_clean"] = (df["subtitle"].str.lower().fillna("").str.
                                         replace(r'\s*[\(\[\<][^()\[\]<>]*[\)\]\>]', '', regex=True).str.strip())
        return df

    def prepare_a_df(self, df):
        df["second_title"] = df["title"].str.extract(r"\(([^)]+)\)")
        return df

    def add_tg_context(self, record):
        context = set()
        self.add_names_from_string_field(record["authors"], context)
        self.add_name(record["subtitle_clean"], context)
        self.add_names_from_string_field(record["year"], context)
        return context

    def clean_amazon_metadata(self, df):
        df["second_title"] = df["title"].str.extract(r"\(([^)]+)\)")
        df["second_title_clean"] = self.clean_title(df["second_title"])

        def extract_dict_field(series, field_name):
            return series.apply(lambda x: x.get(field_name) if isinstance(x, dict) else None)

        df["rel_date"] = extract_dict_field(df["details"], "Release date")
        df["pub_date"] = extract_dict_field(df["details"], "Publication date")
        df["rel_date"] = pd.to_datetime(df["rel_date"], errors="coerce")
        df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
        df["rel_year"] = df["rel_date"].dt.year
        df["pub_year"] = df["pub_date"].dt.year
        return df

    def add_amazon_context(self, record):
        context = set()
        if isinstance(record["author"], dict) and "name" in record["author"]:
            context.add(record["author"]["name"].lower().strip())
        self.add_name(record["second_title_clean"], context)
        self.add_year_if_exists(record["rel_year"], context)
        self.add_year_if_exists(record["pub_year"], context)
        return context

    def add_names_from_string_field(self, field, context):
        if field and isinstance(field, str):
            for val in field.split(','):
                if val:
                    context.add(val.lower().strip())

    def add_name(self, field, context):
        if field and isinstance(field, str):
            context.add(field.lower().strip())

    def add_year_if_exists(self, field, context):
        if pd.isna(field):
            return
        context.add(str(int(field)))


# ---- MAIN EXECUTION ----
for item_type in config.ITEM_TYPES:
    matcher_class = MovieMatcher if item_type == "movies" else BookMatcher
    matcher = matcher_class()

    print(f"Matching {item_type}")
    tg_metadata = matcher.load_tg_metadata()
    a_metadata = matcher.load_amazon_metadata(tg_metadata["title_clean"].unique())
    matched_df = matcher.match(tg_metadata, a_metadata)
    print(f"Matched {len(matched_df)} items")

    matched_df.to_json(f"data/matches/{item_type}.json", lines=True, orient="records")
    print("Saved to ", f"data/matches/{item_type}.json")
    print()
