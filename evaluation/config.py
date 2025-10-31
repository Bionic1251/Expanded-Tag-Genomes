
# Item types for item–tag score generation
# Can be set to "movies" or "books" or both
ITEM_TYPES = ["movies", "books"]

# Datasets available for evaluation
# "amazon" refers to Amazon product data
# "tg" refers to Tag Genome dataset
DATA_SETS = ["amazon", "tg"]

# Feature sets to generate/use
# "core": core features (e.g., BERT and non-BERT)
# "original": original features including tag probability
# "core_original": combination of core and original features
FEATURE_SETS = ["core", "original", "core_original"]


# File paths for Tag Genome datasets (root directories)
MOVIE_TAG_GENOME_PATH = "/movies"
BOOK_TAG_GENOME_PATH = "/books"

# File paths for Amazon datasets
# Metadata files contain product descriptions, highlights, etc.
AMAZON_MOVIE_METADATA_PATH = "meta_Movies_and_TV.jsonl"
AMAZON_MOVIE_REVIEWS_PATH = "Movies_and_TV.jsonl"

AMAZON_BOOK_METADATA_PATH = "meta_Books.jsonl"
AMAZON_BOOK_REVIEWS_PATH = "Books.jsonl"

# Chunk size for reading large datasets
# Determines how many rows are processed at a time
CHUNK_SIZE = 100000
