
# Whether to include BERT-based product highlights (set True for books, False for movies)
INCLUDE_BERT_PRODUCT_HIGHLIGHTS = True

# Path to the training dataset
TRAINING_DATA = "../evaluation/data/all_data/amazon/core/books.csv"

# Output directories for intermediate and final results
OUTPUT_RAW = "data/raw"                    # raw extracted data
OUTPUT_PREPROCESSED = "data/preprocessed"  # cleaned and feature-processed data
OUTPUT_SCORES = "data/scores"              # model-generated tag relevance scores

# Folder containing preprocessed review counts
REVIEW_COUNT_FOLDER = "../evaluation/data/preprocessed/amazon/core/books"

# Paths to metadata and review files
METADATA_PATH = "meta_Books.jsonl"
REVIEW_PATH = "Books.jsonl"

# Parameters for reading datasets in smaller parts
CHUNK_SIZE = 1000        # number of rows per read when loading data

# Parameters for saving datasets into multiple chunks
MAX_CHUNK_SIZE = 100000  # maximum rows per output chunk
MAX_CHUNKS = 30          # maximum number of chunks to generate

