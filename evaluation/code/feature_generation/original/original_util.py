def transform(movie_df):
    movie_df.rename(columns={"movieId": "item_id",
                             "tag_movie_rating_similarity": "rating_similarity",
                             "term_freq_IMDB_log": "log_IMDB",
                             "term_freq_IMDB_log_nostem": "log_IMDB_nostem",
                             "avg_movie_rating": "avg_rating"}, inplace=True)
    movie_df = movie_df[
        ["item_id", "tag", "tag_exists", "rating_similarity", "log_IMDB", "log_IMDB_nostem", "lsi_tags_75",
         "lsi_imdb_175", "avg_rating", "tag_count", "lsi_imdb_25"]].copy()
    print("Feature record num", len(movie_df))
    movie_df = movie_df.drop_duplicates(subset=['item_id', "tag"])
    print("Feature record num after duplicate removal", len(movie_df))
    return movie_df