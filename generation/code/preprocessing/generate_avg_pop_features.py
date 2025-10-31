import pandas as pd
import numpy as np
import util
import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config

def get_ratings(filename, fields, chunksize=100000):
    reader = pd.read_json(filename, lines=True, chunksize=chunksize)
    res = []
    for chunk in reader:
        res.append(chunk[fields])
    df = pd.concat(res)
    return df



# Get folders to process
folders = util.get_folders(config.OUTPUT_RAW)

for folder in folders:
    print(f"Processing {config.OUTPUT_RAW}/{folder}")
    os.makedirs(f"{config.OUTPUT_PREPROCESSED}/{folder}", exist_ok=True)

    ratings = get_ratings(f"{config.OUTPUT_RAW}/{folder}/reviews.json", ["parent_asin", "user_id", "rating"])
    #ratings = ratings.groupby(["parent_asin", "user_id"]).rating.mean().reset_index()

    avg_rating = ratings.groupby("parent_asin").rating.mean().reset_index()
    avg_rating.rename(columns={"rating": "avg_rating"}, inplace=True)

    pop = ratings.groupby("parent_asin").rating.count().reset_index()
    pop.rename(columns={"rating": "pop"}, inplace=True)
    pop["pop"] = np.log(pop["pop"] + 1)

    pop_dict = pop.set_index("parent_asin")["pop"].to_dict()
    avg_rating_dict = avg_rating.set_index("parent_asin")["avg_rating"].to_dict()

    pairs = [(pop_dict, "pop_dict"),
             (avg_rating_dict, "avg_rating_dict")]

    util.save_pickle_files(pairs, f"{config.OUTPUT_PREPROCESSED}/{folder}")