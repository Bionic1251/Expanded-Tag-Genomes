import argparse
import os
import re
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Select item range folders.")
    parser.add_argument("--from", dest="range_from", type=int, default=None, help="Start item index (inclusive)")
    parser.add_argument("--to", dest="range_to", type=int, default=None, help="End item index (inclusive)")
    return parser.parse_args()

def folder_matches_range(folder_name, range_from, range_to):
    match = re.match(r"^(\d+)-(\d+)$", folder_name)
    if not match:
        return False
    folder_start, folder_end = map(int, match.groups())
    if range_from is None and range_to is None:
        return True
    # Check for any overlap between [folder_start, folder_end] and [range_from, range_to]
    return (range_to is None or folder_start <= range_to) and (range_from is None or folder_end >= range_from)

def get_folders(base_path):
    args = parse_args()

    matching_folders = []
    for folder in os.listdir(base_path):
        if folder_matches_range(folder, args.range_from, args.range_to):
            matching_folders.append(folder)

    matching_folders.sort(key=lambda x: int(x.split('-')[0]))  # sort numerically by range start
    print("Matching folders:", matching_folders)
    return matching_folders

def convert_to_nested_dict(df, field_name):
    return {
        item_id: dict(zip(group["tag"], group[field_name]))
        for item_id, group in df.groupby("parent_asin")
    }

def save_pickle_files(pickle_pairs, path):
    for pair in pickle_pairs:
        dictionary, name = pair
        with open(f"{path}/{name}.pkl", "wb") as f:
            pickle.dump(dictionary, f)
        print(f"Saved {name} to {path}/{name}.pkl")