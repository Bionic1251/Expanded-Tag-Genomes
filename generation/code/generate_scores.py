import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import os
import preprocessing.util as util
from math import ceil
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

def log(msg, logfile="scores_debug.log"):
    with open(logfile, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, file=sys.stdout, flush=True)

train_df = pd.read_csv(config.TRAINING_DATA)
tags = train_df.tag.unique()

ITEM_CHUNK_SIZE = 100

item_val_dict_names = ["avg_rating", "pop"]
item_tag_val_dict_names = ["bert_avg_sim", "bert_max_sim", "raw_review_mentions",
                  "lemma_review_mentions", "raw_max_tfidf", "raw_mean_tfidf", "lemma_max_tfidf",
                      "lemma_mean_tfidf", "bert_description"]

if config.INCLUDE_BERT_PRODUCT_HIGHLIGHTS:
    item_tag_val_dict_names.append("bert_highlights")

numerical_cols = item_val_dict_names + item_tag_val_dict_names


class TagnavDataset(torch.utils.data.Dataset):
    def __init__(self, data, tag_to_idx):
        self.inputs = data.drop(columns=["item_id", "tag", "targets"])
        self.tag = data.tag
        self.item = data.item_id
        self.targets = data.targets

        self.tag_to_idx = tag_to_idx

    def __getitem__(self, idx):
        x_base = torch.tensor(np.array(self.inputs.iloc[idx]), dtype=torch.float32)

        item_id = self.item.iloc[idx]
        tag = self.tag.iloc[idx]
        tag_id = self.tag_to_idx[tag]
        tag_idx = torch.tensor(tag_id, dtype=torch.long)

        y = (torch.tensor(self.targets.iloc[idx], dtype=torch.float32) - 1.0) / 4
        return item_id, tag, tag_idx, x_base, torch.unsqueeze(y, 0)

    def __len__(self):
        return len(self.targets)


class TagnavModel(nn.Module):
    def __init__(self, num_tags, base_input_dim, embedding_dim, hidden_dim):
        super(TagnavModel, self).__init__()
        self.num_tags = num_tags

        input_dim = base_input_dim + num_tags  # tag one-hot + item emb + base

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tag_idx, base_features):
        # One-hot encode tags on the fly
        tag_one_hot = torch.nn.functional.one_hot(tag_idx, num_classes=self.num_tags).float()

        x = torch.cat((base_features, tag_one_hot), dim=1)
        return self.fc(x)

def train_model(train_df, all_tags, tag_to_idx, params):
    train_sampler = SubsetRandomSampler(range(len(train_df)))
    train_data_loader = torch.utils.data.DataLoader(TagnavDataset(train_df, tag_to_idx), batch_size=params["batch_size"], shuffle=False, sampler=train_sampler)

    device = torch.device("cpu")
    criterion = nn.L1Loss()

    embedding_dim = 4 # just in case
    base_input_dim = len(train_df.columns) - 3

    model = TagnavModel(
        num_tags=len(all_tags),
        base_input_dim=base_input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=params["hidden_layer_size"]
    )

    model_path = config.OUTPUT_SCORES + "/model.pt"
    if os.path.exists(model_path):
        print(f"Model found at {model_path}, loading and skipping training.")
        log(f"Model found at {model_path}, loading and skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    else:
        print("No saved model found. Training a new model.")
        log("No saved model found. Training a new model.")

    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])

    for epoch in range(params["epochs"]):
        print(f"epoch: {epoch}")
        log(f"epoch: {epoch}")
        model.train()
        running_loss = 0.0
        for item_id, tag, tag_idx, base_input, label in train_data_loader:
            tag_idx = tag_idx.to(device)
            base_input = base_input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(tag_idx, base_input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_data_loader)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    log(f"Model saved to {model_path}")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = {
    "batch_size": 32,
    "hidden_layer_size": 64,
    "lr": 1e-4,
    "epochs": 20, # 20
    "device": device
}





def build_feature_dataframe(ids, tags, feature_dicts):
    """
    Builds a DataFrame with one row per (parent_asin, tag) and one column per feature_dict.

    Parameters:
        ids (set or list): parent_asins to include.
        tags (set or list): tags to include.
        feature_dicts: one or more dictionaries with structure {parent_asin: {tag: value}}.

    Returns:
        pd.DataFrame with columns: parent_asin, tag, feature1, feature2, ...
    """
    data = []

    count = 0
    for parent_asin in ids:
        count += 1
        if count % 1000 == 0:
            print(count, "processed")
            log(f"{count}, processed")
        for tag in tags:
            row = [parent_asin, tag]
            for item in feature_dicts:
                dict_type = item[1]
                d = item[2]

                value = 0
                if dict_type == "item_tag_val":
                    value = d.get(parent_asin, {}).get(tag, 0)
                elif dict_type == "item_val":
                    value = d.get(parent_asin, 0)
                if pd.isna(value):
                    value = 0
                row.append(value)
            data.append(row)

    names = []
    for item in feature_dicts:
        names.append(item[0])

    df = pd.DataFrame(data, columns=['parent_asin', 'tag'] + names)
    return df

def get_predictions(df, tag_to_idx, model):
    train_sampler = SubsetRandomSampler(range(len(df)))
    train_data_loader = torch.utils.data.DataLoader(TagnavDataset(df, tag_to_idx), batch_size=len(df), shuffle=False, sampler=train_sampler)
    predictions = []
    print("Generating predictions")
    log("Generating predictions")
    count = 0
    with torch.no_grad():
        for item_id, tag, tag_idx, base_input, label in train_data_loader:
            count += 1
            print(f"Batch: {count}")
            log(f"Batch: {count}")
            tag_idx = tag_idx.to(device)
            base_input = base_input.to(device)

            output = model(tag_idx, base_input)
            predictions.append(pd.DataFrame({"parent_asin" : item_id, "tag" : tag, "score": output.cpu().numpy().flatten()}))
    return pd.concat(predictions)

def run_predictions(tag_to_idx, tags, scaler, numerical_cols, item_tag_val_dict_names, item_val_dict_names, model, in_path, out_path, ids, columns):
    item_tag_val_dicts = []
    for dict_name in item_tag_val_dict_names:
        d = {}
        with open(f"{in_path}/{dict_name}_dict.pkl", "rb") as f:
            d = pickle.load(f)
        item_tag_val_dicts.append((dict_name, "item_tag_val", d))

    item_val_dicts = []
    for dict_name in item_val_dict_names:
        d = {}
        with open(f"{in_path}/{dict_name}_dict.pkl", "rb") as f:
            d = pickle.load(f)
        item_val_dicts.append((dict_name, "item_val", d))

    feature_dicts = item_tag_val_dicts + item_val_dicts
    print(f"Dictionaries loaded {len(feature_dicts)}")
    log(f"Dictionaries loaded {len(feature_dicts)}")

    ids = list(ids)
    total_chunks = ceil(len(ids) / ITEM_CHUNK_SIZE)
    output_file = f"{out_path}/scores.csv"
    first_chunk = True
    for i in range(total_chunks):
        chunk_ids = ids[i * ITEM_CHUNK_SIZE: (i + 1) * ITEM_CHUNK_SIZE]
        print(f"Processing chunk {i * ITEM_CHUNK_SIZE, (i + 1) * ITEM_CHUNK_SIZE}")
        log(f"Processing chunk {i * ITEM_CHUNK_SIZE, (i + 1) * ITEM_CHUNK_SIZE}")

        features = build_feature_dataframe(chunk_ids, tags, feature_dicts)

        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)
        features[numerical_cols] = scaler.transform(features[numerical_cols])
        features.rename(columns={"parent_asin": "item_id"}, inplace=True)
        features["targets"] = 1
        # ordering columns as in training
        features = features[columns]
        predictions = get_predictions(features, tag_to_idx, model)
        predictions.score = predictions.score.clip(0, 1)
        # Append to file, write header only on first chunk
        predictions.to_csv(output_file, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
        first_chunk = False

        del features
        del predictions
        torch.cuda.empty_cache()
        gc.collect()


scaler = StandardScaler()

train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
columns = train_df.columns # to keep the column order for prediction

tag_to_idx = {tag: i for i, tag in enumerate(tags)}
model = train_model(train_df, tags, tag_to_idx, params)

# Get folders to process
folders = util.get_folders(config.OUTPUT_RAW)

for folder in folders:
    print(f"Processing {config.OUTPUT_PREPROCESSED}/{folder}")
    log(f"Processing {config.OUTPUT_PREPROCESSED}/{folder}")
    in_path = f"{config.OUTPUT_PREPROCESSED}/{folder}"
    out_path = f"{config.OUTPUT_SCORES}/{folder}"
    os.makedirs(out_path, exist_ok=True)

    id_df = pd.read_csv(f"{config.OUTPUT_RAW}/{folder}/items.csv")
    ids = id_df.parent_asin.unique()
    run_predictions(tag_to_idx, tags, scaler, numerical_cols, item_tag_val_dict_names, item_val_dict_names,
                    model, in_path, out_path, ids, columns)