import pandas as pd
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import config

torch.manual_seed(42)
FOLDS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_fold(test_df, train_df, path, params):
    print(f"test_df.shape: {test_df.shape}; train_df.shape: {train_df.shape}")

    all_tags = set(list(test_df.tag.unique()) + list(train_df.tag.unique()))
    len(all_tags)

    all_items = set(list(test_df.item_id.unique()) + list(train_df.item_id.unique()))
    len(all_items)

    print(f"Number of tags: {len(all_tags)}; Number of items: {len(all_items)}")


    with open(f"{path}/params.txt", "w") as file:
        file.write(str(params))

    class TagnavDataset(torch.utils.data.Dataset):
        def __init__(self, data, tag_to_idx, item_to_idx):
            self.inputs = data.drop(columns=["item_id", "tag", "targets"])
            self.tag = data.tag
            self.item = data.item_id
            self.targets = data.targets

            self.tag_to_idx = tag_to_idx
            self.item_to_idx = item_to_idx

        def __getitem__(self, idx):
            x_base = torch.tensor(np.array(self.inputs.iloc[idx]), dtype=torch.float32)

            item = self.item.iloc[idx]
            item_id = self.item_to_idx[item]
            item_idx = torch.tensor(item_id, dtype=torch.long)
            tag = self.tag.iloc[idx]
            tag_id = self.tag_to_idx[tag]
            tag_idx = torch.tensor(tag_id, dtype=torch.long)

            y = (torch.tensor(self.targets.iloc[idx], dtype=torch.float32) - 1.0) / 4
            return item_idx, tag_idx, x_base, torch.unsqueeze(y, 0)

        def __len__(self):
            return len(self.targets)

    class TagnavModel(nn.Module):
        def __init__(self, num_items, num_tags, base_input_dim, embedding_dim, hidden_dim):
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

        def forward(self, item_idx, tag_idx, base_features):

            # One-hot encode tags on the fly
            tag_one_hot = torch.nn.functional.one_hot(tag_idx, num_classes=self.num_tags).float()

            x = torch.cat((base_features, tag_one_hot), dim=1)
            #x = base_features
            return self.fc(x)

    tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    tag_df = pd.DataFrame(list(tag_to_idx.items()), columns=["tag", "tag_id"])
    item_df = pd.DataFrame(list(item_to_idx.items()), columns=["item", "item_id"])

    train_sampler = SubsetRandomSampler(range(len(train_df)))
    train_data_loader = torch.utils.data.DataLoader(TagnavDataset(train_df, tag_to_idx, item_to_idx), batch_size=params["batch_size"], shuffle=False, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(TagnavDataset(test_df, tag_to_idx, item_to_idx), batch_size=params["batch_size"], shuffle=False)

    device = torch.device("cpu")
    criterion = nn.L1Loss()

    embedding_dim = 4 # just in case
    base_input_dim = len(train_df.columns) - 3

    model = TagnavModel(
        num_items=len(all_items),
        num_tags=len(all_tags),
        base_input_dim=base_input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=params["hidden_layer_size"]
    )

    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])

    performance = {"epoch": [],
                   "train_loss": [],
                   "val_loss": []}
    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0.0
        for item_idx, tag_idx, base_input, label in train_data_loader:
            item_idx = item_idx.to(device)
            tag_idx = tag_idx.to(device)
            base_input = base_input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(item_idx, tag_idx, base_input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_data_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item_idx, tag_idx, base_input, label in test_loader:
                item_idx = item_idx.to(device)
                tag_idx = tag_idx.to(device)
                base_input = base_input.to(device)
                label = label.to(device)

                output = model(item_idx, tag_idx, base_input)
                val_loss += criterion(output, label).item()

        val_loss /= len(test_loader)
        print(f"Epoch {epoch}: Train Loss = {running_loss:.4f}, Val Loss = {val_loss:.4f}")
        performance["epoch"].append(epoch)
        performance["train_loss"].append(running_loss)
        performance["val_loss"].append(val_loss)


    res = None
    with torch.inference_mode():
        for item_idx, tag_idx, base_input, label in test_loader:
            item_idx = item_idx.to(device)
            tag_idx = tag_idx.to(device)
            base_input = base_input.to(device)
            label = label.to(device)

            outputs_test = model(item_idx, tag_idx, base_input)
            obj = {"item_id" : item_idx.numpy(), "tag_id" : tag_idx.numpy(), "predicted" : outputs_test.squeeze().numpy(), "targets" : label.squeeze().numpy()}
            if res is None:
                res = pd.DataFrame(obj)
            else:
                res = pd.concat([res, pd.DataFrame(obj)])
    res.columns = ["item_id", "tag_id", "predicted", "targets"]
    res = pd.merge(res, tag_df, on="tag_id", how="left")
    res = pd.merge(res, item_df, on="item_id", how="left")
    res.predicted = res.predicted.clip(lower=0, upper=1)
    mae = mean_absolute_error(res.targets, res.predicted) * 4
    return mae, res, pd.DataFrame(performance)

def run(in_path, out_path):
    print(f"Reading data from {in_path} and saving to {out_path}")
    start = time.time()

    # Saving the current script
    script_path = os.path.abspath(sys.argv[0])
    script_name = os.path.basename(script_path)

    #BATCH_SIZE = 32
    #HIDDEN_LAYER_SIZE = 64
    #LEARNING_RATE = 1e-4
    #NUM_EPOCHS = 20
    params = {
        "batch_size": 32,
        "hidden_layer_size": 64,
        "lr": 1e-4,
        "epochs": 20,
        "device": device
    }

    all_mae = []
    log = f"PyTorch version: {torch.__version__}\n"
    for i in range(FOLDS):
        test = pd.read_csv(f"{in_path}/test{i}.csv")
        train = pd.read_csv(f"{in_path}/train{i}.csv")
        train = train.rename(columns={"movieId" : "item_id"})
        test = test.rename(columns={"movieId" : "item_id"})

        mae, results_df, performance_df = run_fold(test, train, out_path, params)
        with open(f"{out_path}/{i}mae.txt", "w") as file:
            file.write(str(mae))
        log += f"Training log:\n{performance_df}\n\n"
        results_df.to_csv(f"{out_path}/results{i}.csv", index=False)
        all_mae.append(mae)

    end = time.time()
    log += f"time taken: {end - start}\n"
    mean_mae = np.mean(all_mae)
    log += f"All MAE: {all_mae}\nAverage MAE: {mean_mae}\n"
    with open(f"{out_path}/log.txt", "w") as file:
        file.write(log)
    return mean_mae

for item_type in config.ITEM_TYPES:
    for data_set in config.DATA_SETS:
        for feature_set in config.FEATURE_SETS:
            print(f"Processing {item_type}, dataset: {data_set}, features: {feature_set}")
            in_path = f"data/feature_folds/{data_set}/{feature_set}/{item_type}"
            out_path = f"data/evaluation_results/tagdl/{data_set}/{feature_set}/{item_type}"
            run(in_path, out_path)
            print("Done")