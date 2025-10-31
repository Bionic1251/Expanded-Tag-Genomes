# Score Generation

This folder contains the score generation part of the project. It is run independently of the `evaluation` folder, but some inputs may come from it. The purpose of this module is to preprocess data, generate features and produce tag relevance scores using the TagDL model.

---

## Running the project

1. **Set up the environment and dependencies**  
```bash
    sh start.sh
```

2. **Run the score generation**  
```bash
    sh run.sh
```

The `start.sh` script prepares the Python environment (version 3.10.9) and downloads required files. The `run.sh` script executes the score generation workflow.

---

## Requirements

- Python 3.10.9  
- Amazon dataset  
- Training dataset  
  - Either generate it by running the `evaluation` project (see its README)  
  - Or download the pre-built final dataset

Other datasets with the same structure can also be used.

---

## Configuration

All paths and options are set in the `config` file:

- Dataset paths — specify locations of the Amazon dataset and training dataset  
- Item type — choose between *movies* or *books* (only one type can be processed at a time)  

More details are provided in the configuration file.

---

## Workflow

The score generation process consists of three main steps:

1. **Dataset splitting**  
   - The Amazon dataset is divided into smaller chunks to simplify processing, fit data into memory and enable multithreading, since each chunk can be processed independently
   - The split data is stored under `data/raw/`

2. **Feature generation**  
   - For each chunk, multiple scripts generate features and store them as Python dictionaries serialized into pickle files  
   - This approach saves storage space and speeds up access  
   - Feature generation scripts accept `--from` and `--to` parameters to specify which chunks to process  
     - Example:  
         python code/preprocessing/generate_avg_pop_features.py --from 1 --to 10  
       This generates average rating and popularity features for all the items in the chunk that include items numbered 1–10  
   - Generated features are saved in `data/preprocessing/`

3. **Score generation**  
   - Tag relevance scores are produced using the TagDL model  
   - Results are written to `data/scores/`  
   - On the first run, the trained model is saved in `data/scores/` and reused in subsequent runs to avoid retraining

---

## Notes

- The system is designed to handle large datasets efficiently by chunking and parallel processing  
- Intermediate results are cached as pickle files to minimize recomputation  
- Training is required only once, subsequent runs reuse the saved model
