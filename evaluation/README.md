# Evaluation

This folder contains the evaluation part of the project. It can be run independently of the `generation` folder, although the outputs of evaluation can also serve as inputs to generation. The purpose of this module is to generate features and evaluate tag relevance prediction models.

---

## Running the project

1. **Set up the environment and dependencies**
```bash
sh start.sh
```
2. **Run the evaluation**
```bash
sh run.sh
```
Please note that the versions of packages are set up for Python 3.10.9. The project also uses R for some feature generation and model evaluation tasks. Python dependencies are listed in `requirements.txt` and R dependencies in `install_packages.R`.

---

## Datasets

The current version of the project is designed to work with two datasets:

- **Tag Genome dataset**  
- **Amazon dataset**

In principle, the project can work with any dataset that follows the same structure as these two.

---

## Configuration

The configuration file defines paths to the datasets and allows you to set item types, datasets and feature sets. Detailed descriptions of the available options are provided in the configuration file itself.

---

## Project structure

The subproject consists of two main folders:

- `code`: contains all scripts  
- `data`: stores intermediate and final results  

Additionally, to run the evaluation, you need the following unscaled Tag Genome feature files:
To run the evaluation, you need the following unscaled Tag Genome feature files:
evaluation/data/preprocessed/amazon/original/movies/features.txt
evaluation/data/preprocessed/amazon/original/books/features.txt
evaluation/data/preprocessed/tg/original/movies/features.txt
evaluation/data/preprocessed/tg/original/books/features.txt

These files can be either:

- **Generated** using [this repository](https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem), or  
- **Downloaded** from the published dataset.

For convenience, `data_default` mirrors the `data` folder and can be used for debugging.


---

## Workflow

The evaluation process involves several steps that roughly correspond to subfolders in `code` and `data`:

1. **Matching**  
Match Amazon items with labeled items from the Tag Genome dataset. Results are saved in `data/matches`.

2. **Data selection**  
Extract data associated with matched items and save it in `data/raw_selected`.

3. **Fold generation**  
Split labeled item–tag pairs into training and test folds (based on items) and store them in `data/feature_folds/item_tag_target_only`.  
These folds are reused to make folds for all the feature sets.

4. **Feature generation**  
- **Core features**:  
  Generated using two scripts — one for BERT-based features (GPU-friendly) and another for other features. Core features are produced for both Tag Genome and Amazon datasets (if enabled). Training data is also saved in `data/all_data`.  
- **Original features**:  
  Generated in multiple steps. First, prepare the data for tag probability feature generation. Then, generate the tag probability feature (implemented in R). Finally, combine features together.  
- **Core + Original features**:  
  Combine both feature sets for evaluation.

5. **Evaluation**  
Run and compare the following algorithms:  
- Average  
- Glmer (implemented in R)  
- TagDL  

Results are saved in `data/evaluation`.

---

## Programming languages

- **Python (tested with 3.10.9)**: feature generation, evaluation (TagDL, Average)  
- **R**: tag probability feature generation, Glmer evaluation  
