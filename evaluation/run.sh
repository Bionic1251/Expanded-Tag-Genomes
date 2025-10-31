#!/bin/bash

# Evaluation parameters are stored in config.py

# --- Preprocessing ---

# Match items between TG and Amazon datasets
python code/preprocessing/matcher.py

# Select matched TG item data to be processed
python code/preprocessing/data_selector_tg.py

# Select Amazon item data to be processed
python code/preprocessing/data_selector_amazon.py

# Split datasets into training/validation/test folds
python code/preprocessing/create_folds.py

# --- Generating features ---

# Core features

# Extract BERT-based features (GPU recommended)
python code/preprocessing/core/bert_features.py

# Extract other non-BERT features
python code/preprocessing/core/other_features.py

# Original features

# Prepare input for generating tag probability features with TagDL
python code/feature_generation/original/original_tg1.py

# Generate tag probability features using R
Rscript code/feature_generation/original/tag_prob.R

# Combine TG original features
python code/feature_generation/original/original_tg2.py

# Generate original features for Amazon items
python code/feature_generation/original/original_amazon.py

# Generate core features
python code/feature_generation/core/core.py

# Combine core and original features
python code/feature_generation/core_original/combine.py

# --- Evaluation ---

# Evaluate the Average baseline model
python code/evaluation/average/average.py

# Evaluate TagDL for all feature sets, datasets and item types
python code/evaluation/tagdl/tagdl.py

# Evaluate GLMER model using R
Rscript code/evaluation/glmer/glmer.R

# Print GLMER results
python code/evaluation/glmer/print_results.py