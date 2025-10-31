runTenFolds <- function(in_path, out_path){
  levelup = in_path

  train0 = paste0(levelup, "train0.csv")
  train1 = paste0(levelup, "train1.csv")
  train2 = paste0(levelup, "train2.csv")
  train3 = paste0(levelup, "train3.csv")
  train4 = paste0(levelup, "train4.csv")
  train5 = paste0(levelup, "train5.csv")
  train6 = paste0(levelup, "train6.csv")
  train7 = paste0(levelup, "train7.csv")
  train8 = paste0(levelup, "train8.csv")
  train9 = paste0(levelup, "train9.csv")

  test0 = paste0(levelup, "test0.csv")
  test1 = paste0(levelup, "test1.csv")
  test2 = paste0(levelup, "test2.csv")
  test3 = paste0(levelup, "test3.csv")
  test4 = paste0(levelup, "test4.csv")
  test5 = paste0(levelup, "test5.csv")
  test6 = paste0(levelup, "test6.csv")
  test7 = paste0(levelup, "test7.csv")
  test8 = paste0(levelup, "test8.csv")
  test9 = paste0(levelup, "test9.csv")

  levelup = out_path

  predictions0 = paste0(levelup, "predictions_fold_0.txt")
  predictions1 = paste0(levelup, "predictions_fold_1.txt")
  predictions2 = paste0(levelup, "predictions_fold_2.txt")
  predictions3 = paste0(levelup, "predictions_fold_3.txt")
  predictions4 = paste0(levelup, "predictions_fold_4.txt")
  predictions5 = paste0(levelup, "predictions_fold_5.txt")
  predictions6 = paste0(levelup, "predictions_fold_6.txt")
  predictions7 = paste0(levelup, "predictions_fold_7.txt")
  predictions8 = paste0(levelup, "predictions_fold_8.txt")
  predictions9 = paste0(levelup, "predictions_fold_9.txt")

  source("predict_relevance_application.R")

  trainPredictTest(train0, test0, predictions0)
  trainPredictTest(train1, test1, predictions1)
  trainPredictTest(train2, test2, predictions2)
  trainPredictTest(train3, test3, predictions3)
  trainPredictTest(train4, test4, predictions4)
  trainPredictTest(train5, test5, predictions5)
  trainPredictTest(train6, test6, predictions6)
  trainPredictTest(train7, test7, predictions7)
  trainPredictTest(train8, test8, predictions8)
  trainPredictTest(train9, test9, predictions9)
}

config_file <- "config.py"

# Read and parse the config lines
config_lines <- readLines(config_file)
config_list <- strsplit(config_lines, "=")
config_keys <- trimws(sapply(config_list, `[`, 1))
config_values_raw <- trimws(sapply(config_list, `[`, 2))

# Function to parse Python-style lists, e.g., ["a", "b"]
parse_list <- function(x) {
  x <- gsub("\\[|\\]|\"", "", x)      # Remove brackets and quotes
  strsplit(x, ",\\s*")[[1]]           # Split on comma
}

# Set working directory to the directory of this script
this_file <- function() {
  # Works in Rscript (command line)
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  fileArg <- "--file="
  match <- grep(fileArg, cmdArgs)
  if (length(match) > 0) {
    return(normalizePath(sub(fileArg, "", cmdArgs[match])))
  }

  # Works in RStudio (if run via "Source")
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }

  # Fallback
  stop("Cannot determine script location.")
}

# Build the config named list
config <- setNames(lapply(config_values_raw, parse_list), config_keys)

# Now you can access:
item_types <- config[["ITEM_TYPES"]]

cat("Variables:\n")
cat("ITEM_TYPES:\n")
print(item_types)

script_dir <- dirname(this_file())
setwd(script_dir)

for (item_type in item_types) {
    cat("Current item type:\n")
    print(item_type)
    input_folder <- sprintf("../../../data/feature_folds/tg/original/%s/", item_type)
    output_folder <- sprintf("../../../data/evaluation_results/glmer/tg/%s/", item_type)

    runTenFolds(input_folder, output_folder)
}
