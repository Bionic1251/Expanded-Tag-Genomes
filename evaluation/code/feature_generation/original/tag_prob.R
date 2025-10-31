library(lme4)
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

# Build the config named list
config <- setNames(lapply(config_values_raw, parse_list), config_keys)

# Now you can access:
item_types <- config[["ITEM_TYPES"]]

cat("Variables:\n")
cat("ITEM_TYPES:\n")
print(item_types)



# ------- Utility Functions --------
untransform_identity <- function(x) x
transform_identity <- function(x) x

exclude <- function(excludeVals, x) {
  x[!x %in% excludeVals]
}

predict_glmer <- function(fit, data, features, untransformY = untransform_identity) {
  pred <- predict(fit, newdata = data, type = "response", allow.new.levels = TRUE)
  untransformY(pred)
}

fit_glmer <- function(x, y, features, transformY = transform_identity, excludeVals = c(),
                      family = binomial(link = "logit"),
                      control = glmerControl(optCtrl = list(maxfun = 500)),
                      weights = NULL, ...) {
  z <- transformY(exclude(excludeVals, y))
  form <- paste("z ~", paste(features, collapse = " + "))
  for (feature in features) {
    form <- paste(form, "+ (", feature, " - 1 | tag)", sep = "")
  }
  form <- paste(form, "+ (1 | tag)")
  form <- formula(form)

  if (is.null(weights)) {
    fit <- glmer(form, data = x, family = family, control = control)
  } else {
    fit <- glmer(form, data = x, weights = weights, family = family, control = control)
  }

  predictedY <- predict_glmer(fit, x, features)
  attr(fit, "secondaryFit") <- lm(y ~ predictedY)
  fit
}

# ------- EM Reweighting and Model Fitting --------
getTagFit <- function(dataTag, tagExists, posWeight, fitFunction, predictFunction, features, NUM_ITER = 3, samplingRatio = 1, sampleMinNeg = 0) {
  tags <- unique(dataTag$tag)

  if (samplingRatio < 1) {
    positiveIndices <- which(tagExists == 1)
    negativeIndicesKeep <- c()
    for (tag in tags) {
      negativeIndicesTag <- which(tagExists == 0 & dataTag$tag == tag)
      nNegative <- length(negativeIndicesTag)
      nInSample <- max(round(samplingRatio * nNegative), sampleMinNeg)
      nInSample <- min(nInSample, nNegative)
      negativeIndicesKeep <- c(negativeIndicesKeep, sample(negativeIndicesTag, nInSample))
    }
    indicesToKeep <- c(positiveIndices, negativeIndicesKeep)
    dataTag <- dataTag[indicesToKeep, ]
    tagExists <- tagExists[indicesToKeep]
    posWeight <- posWeight[indicesToKeep]
  }

  y <- tagExists
  weights <- posWeight

  for (tag in tags) {
    numPositive <- sum(weights[dataTag$tag == tag & y == 1])
    unlabeled <- dataTag$tag == tag & y == 0
    numUnlabeled <- sum(unlabeled)
    weights[unlabeled] <- ifelse(numUnlabeled > 0, numPositive / numUnlabeled, 0)
  }

  for (i in 1:NUM_ITER) {
    fit <- fitFunction(dataTag, y, features = features, transformY = transform_identity, weights = weights)

    p <- predictFunction(fit, dataTag, features, untransformY = untransform_identity)

    for (tag in tags) {
      numPositive <- sum(weights[dataTag$tag == tag & y == 1])
      unlabeled <- dataTag$tag == tag & y == 0
      weights[unlabeled] <- ifelse(p[unlabeled] > 0.5, 0, weights[unlabeled])
    }

    keepIndices <- !is.na(weights) & weights > 0
    dataTag <- dataTag[keepIndices, ]
    y <- y[keepIndices]
    weights <- weights[keepIndices]

    for (tag in tags) {
      numPositive <- sum(weights[dataTag$tag == tag & y == 1])
      unlabeled <- dataTag$tag == tag & y == 0
      numUnlabeled <- sum(unlabeled)
      weights[unlabeled] <- ifelse(numUnlabeled > 0, numPositive / numUnlabeled, 0)
    }
  }

  predict <- function(dataPredict) {
    predictFunction(fit, dataPredict, features, untransformY = untransform_identity)
  }

  list(predict = predict, fit = fit)
}

# ------- Load Data and Run --------

for (item_type in item_types) {

    input_folder <- sprintf("data/preprocessed/tg/original/%s/folds", item_type)
    output_folder <- sprintf("data/preprocessed/tg/original/%s/tag_prob", item_type)
    cat("Paths\n")
    cat(input_folder)
    cat("\n")
    cat(output_folder)

    for (fold in 0:9) {
        cat(sprintf("Processing fold %d...\n", fold))

        train_path <- file.path(input_folder, sprintf("train%d.csv", fold))
        test_path  <- file.path(input_folder, sprintf("test%d.csv", fold))

        train <- read.csv(train_path)
        test  <- read.csv(test_path)

        train$tag <- as.factor(train$tag)
        test$tag <- as.factor(test$tag)

        # Features
        features <- c("rating_similarity", "lsi_imdb_25", "avg_rating")

        # Labels and initial weights
        tag_exists <- train$tag_exists
        pos_weight <- rep(1, nrow(train))

        # Set number of iterations
        NUM_ITER <- 1

        # Fit using EM-style loop
        fit_result <- getTagFit(
          dataTag = train,
          tagExists = tag_exists,
          posWeight = pos_weight,
          fitFunction = fit_glmer,
          predictFunction = predict_glmer,
          features = features,
          NUM_ITER = NUM_ITER
        )

        # Predict tag_prob
        train$tag_prob <- fit_result$predict(train)
        test$tag_prob <- fit_result$predict(test)

        out_train_path <- file.path(output_folder, sprintf("train_tag_prob%d.csv", fold))
        out_test_path  <- file.path(output_folder, sprintf("test_tag_prob%d.csv", fold))

        # Save outputs
        write.table(train[, c("item_id", "tag", "tag_prob")], out_train_path, sep = ",", row.names = FALSE, quote = FALSE)
        write.table(test[, c("item_id", "tag", "tag_prob")], out_test_path, sep = ",", row.names = FALSE, quote = FALSE)

        cat("Saved tag_prob for train and test.\n")
    }
}

cat("All folds processed and saved.\n")