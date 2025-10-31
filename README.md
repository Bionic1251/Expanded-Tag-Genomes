# Expanded Tag Genomes for Cross-Domain Recommendation

This repository is designed to tackle the **item-tag prediction problem** for **books** and **movies**. An **item** refers to either a book or a movie, and the goal is to predict the relevance of tags for items using multiple feature sets and models.

The repository consists of two main parts:

- [**Evaluation**](evaluation/README.md): Selects items for evaluation, prepares data, generates features and evaluates multiple models.
- [**Score Generation**](generation/README.md): Uses training data (from Evaluation or downloaded directly) to generate item-tag scores.

## Dataset
The dataset generated in this project can be accessed [here](https://strath-my.sharepoint.com/:f:/g/personal/denis_kotkov_strath_ac_uk/EoZz-jf_CRBPin-9e9g6uaQBLGLVffPKmydYbvK2RIpVLg?e=afdZdV).

## License

At this stage, the **code** and **dataset** in this repository are provided **for viewing purposes only**.  
You may **not use, copy, modify, or distribute** any part of the code or dataset at this time.

The materials are temporarily restricted while the paper associated with this repository is under review.  
Once the paper is **accepted for publication**, both the **code** and the **dataset** will be released under a **Creative Commons license** that allows legal and free use in academic and research settings.

---

## Feature Sets

The project uses three feature sets: **original**, **core** and **original_core**.  

### Original Feature Set
This feature set combines information about tag applications, user ratings, and user reviews to predict tag relevance for items.  

- **Tag applications**
  - `tag_exists`: Indicates whether a tag has been applied to an item, with 1 meaning applied and 0 meaning not applied
  - `lsi_tags_75`: Measures similarity between a tag and an item using latent semantic indexing (LSI), where items are represented by their applied tags

- **User ratings**
  - `rating_similarity`: Cosine similarity between the item's ratings and the aggregated ratings of items linked to the tag
  - `avg_rating`: The average rating given by users to the item.

- **User reviews**
  - `log_IMDB`: Log-scaled frequency of the tag appearing in user reviews for the item, calculated after applying stemming to both tags and reviews. 
  - `log_IMDB_nostem`: The same as above, but calculated without applying stemming, so original word forms are preserved.
  - `lsi_imdb_175`: LSI-based similarity between a tag and an item, where items are represented using the bag-of-words from their reviews

- **Feature interactions**
  - `tag_prob`: An estimated relevance score for the relationship between a tag and an item. It is computed using logistic regression, where the target variable is `tag_exists` and all other features are used as input.

### Core Feature Set
The core feature set focuses on tag applications, user ratings, and user reviews using alternative metrics:  

- **User ratings**
  - `avg_rating`: Mean rating of the item
  - `pop`: Item popularity, calculated as log of number of ratings
- **User reviews**
  - `lemma_review_mentions`: Log-scaled frequency of the tag in lemmatized reviews
  - `raw_review_mentions`: Same as above, but without lemmatization
  - `lemma_max_tfidf`, `lemma_mean_tfidf`: TF-IDF scores per lemmatized review, aggregated by max or mean
  - `raw_max_tfidf`, `raw_mean_tfidf`: Same as above, but without lemmatization
  - `bert_avg_sim`, `bert_max_sim`: Cosine similarity between BERT embeddings of tag and item reviews, aggregated by average or max
  - `bert_description`, `bert_highlights`: Cosine similarity between BERT embeddings of tag and item metadata (description and highlights)

### Original_Core Feature Set
Represents the combination of **core** and **original** features
