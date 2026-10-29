# Sentiment Analysis with NLP & Transformers

**AI Academia Internship | Shriansh Shrivastava | October 2026**

## About

An end-to-end sentiment analysis project using the IMDB Movie Reviews dataset. This project compares classical machine learning, a neural network, and a transformer-based model to classify movie reviews as positive or negative.

## Models Built

1. **Logistic Regression** : TF-IDF baseline
2. **Naive Bayes** : TF-IDF baseline
3. **Artificial Neural Network (ANN)** : PyTorch
4. **DistilBERT** : Transformer model using Hugging Face

## Dataset

- **Dataset:** IMDB Movie Reviews
- **Size:** 50,000 reviews
- **Split:** 25,000 training reviews and 25,000 test reviews
- **Labels:** Positive (1) and Negative (0)
- **Source:** Hugging Face Datasets library

The dataset is loaded through the code rather than stored in this repository.

## Key Findings

1. **Logistic Regression achieved 87.56% accuracy.** It was the strongest-performing model among the three evaluated on 25,000 test reviews, making it a useful baseline.
2. **Naive Bayes achieved 83.98% accuracy.** Its accuracy was 3.58 percentage points below Logistic Regression, but it remains a useful lightweight benchmark.
3. **ANN achieved 85.08% accuracy.** Its training loss decreased from 0.4267 to 0.1437 over five epochs, indicating improving performance on the training data.
4. **DistilBERT achieved 86.40% accuracy on 500 reviews.** A larger evaluation using the same test set as the other models is needed for a fair comparison.
5. **DistilBERT reported confidence scores of 0.99–1.00 on three custom reviews.** These results demonstrate its predictions on sample inputs, but further testing is needed to establish reliability.

## Results Summary

| Model | Accuracy | Evaluation set |
|---|---:|---|
| Logistic Regression | 87.56% | 25,000 reviews |
| Naive Bayes | 83.98% | 25,000 reviews |
| ANN | 85.08% | 25,000 reviews |
| DistilBERT | 86.40% | 500 reviews |

**Note:** DistilBERT was evaluated on a smaller sample, so its accuracy should not be directly compared with the other models as if all four used the same test set.

## Charts

The repository includes the following visualizations:

- `ann_loss.png` — ANN training loss
- `final_comparison_annotated.png` — Annotated model accuracy comparison
- `final_model_comparison.png` — Final model comparison
- `length_distribution.png` — Review-length distribution
- `lr_confusion_matrix.png` — Logistic Regression confusion matrix
- `model_comparison.png` — Model comparison
- `top_words.png` — Most predictive words
- `top_words_by_sentiment.png` — Predictive words by sentiment

## How to Run

### 1. Install the dependencies

Run the following commands in a terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install transformers torch datasets
```

### 2. Run the notebook

1. Clone or download this repository.
2. Install Jupyter Notebook if it is not already installed.
3. Open `Sentiment_Analysis_Final.ipynb` in Jupyter.
4. Run the notebook cells from top to bottom.

**Note:** Some libraries or additional dependencies may be required depending on the notebook environment. A working internet connection may be needed to download the dataset and pretrained model.

## Limitations

- The models were not all evaluated on the same number of reviews, limiting direct accuracy comparisons.
- The ANN uses TF-IDF features, which do not preserve word order or full contextual meaning.

## Author

**Shriansh** - AI Academia AI/ML Internship, 2026
