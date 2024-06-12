# Predictive Model for E-commerce Purchase Intent

## Overview
This project focuses on developing a predictive model to determine a user's purchase intention when visiting an e-commerce website. Various machine learning algorithms and techniques are employed to analyze session data and predict whether a user will make a transaction.

## Data
The dataset comprises 12,330 sessions obtained from the UCI ML repository, representing user interactions on an e-commerce platform over a year. The dataset includes 10,422 negative class samples (no purchase) and 1,908 positive class samples (purchase).

## Data Preparation and Cleaning
- **Feature Selection:** Using SelectKBest to select the most informative features.
- **Encoding:** Categorical variables are encoded using LabelEncoder.
- **Scaling:** Numerical features are scaled using StandardScaler.
- **Shuffling:** Data is shuffled to remove bias.

## Sampling Strategy
The training dataset is divided into batches with different ratios of positive to negative samples (1:1, 1:2, 1:3, 1:4) to analyze model performance under varying class distributions.

### Batch Performance Summary

| Batch | Algorithm         | Accuracy | Precision | Recall | F1 Score |
|-------|-------------------|----------|-----------|--------|----------|
| 1:1   | Decision Tree     | 79.64%   |  -        |  -     |  -       |
| 1:2   | Random Forest     | 86.98%   | 58.52%    | 75.18% | 65.81%   |
| 1:3   | XGBoost           | 88.52%   | 64.41%    | 69.58% | 66.90%   |
| 1:4   | Gradient Boosting | 87.88%   | 64.21%    | 61.55% | 62.86%   |

## Model Evaluation and Conclusion
- XGBoost consistently outperforms other algorithms across different batch sizes.
- Further optimization may enhance model performance.
- XGBoost is recommended for predicting purchase intent in e-commerce sessions.

## How to Run the Models
1. To run our Decision Tree Classifier - run /main/decision_tree.py
2. To run our SVM Classifier - run /main/svmGK.py
3. To run Scikit Deciison Tree Classifier - you can comment PART-a in the file /main/decision_tree.py and run PART-b
4. To run Scikit Gradient Boosting - run /main/gradient_boosting.py
5. To run Logistic Regression - run /main/logistic_regression.py
6. To run Random Forest - run /main/random_forest.py

**Note:** The original dataset is online_shoppers_intension.csv. The /main/data folder contains all the batches data, and the /main/test folder contains separate test sets.
