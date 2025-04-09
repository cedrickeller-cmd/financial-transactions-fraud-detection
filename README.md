# Exploring Machine Learning Algorithms for Fraud Detection in Credit Card Transactions

This project explores machine learning algorithms for detecting fraudulent credit card transactions, aiming to improve existing solutions. The dataset, sourced from Kaggle, contains transaction data from European cardholders collected over two days in September 2013.

## Instructions

The repository includes the following Jupyter Notebooks, which can be run independently:

-   `EDA.ipynb`: Conducts exploratory data analysis. Given that the data primarily consists of principal components, preprocessing was limited to testing the impact of under- and oversampling within the random forest models (resampling performed in `random_forest.ipynb`).
-   `random_forest.ipynb`: Implements decision tree and random forest models with resampling and boosting, along with model performance analysis.
-   `knn_svm.ipynb`: Contains KNN and SVM models, providing performance metrics as outputs.

## Dataset Overview

The dataset comprises transaction data from European cardholders spanning two days in September 2013.

Features:

-   Non-transformed features include `Time` (seconds since the first transaction) and [Transaction] `Amount`.
-   Features `V1` through `V28` are the result of a PCA transformation, ensuring zero correlation between them.
-   `Class` represents the response variable: `0` for non-fraudulent and `1` for fraudulent transactions.

Data Source: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

## Modeling and Results

The dataset was divided into training (60%), validation (20%), and test (20%) sets, with a 5-fold stratified cross-validation implemented. To address the data's imbalanced nature, the training set was resampled for two random forest models: one using RandomUnderSampler to achieve a 1:1 ratio of normal to fraudulent transactions, and another using ADASYN to create a 2:1 ratio. For the KNN and SVM models, StandardScaler was applied to standardize the features.

A basic decision tree was initially used to establish a performance baseline. However, due to the problem's complexity, this approach was not pursued further. The random forest models, incorporating various sampling techniques, and the boosting model were optimized using GridSearchCV to enhance their performance.

| Model           | Test F1 (%) | Test AUC (%) |
| --------------- | ----------- | ------------ |
| Decision Tree   | 79.6        | 88.6         |
| Imbalanced RF   | 86.8        | 99.0         |
| Undersampled RF | 11.0        | 98.6         |
| Oversampled RF  | 86.3        | 98.8         |
| XGBoost         | 87.4        | 99.0         |
| KNN             | 83.2        | 86.7         |
| SVM             | 77.6        | 82.7         |

The XGBoost, oversampled random forest, and imbalanced random forest models demonstrated the strongest performance, achieving results comparable to other robust fraud detection techniques on this dataset.

## Future Work

Further hyperparameter optimization presents an opportunity for improvement. The XGBoost model, in particular, may benefit from early stopping during training to mitigate overfitting, though it already performs well on new data.

For enhanced sampling, exploring a GAN to generate synthetic fraudulent transactions could be beneficial, potentially replacing ADASYN or SMOTE. Additionally, neural network models could be compared to the best-performing random forest models for classification tasks.