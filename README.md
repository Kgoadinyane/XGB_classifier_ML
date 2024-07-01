# Sasol Data Analysis and Prediction Project utilising XGB_classifier_ML

This project involves data analysis and prediction using machine learning techniques on the Sasol dataset. In order to create a machine-learning model that can forecast the probability of each customer becoming inactive and refraining from making any transactions for a period of 90 days.

Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Results](#results)

## Introduction

This project utilizes machine learning algorithms such as XGBoost and Random Forest to predict if a csutomer will inactive or active based on features collected in the last 90 days to the days of prediction. Which will enable a business to identify customers who may be on the verge of becoming inactive, allowing them to implement strategies in advance to retain these customers.

## Dataset

The dataset used in this project includes:
- `Train.csv`: Training data with features and target variable.
- `Test.csv`: Test data with features.
- `VariableDescription.csv`: Description of variables/features in the dataset.
There are 1.88 million clients in this dataset. There are 1.5 million clients in train and 380,000 clients in test.

The dataset used in this project includes the following variables:

1. **ID**: Unique identifier for each record.
2. **region**: Geographical region where the data was collected.
3. **tenure**: The duration or length of time a customer has been with the company.
4. **Amount**: The amount of money spent by the customer.
5. **refill_frequency**: The frequency at which a customer refills their account.
6. **revenue**: Total revenue generated from the customer.
7. **arpu_segment**: Average Revenue Per User segment, categorizing users based on their spending.
8. **frequency**: The frequency of customer transactions or interactions.
9. **data_volume**: The volume of data consumed by the customer.
10. **on_net**: The amount of usage within the same network.
11. **Procuct_1**: Usage or subscription to the first product.
12. **Procuct_2**: Usage or subscription to the second product.
13. **zone1**: Indicator or variable related to a specific zone or area (zone 1).
14. **zone2**: Indicator or variable related to a specific zone or area (zone 2).
15. **mrg**: Marital status or other related demographic information.
16. **regularity**: Regularity of customer usage or interaction with the service.
17. **top_pack**: The most frequently used or preferred package by the customer.
18. **freq_top_pack**: Frequency of usage of the top package.
19. **Target**: The target variable for prediction, indicating the desired outcome or class label.

### Example Data Record

| ID | region | tenure | Amount | refill_frequency | revenue | arpu_segment | frequency | data_volume | on_net | Procuct_1 | Procuct_2 | zone1 | zone2 | mrg | regularity | top_pack | freq_top_pack | Target |
|----|--------|--------|--------|------------------|---------|--------------|-----------|-------------|--------|-----------|-----------|-------|-------|-----|------------|----------|---------------|--------|
| 1  | North  | 24     | 150    | 4                | 1200    | High         | 12        | 500         | 300    | Yes       | No        | 1     | 0     | Single | High        | Pack_A   | 6             | 1      |

## Installation

To run the code in this project, you'll need to have the following libraries installed:

- **pandas**: For data manipulation and analysis, especially for reading and handling CSV files.
- **numpy**: For numerical operations, particularly when dealing with arrays and mathematical functions.
- **matplotlib**: For creating static, animated, and interactive visualizations in Python.
- **seaborn**: For making statistical graphics and enhancing matplotlib visualizations.
- **scikit-learn**: For machine learning algorithms and tools, including preprocessing, model training, and evaluation.
- **xgboost**: For implementing the XGBoost machine learning algorithm, which is used for classification and regression tasks.

## Usage 

1. **Load Data**: Load the training and test data using pandas.
2. **Preprocess Data**: Preprocess the data including encoding categorical variables and scaling features.
3. **Train Models**: Train machine learning models (XGBoost and Random Forest) on the training data.
4. **Evaluate Models**: Evaluate the performance of the models using metrics such as confusion matrix and classification report.
5. **Make Predictions**: Use the trained models to make predictions on the test data.
6. **Submission**: Create a submission file for the predicted results.

**Libraries and Their Usage**

1. **pandas**: Used for loading and manipulating the dataset, including reading CSV files, handling missing values, and transforming data.
2. **numpy**: Used for performing numerical operations and handling arrays, which are essential for mathematical computations and data manipulation.
3. **matplotlib**: Used for visualizing the data and model results, including plotting graphs and charts to understand data distribution and model performance.
4. **seaborn**: Used for creating advanced visualizations and statistical graphics, enhancing the basic plotting capabilities of matplotlib.
5. **scikit-learn**: Used for data preprocessing (e.g., encoding categorical variables, scaling features), splitting data into training and test sets, and implementing machine learning algorithms such as Random Forest. It also provides tools for model evaluation.
5. **xgboost**: Used for implementing the XGBoost algorithm, which is a powerful and efficient machine learning model for classification and regression tasks.

## Modeling
**The project includes the following steps for modeling:**
1. **Data Preprocessing**: Preprocessing the data by handling categorical variables, missing values, and scaling features.
2. **Model Training**: Training XGBoost and Random Forest models on the preprocessed data.
3. **Model Evaluation**: Evaluating the models using accuracy, confusion matrix, and classification report.

## Evaluation
The models are evaluated using the following metrics:

- Accuracy
- Confusion Matrix
- Classification Report
