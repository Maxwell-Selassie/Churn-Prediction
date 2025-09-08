**Customer Churn Prediction Model**

This project builds and evaluates a machine learning model to predict customer churn. The aim is to help businesses identify customers at risk of leaving, so proactive retention strategies (e.g., offers, outreach) can be applied.

**Project Overview**

Objective: Predict whether a customer will churn (leave the service).

Approach:

Exploratory Data Analysis (EDA) to understand customer behavior.

Data preprocessing (handling missing values, encoding categorical features, scaling).

Training multiple ML models and evaluating their performance.

Hyperparameter tuning with cross-validation.


**Tech Stack**

Language: Python 3.11

Libraries:

pandas, numpy, matplotlib, seaborn → data handling + visualization

scikit-learn → preprocessing, pipelines, model training, evaluation

xgboost → gradient boosting model


Workflow
1. Data Preprocessing

Missing values imputed (SimpleImputer for categorical, IterativeImputer for numeric).

Categorical features → OneHotEncoding.

Numeric features → Scaling (StandardScaler).

Train-test split applied with stratification to balance churn vs. non-churn.

2. Models Trained

XGBoost 

3. Model Evaluation

Metrics used:

Accuracy: overall correctness of predictions.

Precision / Recall / F1: balance between false positives & false negatives.

ROC-AUC: ability to rank churn vs. non-churn correctly.

Brier Score: probability calibration check.

Example results (for tuned XGBoost):

Accuracy   : 0.86  
Precision  : 0.89
Recall     : 0.88  
F1-Score   : 0.88  
ROC-AUC    : 0.97  
Brier Score: 0.0361 (well-calibrated at threshold=0.3)

4. Diagnostics & Error Analysis

Confusion matrix to see where the model misclassifies churn vs. non-churn.

Calibration curve → checked probability reliability.

Key Insights

Customers on month-to-month contracts with high monthly charges are most likely to churn.

Longer-tenure customers with stable plans are less likely to churn.

Calibrated probability thresholds (0.3) gave better recall, catching more churners.

How to Run

Clone the repo:

git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction