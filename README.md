# End-to-End ML Pipeline for Customer Churn Prediction

## Objective
Build a reusable machine learning pipeline to predict customer churn using the Telco Churn Dataset, with preprocessing, Logistic Regression and Random Forest models, hyperparameter tuning, and pipeline export.

## Methodology
- **Dataset**: Telco Churn Dataset (7043 samples)
- **Preprocessing**: Handle missing values, encode categorical features (OneHotEncoder), scale numerical features (StandardScaler)
- **Models**: Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV (5-fold CV)
- **Evaluation**: Accuracy and F1-score
- **Export**: Saved pipeline as `churn_pipeline.pkl`

## Results
- Logistic Regression: Accuracy 0.8048, F1-Score 0.7996
- Random Forest: Accuracy 0.8020, F1-Score 0.7932
- Confusion Matrix: See `confusion_matrix_pipeline.png`

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook ml_pipeline.ipynb

Note: Telco Churn Dataset available at https://www.kaggle.com/blastchar/telco-customer-churn.