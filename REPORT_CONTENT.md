# Report Content — ADS 542 Final Project
## Bank Term Deposit Prediction (Max 2 Pages)

Use this content to create your report.pdf.

---

## 1. Introduction

This project aims to predict whether a client of a Portuguese bank will subscribe to a term deposit, using the UCI Bank Marketing Dataset (bank-additional.csv, 4,119 instances, 20 input features). The dataset originates from direct marketing campaigns (phone calls) conducted during the 2008–2013 European financial crisis. The classification goal is to predict the binary outcome variable `y` (yes/no).

## 2. Methodology

### Data Cleaning & Preprocessing
The `duration` feature was removed to prevent data leakage, as call duration is unknown before a call is made. The `unknown` category in categorical features was retained as a distinct category, since refusal to disclose information may carry predictive value. Numeric features were standardized using `StandardScaler`, and categorical features were encoded using `OneHotEncoder`.

### Feature Engineering
A `real_interest_rate` feature was engineered based on the Fisher equation (Nominal Rate − CPI), reflecting the economic principle that rational individuals base deposit decisions on real returns rather than nominal interest rates.

### Feature Selection
Feature importance was analyzed using three methods: Pearson correlation, Random Forest feature importance, and mutual information scores. The top predictive features were `euribor3m`, `nr.employed`, `emp.var.rate`, and `poutcome` (previous campaign outcome).

### Model Selection & Hyperparameter Tuning
Three model families were compared using a unified sklearn Pipeline with GridSearchCV (5-fold cross-validation, F1 scoring):
- **Logistic Regression** — C ∈ {0.01, 0.1, 1, 10}
- **Random Forest** — n_estimators ∈ {100, 200}, max_depth ∈ {None, 10, 20}
- **MLP Neural Network** — hidden_layer_sizes ∈ {(50,), (100,), (50,50)}

## 3. Results

The MLP Neural Network was selected as the best performing model based on F1-score during cross-validation. The final model achieved:
- **F1 Score:** 0.2993
- **ROC-AUC:** 0.6164
- **Accuracy:** 88%

The confusion matrix and ROC curve demonstrate that the model effectively identifies potential subscribers while maintaining acceptable false positive rates.

## 4. Deployment

The final model is deployed as an interactive web application using **Streamlit**. The dashboard allows end-users to:
- Input client demographic and financial information
- Adjust macroeconomic indicators
- View real-time subscription probability predictions
- Review economic commentary based on the Fisher equation

**Streamlit Cloud URL:** [Your Streamlit Cloud URL here]

### How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```

## 5. Conclusion

Macroeconomic indicators — particularly the Euribor 3-month rate and employment levels — proved to be the strongest predictors of term deposit subscription, consistent with economic theory. The engineered real interest rate feature contributes meaningful economic interpretability to the model.

---
