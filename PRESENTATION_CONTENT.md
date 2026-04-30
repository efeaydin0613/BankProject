# Presentation Content — ADS 542 Final Project
## Bank Term Deposit Prediction (Max 5 Slides)

Use this content to create your PowerPoint presentation.

---

## Slide 1: Title & Problem Statement

**Title:** Bank Term Deposit Prediction — ADS 542 Final Project

**Content:**
- **Goal:** Predict whether a bank client will subscribe to a term deposit
- **Dataset:** UCI Bank Marketing — 4,119 samples, 20 features
- **Context:** Portuguese banking institution, 2008–2013 crisis period
- **Business Value:** Optimize marketing campaign targeting to improve conversion rates

---

## Slide 2: Data Pipeline

**Title:** Data Cleaning, Preprocessing & Feature Engineering

**Content:**
- **Data Cleaning:** Dropped `duration` (data leakage), handled `unknown` values as separate category
- **Preprocessing:** StandardScaler for numeric features, OneHotEncoder for categorical features
- **Feature Engineering:** Created `real_interest_rate` using Fisher Equation (Nominal Rate − CPI)
- **Pipeline:** sklearn Pipeline integrating preprocessing + classification in a single object

---

## Slide 3: Feature Selection & Analysis

**Title:** Feature Selection

**Content:**
- Methods used: Correlation analysis, Random Forest importance, Mutual Information
- Top features: `euribor3m`, `nr.employed`, `emp.var.rate`, `poutcome`
- Engineered `real_interest_rate` adds economic interpretability
- All features retained — regularization handles irrelevant ones

*(Include correlation heatmap and feature importance chart from notebook)*

---

## Slide 4: Model Comparison & Results

**Title:** Model Selection & Evaluation

**Content:**
- Compared 3 models via GridSearchCV (5-fold CV, F1 scoring):
  - Logistic Regression (C: 0.01–10)
  - Random Forest (100–200 trees, depth: None/10/20)
  - MLP Neural Network (various architectures)
- Best model: MLP Neural Network (hidden_layer_sizes=(100,))
- Metrics: Accuracy: 88%, F1 Score: 0.2993, ROC-AUC: 0.6164

*(Include confusion matrix and ROC curve from notebook)*

---

## Slide 5: Deployment & Conclusion

**Title:** Deployment & Conclusions

**Content:**
- Deployed on **Streamlit Cloud**: [Your URL here]
- Interactive dashboard with economic indicators
- Real-time prediction with Fisher equation analysis
- **Key insight:** Macroeconomic indicators (especially Euribor rate and employment) are the strongest predictors of deposit subscription

---
