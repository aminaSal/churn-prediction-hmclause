# Customer Churn Prediction – HM Clause (Mocked Data)

## 🎯 Objective
Predict customer churn using transactional and behavioural data in order to help the business anticipate disengagement and improve retention strategies.

## 🛠️ Tools & Technologies
- **SQL Server** (CTE, joins, data cleaning, feature engineering)
- **Python** (pandas, numpy, scikit-learn, statsmodels)
- **Machine Learning** (Logit model)
- **Visualisation** (matplotlib, seaborn)
- **Jupyter Notebooks**

## 📊 Key Results
- **AUC = 0.886**
- **Pseudo R² = 0.38**
- Identification of early churn signals
- Segmentation of risky customers
- Insights used to guide retention decisions (simulated dataset)

## 🚀 Workflow
1. **Extraction** of CRM & sales data using SQL (CTE, cleaning, joins)
2. **Data preprocessing** in Python (missing values, encoding, scaling)
3. **Feature engineering** (behavioural & transactional variables)
4. **Modelling** with Logit regression (statsmodels & scikit-learn)
5. **Evaluation** using ROC curve, AUC, pseudo R²
6. **Interpretation & recommendations**

## 📁 Repository Structure
churn-prediction-hmclause/
├── data/          → mock data only (no real HM Clause data)
├── sql/           → SQL scripts for extraction & cleaning
├── python/        → preprocessing + model scripts
├── notebooks/     → EDA & modelling notebooks
└── visuals/       → ROC curve, confusion matrix, plots

## ⚠️ Disclaimer
All data in this project is **fully anonymised or simulated**.  
No confidential HM Clause data is shared.

