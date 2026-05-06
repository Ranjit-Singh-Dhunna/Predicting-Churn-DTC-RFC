#  Predicting Customer Churn - Decision Tree & Random Forest Classifier

 
Predicting subscriber churn at *StreamFlex*, a subscription-based streaming service, using Decision Tree Classification with hyperparameter tuning via GridSearchCV. 


---

## Overview

StreamFlex has experienced a rise in customer churn - subscribers cancelling their memberships. This project builds a **Decision Tree Classifier** to:

1. Identify the key factors driving customer churn
2. Build a predictive model that flags at-risk subscribers
3. Derive actionable business recommendations to reduce churn

The pipeline covers data cleansing, exploratory data analysis (EDA), model training, hyperparameter tuning with `GridSearchCV`, and final evaluation on a held-out test set. Click here to view [Visualisation](./diagrams_screenshots) 

---

## Dataset

| Property | Value |
|---|---|
| File | `customer_churn_cleaned.csv` |
| Records | **1,001 customers** |
| Features | 10 predictor variables |
| Target | `Churn` (0 = retained, 1 = churned) |
| Missing values | **None** |
| Overall churn rate | **~76%** |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Customer age in years |
| `Subscription_Length_Months` | Numeric | Duration of subscription (months) |
| `Watch_Time_Hours` | Numeric | Total hours of content watched |
| `Number_of_Logins` | Numeric | Total number of platform logins |
| `Payment_Issues` | Binary | Payment issues flag (0 = no, 1 = yes) |
| `Number_of_Complaints` | Numeric | Total complaints filed |
| `Resolution_Time_Days` | Numeric | Average complaint resolution time (days) |
| `Membership_Type` | Categorical | Basic / Standard / Premium |
| `Payment_Method` | Categorical | Credit Card / Debit / PayPal |
| `Preferred_Content_Type` | Categorical | Movies / TV Shows / Sports |
| **`Churn`** | Binary | **Target** - 0 = retained, 1 = churned |

---

## Project Structure

```
Predicting-Churn-DTC-RFC/
│
├── datacleansing.py               # Data loading, cleaning, EDA & initial visualisations
├── DecisionTree.py                # Full ML pipeline: EDA → preprocessing → training → tuning → evaluation
├── combined.ipynb                 # Jupyter notebook combining both scripts
│
├── customer_churn_cleaned.csv     # Cleaned dataset (output of datacleansing.py)
│
├── StreamFlex_Churn_Report.md     # Full analytical report with findings & recommendations
│
└── diagrams_screenshots/          # All generated plots
    ├── eda_01_churn_distribution.png
    ├── eda_02_numeric_distributions.png
    ├── eda_03_categorical_churn_rates.png
    ├── eda_04_correlation_heatmap.png
    ├── eda_05_boxplots.png
    ├── eda_boxplots.png
    ├── eda_correlation.png
    ├── eda_histograms.png
    ├── model_01_confusion_matrix.png
    ├── model_02_metrics_comparison.png
    ├── model_03_feature_importances.png
    ├── model_04_decision_tree_full.png
    ├── model_05_decision_tree_top4.png
    └── model_06_gridsearch_heatmap.png
```

---

## Methodology

### 1. Data Cleansing (`datacleansing.py`)

- Loads raw `customer_churn.csv` via `pandas`
- Displays shape, dtypes, and summary statistics
- Identifies and handles missing values (drops rows with nulls)
- Saves the cleaned dataset to `customer_churn_cleaned.csv`
- Generates preliminary EDA plots: histograms, box plots, correlation heatmap

### 2. ML Pipeline (`DecisionTree.py`)

**Step-by-step:**

| Step | Description |
|---|---|
| **1. Data Loading** | Load `customer_churn_cleaned.csv`, inspect shape and distribution |
| **2. EDA** | Generate churn distribution, numeric histograms, categorical churn rates, correlation heatmap, box plots |
| **3. Preprocessing** | Drop `CustomerID`, label-encode `Membership_Type`, `Payment_Method`, `Preferred_Content_Type` |
| **4. Train/Test Split** | 80/20 split with `stratify=y` to preserve class balance |
| **5. Baseline DT** | Train default `DecisionTreeClassifier` (no restrictions) |
| **6. GridSearchCV** | Exhaustive search over **224 hyperparameter combinations** with Stratified 5-Fold CV |
| **7. Evaluation** | Compare baseline vs. tuned model across Accuracy, Precision, Recall, F1 |
| **8. Visualisation** | Confusion matrix, metrics comparison, feature importances, tree visualisation |

### GridSearchCV Parameter Grid

```python
param_grid = {
    'max_depth'        : [3, 4, 5, 6, 7, 8, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 5, 10],
    'criterion'        : ['gini', 'entropy'],
}
# Scoring metric: F1 | CV: StratifiedKFold(n_splits=5)
```

### Best Hyperparameters Found

```
criterion        : gini
max_depth        : 7
min_samples_leaf : 10
min_samples_split: 2
Best CV F1-score : 0.8954
```

---

## Results

### Model Performance Comparison

| Metric | Baseline DT | Tuned DT | Improvement |
|---|---|---|---|
| Accuracy | 0.7600 (76.00%) | **0.8150 (81.50%)** | +5.50% |
| Precision | 0.8036 | **0.8333** | +2.97% |
| Recall | 0.8696 | **0.9275** | +5.79% |
| F1-Score | 0.8353 | **0.8840** | +4.87% |

> The tuned model correctly identifies **over 9 out of 10 actual churners** (Recall = 92.75%).

### Feature Importances (Tuned Model)

| Rank | Feature | Importance |
|---|---|---|
|  1 | `Number_of_Complaints` | **0.3017** |
|  2 | `Payment_Issues` | 0.1575 |
|  3 | `Membership_Type` | 0.1268 |
| 4 | `Watch_Time_Hours` | 0.1122 |
| 5 | `Resolution_Time_Days` | 0.1107 |

The root split of the Decision Tree is on `Number_of_Complaints ≤ 5.5`, confirming its dominant role.

### Classification Report (Tuned DT on Test Set)

```
              precision    recall  f1-score   support

    No Churn       0.75      0.56      0.64        48
     Churned       0.83      0.93      0.88       138

    accuracy                           0.82       186
   macro avg       0.79      0.74      0.76       186
weighted avg       0.81      0.82      0.81       186
```

---

## Visualisations

All plots are saved to the `diagrams_screenshots/` directory.

| Plot | Description |
|---|---|
| `eda_01_churn_distribution.png` | Pie chart - overall churn vs retained ratio |
| `eda_02_numeric_distributions.png` | Histograms of 6 numeric features by churn status |
| `eda_03_categorical_churn_rates.png` | Churn rate per category across 4 categorical features |
| `eda_04_correlation_heatmap.png` | Correlation heatmap of numeric features |
| `eda_05_boxplots.png` | Box plots of key features by churn status |
| `model_01_confusion_matrix.png` | Confusion matrix - tuned Decision Tree |
| `model_02_metrics_comparison.png` | Bar chart - Baseline vs Tuned model metrics |
| `model_03_feature_importances.png` | Feature importance ranking (Gini-based) |
| `model_04_decision_tree_full.png` | Full tuned Decision Tree (depth 7) |
| `model_05_decision_tree_top4.png` | Decision Tree truncated to top 4 levels (readable) |
| `model_06_gridsearch_heatmap.png` | GridSearchCV F1 heatmap - max_depth vs min_samples_split |

---

## Business Recommendations

Based on both EDA and model findings, three actionable interventions are recommended:

### 1.  Early-Engagement Intervention Programme
Target customers in their **first 6 months** with proactive outreach (push notifications, guided onboarding, curated content). Offer a loyalty discount at the 3-month mark.  
*Addresses: low Watch Time & short Subscription Length*

### 2.  Priority Support Queue for At-Risk Customers
Use the model to flag customers with **≥ 2 complaints** and/or resolution times exceeding 14 days. Route them to a dedicated support tier with a guaranteed **24-hour SLA**.  
*Addresses: Number of Complaints & Resolution Time*

### 3.  Frictionless Payment Recovery Flow
Implement an automated payment failure pipeline - real-time detection, 3-day grace periods, one-click payment updates, and account credits as goodwill.  
*Addresses: Payment Issues*

---

## Installation & Usage

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Step 1 - Data Cleansing

```bash
python datacleansing.py
```

Outputs: `customer_churn_cleaned.csv`, `eda_histograms.png`, `eda_boxplots.png`, `eda_correlation.png`

### Step 2 - Decision Tree Pipeline

```bash
python DecisionTree.py
```

Outputs: All EDA + model plots saved to the working directory.

### Step 3 - Notebook (Optional)

Open `combined.ipynb` in Jupyter for an interactive, end-to-end walkthrough:

```bash
jupyter notebook combined.ipynb
```

---

## Technologies

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and analysis |
| `scikit-learn` | Model training, GridSearchCV, evaluation metrics |
| `matplotlib` | Custom dark-themed visualisations |
| `seaborn` | Heatmaps and statistical plots |

- **Language:** Python 3.x  
- **Algorithm:** Decision Tree Classifier (CART)  
- **Tuning:** GridSearchCV with Stratified K-Fold cross-validation  
- **Criterion:** Gini Impurity

---

## References

- Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- StreamFlex. (2026). *Customer Churn Dataset*. Internal data release.

---

*COMP 471 · Project 1 · February 23, 2026*
