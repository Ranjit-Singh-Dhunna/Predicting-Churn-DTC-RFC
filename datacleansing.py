"""
Data Preparation and Exploration (EDA)
StreamFlex Customer Churn Prediction

Loads customer_churn.csv, displays structure and summary statistics,
identifies and handles missing values, and performs EDA (histograms,
box plots, correlation analysis). Uses pandas, matplotlib, seaborn.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load data

print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)
df = pd.read_csv('customer_churn.csv')
print("Dataset loaded successfully.")
print("\nFirst 10 rows:")
print(df.head(10))


# Step 2: Dataset structure

print("\n" + "=" * 60)
print("2. DATASET STRUCTURE")
print("=" * 60)
print("Shape (rows, columns):", df.shape)
print("\nColumn names and dtypes:")
print(df.dtypes)
print("\nInfo (non-null counts and dtypes):")
df.info()


# Step 3: Summary statistics

print("\n" + "=" * 60)
print("3. SUMMARY STATISTICS")
print("=" * 60)
print("Numerical summary:")
print(df.describe())
print("\nMembership_Type value counts:")
print(df['Membership_Type'].value_counts())
print("\nPayment_Method value counts:")
print(df['Payment_Method'].value_counts())
print("\nPreferred_Content_Type value counts:")
print(df['Preferred_Content_Type'].value_counts())
print("\nTarget (Churn) distribution:")
print(df['Churn'].value_counts())


# Step 4: Identify and handle missing values

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) == 0:
    print("No missing values found in the dataset.")
else:
    print(missing_df)

if df.isnull().any().any():
    rows_before = len(df)
    df = df.dropna()
    print(f"Dropped {rows_before - len(df)} rows with missing values. Shape now: {df.shape}")
else:
    print("No missing values to handle. Proceeding with full dataset.")

# Save cleansed dataset for downstream use (Tasks 2, 3) and for submission
df.to_csv('customer_churn_cleaned.csv', index=False)
print(f"Saved cleansed data to customer_churn_cleaned.csv ({len(df)} rows, {len(df.columns)} columns).")


# Step 5 Exploratory data analysis (visualizations)

numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']
                and c not in ['CustomerID', 'Churn']]

# Histograms (7 numeric features -> 3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=25, edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
# Hide unused subplots (we have 7 plots, grid has 9)
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Histograms of numeric features', fontsize=12)
plt.tight_layout()
plt.savefig('eda_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: eda_histograms.png")

# Box plots by Churn (7 numeric features -> 3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='Churn', y=col, ax=axes[i])
    axes[i].set_title(col + ' by Churn')
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Box plots of numeric features by Churn', fontsize=12)
plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_boxplots.png")

# Correlation matrix
corr_cols = numeric_cols + ['Churn']
corr = df[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Correlation matrix (numeric features and Churn)')
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_correlation.png")

print("\n" + "=" * 60)
print("5. CORRELATION WITH CHURN (target)")
print("=" * 60)
print(corr['Churn'].drop('Churn').sort_values(ascending=False))


# Step 6: Here is a brief description of patterns and anomalies (printed summary for convenience)

print("\n" + "=" * 60)
print("6. PATTERNS AND ANOMALIES (summary)")
print("=" * 60)
print("""
- Distributions: Age and Subscription_Length_Months are roughly symmetric;
  Watch_Time_Hours, Number_of_Logins, Resolution_Time_Days show some skew.
  Number_of_Complaints and Payment_Issues are discrete with limited range.
- Churn vs features: Box plots suggest higher Resolution_Time_Days and more
  complaints may be associated with churn; lower watch time or logins may
  also relate to churn (to be tested by models).
- Correlation with Churn: The heatmap shows which numeric features are most
  linearly associated with Churn. Categorical features need encoding for
  modeling.
- Anomalies: Check for impossible values (e.g. negative ages/times). For
  tree/random forest models, extreme outliers typically have limited impact.
""")
print("EDA complete. Figures saved in current directory.")
