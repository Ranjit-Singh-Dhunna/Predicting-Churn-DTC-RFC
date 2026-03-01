# SECTION 0: IMPORTS
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Plot styling
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3d5c',
    'axes.labelcolor': '#e0e4ff',
    'text.color': '#e0e4ff',
    'xtick.color': '#a0a4bf',
    'ytick.color': '#a0a4bf',
    'grid.color': '#2a2d4a',
    'grid.alpha': 0.5,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#0f1117',
})

PALETTE = ['#5b6fe6', '#e65b7f']
ACCENT  = '#5b6fe6'
POP     = '#e65b7f'


print("  StreamFlex Customer Churn Analysis, Decision Tree Classifier")


# SECTION 1: DATA LOADING & OVERVIEW
print("\n[1] Loading and inspecting dataset...")

df = pd.read_csv('customer_churn_cleaned.csv')

print(f"\n  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Data types & non-null counts ---")
print(df.info())
print("\n--- Descriptive statistics (numeric) ---")
print(df.describe())
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\n--- Churn distribution ---")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"  Churn rate: {churn_counts[1] / len(df) * 100:.1f}%")

# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
print("\n[2] Generating EDA plots...")

# --- 2a. Overall Churn Distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Did Not Churn (0)', 'Churned (1)']
colors = [ACCENT, POP]
wedges, texts, autotexts = ax.pie(
    churn_counts, labels=labels, autopct='%1.1f%%',
    colors=colors, startangle=90,
    wedgeprops=dict(edgecolor='#0f1117', linewidth=2)
)
for t in texts + autotexts:
    t.set_color('#e0e4ff')
ax.set_title('Overall Churn Distribution', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('eda_01_churn_distribution.png')
plt.show()
print("  Saved: eda_01_churn_distribution.png")

# --- 2b. Numeric feature distributions by Churn ---
numeric_cols = ['Age', 'Subscription_Length_Months', 'Watch_Time_Hours',
                'Number_of_Logins', 'Number_of_Complaints', 'Resolution_Time_Days']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    for val, lbl, clr in zip([0, 1], ['No Churn', 'Churned'], [ACCENT, POP]):
        subset = df[df['Churn'] == val][col]
        ax.hist(subset, bins=25, alpha=0.65, label=lbl, color=clr, edgecolor='none')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle('Numeric Feature Distributions by Churn Status', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_02_numeric_distributions.png')
plt.show()
print("  Saved: eda_02_numeric_distributions.png")

# --- 2c. Categorical feature churn rates ---
cat_cols = ['Membership_Type', 'Payment_Method', 'Preferred_Content_Type', 'Payment_Issues']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, col in enumerate(cat_cols):
    ax = axes[i]
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    bars = ax.bar(churn_rate.index, churn_rate.values * 100, color=ACCENT, edgecolor='none', alpha=0.85)
    # Highlight max bar
    max_idx = churn_rate.values.argmax()
    bars[max_idx].set_color(POP)
    ax.set_title(f'Churn Rate by\n{col}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Churn Rate (%)', fontsize=9)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    for bar_val, bar in zip(churn_rate.values, bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar_val*100:.1f}%', ha='center', va='bottom', fontsize=8, color='#e0e4ff')
fig.suptitle('Churn Rate by Categorical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_03_categorical_churn_rates.png')
plt.show()
print("  Saved: eda_03_categorical_churn_rates.png")

# --- 2d. Correlation heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[numeric_cols + ['Churn']].corr()
mask = pd.DataFrame(True, index=corr.index, columns=corr.columns)
for i in range(len(corr)):
    for j in range(i, len(corr)):
        mask.iloc[i, j] = True
    for j in range(0, i):
        mask.iloc[i, j] = False
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', ax=ax,
    cmap='coolwarm', center=0, linewidths=0.5,
    linecolor='#0f1117', annot_kws={'size': 9},
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('eda_04_correlation_heatmap.png')
plt.show()
print("  Saved: eda_04_correlation_heatmap.png")

# --- 2e. Box plots: key features vs Churn ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
box_features = ['Subscription_Length_Months', 'Number_of_Complaints', 'Resolution_Time_Days']
for i, col in enumerate(box_features):
    ax = axes[i]
    data_0 = df[df['Churn'] == 0][col]
    data_1 = df[df['Churn'] == 1][col]
    bp = ax.boxplot([data_0, data_1], patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(color='#a0a4bf'),
                    capprops=dict(color='#a0a4bf'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4, color='#a0a4bf'))
    bp['boxes'][0].set_facecolor(ACCENT)
    bp['boxes'][1].set_facecolor(POP)
    ax.set_xticklabels(['No Churn', 'Churned'])
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle('Key Feature Distributions by Churn Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_05_boxplots.png')
plt.show()
print("  Saved: eda_05_boxplots.png")

# SECTION 3: PREPROCESSING
print("\n[3] Preprocessing data...")

df_model = df.copy()

# Drop CustomerID (not a feature)
if 'CustomerID' in df_model.columns:
    df_model.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
le = LabelEncoder()
cat_encode = ['Membership_Type', 'Payment_Method', 'Preferred_Content_Type']
for col in cat_encode:
    df_model[col] = le.fit_transform(df_model[col])

print("  Encoded columns:", cat_encode)
print(f"  Final feature set: {list(df_model.columns)}")
print(f"  Shape after preprocessing: {df_model.shape}")

X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

feature_names = list(X.columns)
print(f"\n  Features used: {feature_names}")
print(f"  Target distribution:\n{y.value_counts()}")

# SECTION 4: TRAIN / TEST SPLIT
print("\n[4] Splitting data 80/20 (train/test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Train churn rate : {y_train.mean()*100:.1f}%")
print(f"  Test  churn rate : {y_test.mean()*100:.1f}%")

# SECTION 5: BASELINE DECISION TREE
print("\n[5] Training baseline Decision Tree...")

dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
prec_base = precision_score(y_test, y_pred_base)
rec_base  = recall_score(y_test, y_pred_base)
f1_base   = f1_score(y_test, y_pred_base)

print(f"\n  Baseline Decision Tree (no tuning):")
print(f"    Accuracy  : {acc_base:.4f}")
print(f"    Precision : {prec_base:.4f}")
print(f"    Recall    : {rec_base:.4f}")
print(f"    F1-Score  : {f1_base:.4f}")
print(f"    Tree depth: {dt_base.get_depth()}")

# SECTION 6: GRIDSEARCHCV HYPERPARAMETER TUNING
print("\n[6] Running GridSearchCV for hyperparameter tuning...")

param_grid = {
    'max_depth'        : [3, 4, 5, 6, 7, 8, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 5, 10],
    'criterion'        : ['gini', 'entropy'],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"\n  Best parameters found:")
for k, v in grid_search.best_params_.items():
    print(f"    {k}: {v}")
print(f"\n  Best CV F1-score : {grid_search.best_score_:.4f}")

# SECTION 7: TUNED MODEL EVALUATION
print("\n[7] Evaluating tuned Decision Tree on test set...")

best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n  === Tuned Decision Tree - Final Metrics ===")
print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"    Tree depth: {best_dt.get_depth()}")
print(f"\n  Confusion Matrix:")
print(cm)
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churned']))

# --- Comparison table ---
print("\n  --- Baseline vs. Tuned Model ---")
print(f"  {'Metric':<16} {'Baseline':>10} {'Tuned':>10} {'Delta':>10}")
print(f"  {'-'*50}")
for metric, base_val, tuned_val in [
    ('Accuracy',  acc_base, acc),
    ('Precision', prec_base, prec),
    ('Recall',    rec_base, rec),
    ('F1-Score',  f1_base, f1),
]:
    delta = tuned_val - base_val
    sign  = '+' if delta >= 0 else ''
    print(f"  {metric:<16} {base_val:>10.4f} {tuned_val:>10.4f} {sign+f'{delta:.4f}':>10}")


# SECTION 8: VISUALISATIONS, MODEL RESULTS
print("\n[8] Generating model result plots...")

# --- 8a. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax,
    cmap='Blues', linewidths=2, linecolor='#0f1117',
    xticklabels=['No Churn', 'Churned'],
    yticklabels=['No Churn', 'Churned'],
    annot_kws={'size': 18, 'weight': 'bold'},
    cbar_kws={'shrink': 0.8}
)
ax.set_xlabel('Predicted Label', fontsize=12, labelpad=8)
ax.set_ylabel('True Label', fontsize=12, labelpad=8)
ax.set_title('Confusion Matrix, Tuned Decision Tree', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('model_01_confusion_matrix.png')
plt.show()
print("  Saved: model_01_confusion_matrix.png")

# --- 8b. Metrics bar chart (Baseline vs. Tuned) ---
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
base_vals = [acc_base, prec_base, rec_base, f1_base]
tune_vals = [acc, prec, rec, f1]

x   = list(range(len(metrics)))
w   = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar([xi - w/2 for xi in x], base_vals, w, label='Baseline DT', color=ACCENT, alpha=0.80, edgecolor='none')
bars2 = ax.bar([xi + w/2 for xi in x], tune_vals,  w, label='Tuned DT',    color=POP,   alpha=0.85, edgecolor='none')
ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Baseline vs. Tuned Decision Tree, Performance Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, color='#e0e4ff')
plt.tight_layout()
plt.savefig('model_02_metrics_comparison.png')
plt.show()
print("  Saved: model_02_metrics_comparison.png")

# --- 8c. Feature Importances ---
importances = best_dt.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [POP if imp == feat_imp_df['Importance'].max() else ACCENT
              for imp in feat_imp_df['Importance']]
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
               color=colors_bar, edgecolor='none', alpha=0.85)
ax.set_xlabel('Importance Score (Gini)', fontsize=11)
ax.set_title('Feature Importances, Tuned Decision Tree', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
for bar_h in bars:
    ax.text(bar_h.get_width() + 0.002, bar_h.get_y() + bar_h.get_height()/2,
            f'{bar_h.get_width():.4f}', va='center', fontsize=9, color='#e0e4ff')
plt.tight_layout()
plt.savefig('model_03_feature_importances.png')
plt.show()
print("  Saved: model_03_feature_importances.png")

# SECTION 9: DECISION TREE VISUALISATION
print("\n[9] Visualising the Decision Tree...")

# Full tree (may be large)
fig, ax = plt.subplots(figsize=(28, 12))
plot_tree(
    best_dt,
    feature_names=feature_names,
    class_names=['No Churn', 'Churned'],
    filled=True, rounded=True, ax=ax,
    fontsize=6, impurity=True, proportion=False,
    precision=3,
)
ax.set_title('Decision Tree, Tuned Model (Full)', fontsize=16, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('model_04_decision_tree_full.png', dpi=100)
plt.show()
print("  Saved: model_04_decision_tree_full.png")

# Pruned tree (depth 4) for readability
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    best_dt,
    feature_names=feature_names,
    class_names=['No Churn', 'Churned'],
    filled=True, rounded=True, ax=ax,
    fontsize=9, impurity=True, proportion=False,
    precision=3, max_depth=4,
)
ax.set_title('Decision Tree, Top 4 Levels (Readable View)', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('model_05_decision_tree_top4.png', dpi=120)
plt.show()
print("  Saved: model_05_decision_tree_top4.png")

# Text representation
print("\n  Decision Tree Rules (top 4 levels):")
tree_rules = export_text(best_dt, feature_names=feature_names, max_depth=4)
print(tree_rules)

# SECTION 10: GRIDSEARCHCV RESULTS SUMMARY
print("\n[10] GridSearchCV, Top 10 parameter combinations:")

cv_results = pd.DataFrame(grid_search.cv_results_)
top10 = (cv_results[['param_max_depth', 'param_min_samples_split',
                       'param_min_samples_leaf', 'param_criterion',
                       'mean_test_score', 'std_test_score']]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .reset_index(drop=True))
print(top10.to_string(index=False))

# Heatmap: max_depth vs min_samples_split for best criterion
best_crit = grid_search.best_params_['criterion']
pivot_data = (cv_results[cv_results['param_criterion'] == best_crit]
              .pivot_table(index='param_max_depth',
                           columns='param_min_samples_split',
                           values='mean_test_score',
                           aggfunc='max'))

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='#0f1117',
            annot_kws={'size': 9},
            cbar_kws={'shrink': 0.8, 'label': 'Mean CV F1'})
ax.set_title(f'GridSearchCV F1 Scores (criterion={best_crit})\n'
             f'max_depth vs min_samples_split', fontsize=12, fontweight='bold')
ax.set_xlabel('min_samples_split', fontsize=10)
ax.set_ylabel('max_depth', fontsize=10)
plt.tight_layout()
plt.savefig('model_06_gridsearch_heatmap.png')
plt.show()
print("  Saved: model_06_gridsearch_heatmap.png")

# SECTION 11: BUSINESS INSIGHTS
print("\n[11] BUSINESS INSIGHTS & RECOMMENDATIONS")

# Top features
top_feats = feat_imp_df.sort_values('Importance', ascending=False).head(5)
print("\n  Top 5 Churn Drivers (by feature importance):")
for _, row in top_feats.iterrows():
    print(f"    {row['Feature']:<35} {row['Importance']:.4f}")

print("  Analysis complete, all plots saved to current directory.")

