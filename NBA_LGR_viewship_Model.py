import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score)

df = pd.read_csv('nba_finals_ratings_CLEAN.csv')

median = df['Average_Viewers'].median()
df['High_Viewership'] = (df['Average_Viewers'] >= median).astype(int)

df['Series_Length'] = df['Results'].str.extract(r'(\d)-(\d)').sum(axis=1)
df['Was_Sweep'] = df['Results'].str.contains('4-0').astype(int)
df['Went_7_Games'] = df['Results'].str.contains('4-3').astype(int)

game_cols = [f'Game_{i}_Viewers' for i in range(1, 8) if f'Game_{i}_Viewers' in df.columns]
df['Avg_Game_Viewership_Variance'] = df[game_cols].var(axis=1)

games = []
for idx, row in df.iterrows():
    for game_num in range(1, 8):
        if pd.notna(row[f'Game_{game_num}_Viewers']):
            games.append({
                'Year': row['Year'],
                'Game_Number': game_num,
                'Viewers': row[f'Game_{game_num}_Viewers'],
                'Series_Avg': row['Average_Viewers'],
                'Is_Elimination_Possible': 1 if game_num >= 4 else 0,
                'High_Viewership': 1 if row[f'Game_{game_num}_Viewers'] >= median else 0
            })

games_df = pd.DataFrame(games)
games_df = games_df.merge(
    df[['Year', 'Series_Length', 'Was_Sweep', 'Went_7_Games', 'Avg_Game_Viewership_Variance']],
    on='Year',
    how='left'
)

X = games_df[[
    'Game_Number', 
    'Is_Elimination_Possible',
    'Series_Length',      
    'Went_7_Games',
    'Was_Sweep',
    'Avg_Game_Viewership_Variance'
]].fillna(0)

y = games_df['High_Viewership']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

L_G_R = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handles imbalance
    random_state=42
)
L_G_R.fit(X_train_scaled, y_train)

# Predictions
y_pred = L_G_R.predict(X_test_scaled)
y_proba = L_G_R.predict_proba(X_test_scaled)[:, 1]

# Metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred):.3f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LR (AUC={roc_auc_score(y_test, y_proba):.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Get feature importance from coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': L_G_R.coef_[0],
    'Abs_Coefficient': np.abs(L_G_R.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(feature_importance)

# Plot it
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression)')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()
