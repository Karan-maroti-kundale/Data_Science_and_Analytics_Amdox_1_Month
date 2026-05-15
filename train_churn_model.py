import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

print("1. Loading the Feature Master data...")
# Load the clean data you created in Phase 1
df = pd.read_csv("feature_master_online_retail_ii.csv")

# Create a simple 'Churn' label (Example: 1 if they haven't bought in 60 days, else 0)
df['churn_label'] = (df['recency_days'] > 60).astype(int)

# Select the features your API will use
features = ['recency_days', 'frequency', 'monetary', 'product_diversity_score']
X = df[features]
y = df['churn_label']

print("2. Splitting data into training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training the XGBoost AI Model...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

print("4. Saving the trained model...")
# This is the magic line that creates your .pkl file!
joblib.dump(model, "models/churn_xgb.pkl")

print("Success! Your churn_xgb.pkl file is now saved in the models/ folder.")