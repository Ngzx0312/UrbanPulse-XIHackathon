import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
print("Loading Dataset...")
df = pd.read_csv("kl_training_data.csv")

# 2. Encode District Names to Numbers
le = LabelEncoder()
df['district_code'] = le.fit_transform(df['district'])

# 3. Train Models
print("Training Random Forest Models...")
X = df[["traffic", "weather", "intervention", "density", "district_code"]]

# Model A: Stress
model_stress = RandomForestRegressor(n_estimators=100, random_state=42)
model_stress.fit(X, df["stress"])

# Model B: ROI
model_roi = RandomForestRegressor(n_estimators=100, random_state=42)
model_roi.fit(X, df["roi"])

# 4. Save Artifacts
print("Saving Models locally...")
with open("model_stress.pkl", "wb") as f: pickle.dump(model_stress, f)
with open("model_roi.pkl", "wb") as f: pickle.dump(model_roi, f)
with open("encoder.pkl", "wb") as f: pickle.dump(le, f)

print("Training Complete! Models are ready.")