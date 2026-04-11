import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("final_master_dataset.csv")

def calculate_aqi(row):
    
    try:
        p25 = float(row['pm2.5'])
        p10 = float(row['pm10'])
        return max(p25, p10) 
    except:
        return np.nan

if 'aqi' not in df.columns:
    print("Calculating aqi...")
    df['aqi'] = df.apply(calculate_aqi, axis=1)


df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['dayofweek'] = df['timestamp'].dt.dayofweek


features = ['pm2.5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'ozone', 'day', 'month', 'dayofweek']
target = 'aqi'

df = df.dropna(subset=features + [target])


le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

final_features = ['city_encoded', 'pm2.5', 'pm10', 'no2', 'nh3', 'so2', 'co', 'ozone', 'day', 'month', 'dayofweek']

X = df[final_features]
y = df['aqi']


print("Model training started..")
model = RandomForestRegressor(
    n_estimators=100,    # Trees kam karo (100 ki jagah 50)
    max_depth=10,       # Tree ki height limit karo
    min_samples_leaf=5, # Overfitting roko
    random_state=42
)
model.fit(X, y)
joblib.dump(le, 'city_encoder.pkl')
joblib.dump(model, 'aqi_7day_model.pkl', compress=3)


print("Success! 'aqi_7day_model.pkl' generated.")
