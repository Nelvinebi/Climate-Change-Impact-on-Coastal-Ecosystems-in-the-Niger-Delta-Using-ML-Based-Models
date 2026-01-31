
# ============================================================
# Climate Change Impact on Coastal Ecosystems in the Niger Delta
# Using ML-Based Models (Synthetic Data)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. Synthetic Climate & Ecosystem Data Generator
# ------------------------------------------------------------

def generate_climate_ecosystem_data(samples=2500):
    np.random.seed(42)

    temperature_c = np.random.normal(28.5, 1.8, samples).clip(24, 35)
    rainfall_mm = np.random.normal(260, 55, samples).clip(80, 500)
    sea_level_rise_cm = np.random.normal(4.2, 1.5, samples).clip(0.5, 10)
    salinity_ppt = np.random.normal(24, 6, samples).clip(5, 38)
    storm_frequency = np.random.poisson(3, samples).clip(0, 10)
    soil_moisture_index = np.random.uniform(0.3, 0.9, samples)

    ecosystem_health_index = (
        100
        - 1.5 * (temperature_c - 28)
        - 0.04 * rainfall_mm
        - 6.5 * sea_level_rise_cm
        - 1.2 * salinity_ppt
        - 2.8 * storm_frequency
        + 18 * soil_moisture_index
    )

    ecosystem_health_index += np.random.normal(0, 5, samples)
    ecosystem_health_index = np.clip(ecosystem_health_index, 0, 100)

    return pd.DataFrame({
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "sea_level_rise_cm": sea_level_rise_cm,
        "salinity_ppt": salinity_ppt,
        "storm_frequency": storm_frequency,
        "soil_moisture_index": soil_moisture_index,
        "ecosystem_health_index": ecosystem_health_index
    })

# Generate dataset
df = generate_climate_ecosystem_data()

# ------------------------------------------------------------
# 2. Feature Scaling & Train-Test Split
# ------------------------------------------------------------

X = df.drop("ecosystem_health_index", axis=1)
y = df["ecosystem_health_index"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------
# 3. Machine Learning Model
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Model Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Climate Change Impact Model Performance")
print("--------------------------------------")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# ------------------------------------------------------------
# 5. Feature Importance
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_)
plt.xlabel("Importance Score")
plt.title("Drivers of Coastal Ecosystem Degradation")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Climate Scenario Simulation
# ------------------------------------------------------------

def future_climate_scenario():
    scenario = pd.DataFrame([{
        "temperature_c": 31.5,
        "rainfall_mm": 380,
        "sea_level_rise_cm": 8.0,
        "salinity_ppt": 34,
        "storm_frequency": 7,
        "soil_moisture_index": 0.42
    }])

    scaled = scaler.transform(scenario)
    predicted_health = model.predict(scaled)[0]

    print("\nFuture Climate Scenario Assessment")
    print("----------------------------------")
    print(f"Predicted Ecosystem Health Index: {predicted_health:.1f} / 100")

    if predicted_health > 70:
        print("Healthy ecosystem condition")
    elif predicted_health > 40:
        print("Moderately stressed ecosystem")
    else:
        print("Severe ecosystem degradation risk")

future_climate_scenario()
