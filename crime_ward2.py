import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import geopandas as gpd
import folium

# ========== 0. SELECT YEAR TO PREDICT ==========
TARGET_YEAR = 2022  # <<< Change this to predict any year from 2003–2022

# ========== 1. LOAD DATA ==========

df = pd.read_csv("../data/ChicagoCrime_2003-2022_Geo.csv", parse_dates=["Date"])

# Drop records without Wards or key fields
df = df.dropna(subset=["Ward", "Month", "Year", "Season", "Daypart"])
df['Ward'] = df['Ward'].astype(int)

# ========== 2. AGGREGATE TO WARD-MONTH LEVEL ==========

ward_monthly = df.groupby(['Year', 'Month', 'Ward']).size().reset_index(name='Crime_Count')

# ========== 3. FEATURE ENGINEERING ==========

# Sort and create lag features
ward_monthly = ward_monthly.sort_values(by=['Ward', 'Year', 'Month'])
ward_monthly['Crime_Lag1'] = ward_monthly.groupby('Ward')['Crime_Count'].shift(1)
ward_monthly['Crime_Lag3_Avg'] = ward_monthly.groupby('Ward')['Crime_Count'].rolling(3).mean().shift(1).reset_index(0, drop=True)

# Merge season and daypart
season_daypart = df.groupby(['Year', 'Month', 'Ward'])[['Season', 'Daypart']].agg(lambda x: x.mode()[0]).reset_index()
ward_model = pd.merge(ward_monthly, season_daypart, on=['Year', 'Month', 'Ward'], how='left')

# Drop rows with missing lag values
ward_model = ward_model.dropna()

# Encode categorical features
ward_model['Season'] = ward_model['Season'].astype('category').cat.codes
ward_model['Daypart'] = ward_model['Daypart'].astype('category').cat.codes

# ========== 4. SPLIT DATA ==========

feature_cols = ['Month', 'Year', 'Ward', 'Season', 'Daypart', 'Crime_Lag1', 'Crime_Lag3_Avg']
X = ward_model[feature_cols]
y = ward_model['Crime_Count']

train = ward_model[ward_model['Year'] != TARGET_YEAR]
test = ward_model[ward_model['Year'] == TARGET_YEAR]

X_train = train[feature_cols]
y_train = train['Crime_Count']
X_test = test[feature_cols]
y_test = test['Crime_Count']

# ========== 5. TRAIN MODEL ==========

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 6. EVALUATE ==========

print("\nEvaluation Metrics:")

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"Average Percent Accuracy: {accuracy:.2f}%")

# Merge predictions back
test = test.copy()
test['Predicted_Crime'] = y_pred

# ========== 7. LOAD WARD SHAPEFILE ==========
wards = gpd.read_file("../data/shape_ward/geo_export_65364e07-cd03-4783-b588-9b9d29263c58.shp")  # <<< Update path to your shapefile
wards['ward'] = wards['ward'].astype(int)

# ========== 8. MERGE WITH ACTUAL AND PREDICTED DATA ==========

# December Only (for clean map snapshot)
map_data = test[test['Month'] == 7]
map_df = wards.merge(map_data, left_on='ward', right_on='Ward', how='left')

# ========== 9. MATPLOTLIB STATIC MAPS WITH LABELS ==========

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Predicted Crime Map
map_df.plot(column='Predicted_Crime', ax=ax[0], legend=True, cmap='OrRd', edgecolor='black')
ax[0].set_title(f"Predicted Crime - Wards (Jul {TARGET_YEAR})")
ax[0].axis('off')

# Actual Crime Map
map_df.plot(column='Crime_Count', ax=ax[1], legend=True, cmap='Blues', edgecolor='black')
ax[1].set_title(f"Actual Crime - Wards (Jul {TARGET_YEAR})")
ax[1].axis('off')

# Add labels
for idx, row in map_df.iterrows():
    if pd.notnull(row['Predicted_Crime']):
        point = row['geometry'].representative_point()
        txt = ax[0].text(  # capture the Text object in `txt`
            point.x, point.y,
            f"{int(row['Predicted_Crime'])}",
            horizontalalignment='center',
            fontsize=7,
            color='black',
            fontweight='bold'
        )
        # Apply white outline
        txt.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white'),
            path_effects.Normal()
        ])
    if pd.notnull(row['Crime_Count']):
        point = row['geometry'].representative_point()
        txt = ax[1].text(
            point.x, point.y,
            f"{int(row['Crime_Count'])}",
            horizontalalignment='center',
            fontsize=7,
            color='black',
            fontweight='bold'
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white'),
            path_effects.Normal()
        ])

plt.tight_layout()
plt.show()

# ========== 10. FOLIUM INTERACTIVE MAP ==========

m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

map_df_valid = map_df.dropna(subset=['Predicted_Crime'])

folium.Choropleth(
    geo_data=map_df_valid,
    data=map_df_valid,
    columns=['ward', 'Predicted_Crime'],
    key_on='feature.properties.ward',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Predicted Crime Count'
).add_to(m)

# Optional: Add popups with crime counts
for _, row in map_df_valid.iterrows():
    point = row['geometry'].centroid
    folium.Marker(
        location=[point.y, point.x],
        popup=f"Ward {row['ward']}\nPredicted: {int(row['Predicted_Crime'])}"
    ).add_to(m)

# Save map
m.save(f"Predicted_Crime_Ward_{TARGET_YEAR}.html")
print(f"\nInteractive map saved as Predicted_Crime_Ward_{TARGET_YEAR}.html")
