import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from catboost import CatBoostRegressor, Pool, cv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

"""" Constant value """
FILE_PATH = "YOUR PATH"
MODEL_FILENAME = '../../hockey_predictor/models/catboost_all_features_predictor_cv.cbm'
SCALER_FILENAME = '../../hockey_predictor/models/robust_scaler.joblib'
AVG_VALUES_FILENAME = '../../hockey_predictor/models/avg_values.joblib'
CLUBS_FILENAME = '../../hockey_predictor/models/clubs.joblib'

# Load data
df_in = pd.read_excel(FILE_PATH, engine='openpyxl')
# df = df_in.copy()
df = df_in.drop('Unnamed: 0', axis=1)

""" Pre-processing """
# Prepare data
features = [col for col in df.columns if col not in ['rank', 'year']]
X = df[features]
y = df['rank']
categorical_features = ['club']
numerical_features = [col for col in features if col not in categorical_features]

# avg_values = df.drop(columns=['rank', 'year', 'club']).mean().to_dict()
avg_by_club_df = df[features].groupby('club').mean()
avg_club_values = avg_by_club_df.to_dict('index')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normalization
scaler = RobustScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Split CV
cv_params = {'iterations': 400, 'loss_function': 'RMSE', 'random_seed': 42, 'verbose': 0}
cv_pool = Pool(X_train, y_train, cat_features=categorical_features)
cv_results = cv(pool=cv_pool, params=cv_params, fold_count=5, verbose=False)
"""Load and data transformation end"""

""" Fit model start """
best_iter = cv_results['iterations'].max()
cat_model = CatBoostRegressor(iterations=best_iter, learning_rate=0.2, loss_function='RMSE', random_seed=42,
                              verbose=0, allow_writing_files=False, depth=3, l2_leaf_reg=2, early_stopping_rounds=20)
cat_model.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_test, y_test), verbose=False)
""" Fit model end """


""" Model save start """
joblib.dump(scaler, SCALER_FILENAME)
club_options = sorted(df['club'].unique().tolist())
joblib.dump(club_options, CLUBS_FILENAME)
print(f"Club list saved to : {CLUBS_FILENAME}")

# avg_values = df.drop(columns=['rank', 'year', 'club']).mean().to_dict()
joblib.dump(avg_club_values, AVG_VALUES_FILENAME)
print(f"Average value saved to: {AVG_VALUES_FILENAME}")

cat_model.save_model(MODEL_FILENAME)
print(f"Model saved to: {MODEL_FILENAME}")


