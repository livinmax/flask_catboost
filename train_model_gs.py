import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor, Pool, cv

# Upload data
path = 'YOUR PATH'
df_in = pd.read_excel(path)
df_temp = df_in.copy()

# Prepare data
df = df_temp.copy()
features = [col for col in df.columns if col not in ['rank', 'year']]
X = df[features]
y = df['rank']
categorical_features = ['club']
numerical_features = [col for col in features if col not in categorical_features]

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)],
    remainder='passthrough')


# Set Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('catboost', CatBoostRegressor(random_seed=42, verbose=0,
                                   allow_writing_files=False))])

# Set GS
cat_index_start = len(numerical_features)
cat_indices = list(range(cat_index_start, cat_index_start + len(categorical_features)))
fit_params = {'catboost__cat_features': cat_indices}
param_grid = {
    'catboost__iterations': [300, 500, 700],
    'catboost__learning_rate': [0.2, 0.25, 0.3, 0.4, 0.5],
    'catboost__depth': [2, 3, 5, 7],
    'catboost__early_stopping_rounds': [30, 50],
    'catboost__loss_function': ['RMSE', 'MAE'],
    'catboost__l2_leaf_reg': [1, 2, 3],
    'catboost__eval_metric': ['RMSE']}

# Set CV and GS
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='r2',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1)

# Fit model
grid_search.fit(X, y, **fit_params)

# Extract metrics for GS
results_df = pd.DataFrame(grid_search.cv_results_)
n_folds = cv_strategy.n_splits
r2_cols = [col for col in results_df.columns if col.startswith('split') and col.endswith('_test_score')]
r2_cols.append('mean_test_score')
r2_by_fold_df = results_df[r2_cols].copy()
r2_by_fold_df.columns = [f'R2_Fold_{i + 1}' for i in range(n_folds)] + ['R2_Mean']
r2_by_fold_df = r2_by_fold_df.round(4)

params_df = results_df['params'].apply(pd.Series).rename(columns=lambda x: x.replace('catboost__', ''))
params_by_fold_df = pd.concat([params_df, r2_by_fold_df], axis=1)

# Extract data
df_out_sorted = params_by_fold_df.sort_values(by='R2_Mean', ascending=False)
df_out_sorted.to_excel(path + 'df_out.xlsx')