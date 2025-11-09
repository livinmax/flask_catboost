import itertools
import numpy as np
import pandas as pd
import re
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt


''' Upload data '''

path = 'YOUR PATH'
df_in = pd.read_excel(path)
df_temp = df_in.copy()

''' Defs for visual '''

def f_get_param_plot(str_title):
    plt.figure(figsize=(10, 10))
    plt.title(str_title, fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.set_theme(style="whitegrid", palette="Set2", font='sans-serif', font_scale=0.95)


def f_get_heatmap_plot(df, details):
    """In: dataframe; Out: heatmap plot"""
    try:
        str_title: str = 'Heatmap' + '_' + str(details)
        f_get_param_plot(str_title)
        sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True, fmt=".1f",
                    linewidth=.7, linecolor='white')
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


def f_get_bar_plot(df, x, y, hue, mean_val):
    """In: dataframe; Out: scatter plot; Step: save data"""
    try:
        str_title: str = 'Bar' + '_' + str(hue)
        f_get_param_plot(str_title)
        sns.barplot(x=x, y=y, data=df, palette="viridis")
        plt.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        # plt.legend()
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


def f_get_plot_hist_qq(data, name):
    """In: dataframe; Out: scatter plot; Step: save data"""
    try:
        str_title = 'Hist_QQ_Plots' + '_' + 'distribution_for' + '_' + str(name)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data, kde=True, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogramm for {name} ')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.subplot(1, 2, 2)
        sm.qqplot(data, line='s', ax=plt.gca(), fit=True)
        plt.title(f'Q-Q Plot for {name}')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


def f_get_df_style(df):
    """    """
    color_odd = '#d7e2fc'
    color_even = '#e8faec'
    font_color = '#020d04'

    def color_alternate_rows(row):
        row_index = row.name
        if row_index % 2 == 0:
            bg_color = color_odd
        else:
            bg_color = color_even
        style_string = f'background-color: {bg_color}; color: {font_color}'
        return [style_string for _ in range(len(row))]

    return df.style.apply(color_alternate_rows, axis=1)


def f_get_box_plot(df, x, y):
    """    """
    try:
        str_title: str = 'Box' + '_' + str(x) + '_' + str(y)
        f_get_param_plot(str_title)
        sns.boxplot(data=df, y=y, x=x)
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


def f_get_train_history(model):
    try:
        results = model.get_evals_result()
        epochs = len(results['learn']['RMSE'])
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), results['learn']['RMSE'], label='Train set')
        plt.plot(range(epochs), results['validation']['RMSE'], label='Test set')
        plt.title('Train CatBoost (RMSE per iteration)')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


def f_get_fact_vs_predict(y_test, y_pred):
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Фактический ранг (Y_test)')
        plt.ylabel('Предсказанный ранг (Y_pred)')
        plt.title('Фактический VS Предсказанный ранг')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f'Issue here, catch: {e}')


''' defs for research '''

def f_overview_print(df):
    print(df.head())  # print 5 rows of dataframe
    print(df.info())  # print columns and types
    print(df.isna().sum())  # print NAN
    print(df.isnull().sum())
    print(df.describe())
    print(df.dtypes)


def f_change_column_type(df, column_old: list, data_type):
    """In: dataframe; Out: dataframe; Step: change column type"""
    df_out = df.copy()
    for _ in column_old:
        df_out[_] = df_out[_].astype(data_type)
    return df_out


def f_get_time_to_seconds(time_value):
    if pd.isna(time_value) or time_value == '':
        return np.nan
    time_str = str(time_value).strip()
    match = re.match(r'(\d+):(\d+)', time_str)
    if match:
        try:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        except ValueError:
            return np.nan
    else:
        try:
            return float(time_str)
        except ValueError:
            return np.nan


def f_get_df_types_2(df, is_numeric: bool):
    """In: dataframe; Out: dataframe; Step: get df with types"""
    if is_numeric is True:
        df_out = df.select_dtypes(include=['int', 'int64', 'float', 'float64', 'int32', 'float32'])
    else:
        df_out = df.select_dtypes(include=['object', 'string'])
    return list(df_out)


def f_get_combination(lst_numeric):
    """In: dataframe; Out: dataframe; Step: get df with types"""
    lst_out = []
    for a, b in itertools.combinations(lst_numeric, 2):
        lst = [a, b]
        lst_out.append(lst)
    return lst_out


def f_get_overview(df, lst_numeric):
    """In: dataframe; Out: dataframe; Step: get df with types"""
    df_in = df.copy()
    lst_col, lst_observation_cnt, lst_mean, lst_median, lst_std, lst_skewness, lst_kurtosis = [], [], [], [], [], [], []
    for _ in lst_numeric:
        lst_col.append(_)
        lst_observation_cnt.append(len(df_in[_]))
        lst_mean.append(np.mean(df_in[_]))
        lst_median.append(np.median(df_in[_]))
        lst_std.append(np.std(df_in[_]))
        lst_skewness.append(stats.skew(df_in[_]))
        lst_kurtosis.append(stats.kurtosis(df_in[_]))

    lst_col_names = ['col_name', 'observation cnt', 'mean', 'median', 'std', 'skewness', 'kurtosis']

    df_out = pd.DataFrame(
        list(zip(lst_col, lst_observation_cnt, lst_mean, lst_median, lst_std, lst_skewness, lst_kurtosis)),
        columns=lst_col_names)

    return df_out


def f_check_distribution_normality(df, lst_numeric: list, alpha: float):
    """In: dataframe; Out: dataframe; Step: get df with types"""
    df_in = df.copy()
    for _ in lst_numeric:
        df_ = df_in[_]
        f_get_plot_hist_qq(df_, _)


def f_get_outliers_df(df, iqr_factor: float):
    """In: dataframe; Out: dataframe; Step: get df with types"""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    (lst_cnt, lst_col_numeric, lst_Q1, lst_Q3, lst_IQR, lst_lower_b, lst_upper_b, lst_col_outliers,
     lst_total) = ([], [], [], [], [], [], [], [], [])
    for col in numeric_cols:
        lst_col_numeric.append(col)
        if len(df[col].dropna()) < 4:
            lst_cnt.append('Not enough data')
        else:
            lst_cnt.append('OK')
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        lst_Q1.append(Q1)
        lst_Q3.append(Q3)
        lst_IQR.append(IQR)
        lst_lower_b.append(lower_bound)
        lst_upper_b.append(upper_bound)
        lst_col_outliers.append(len(col_outliers))
        lst_total.append(len(df[col]))

    lst_values = [numeric_cols, lst_Q1, lst_Q3, lst_IQR, lst_lower_b, lst_upper_b, lst_total, lst_col_outliers,
                  lst_cnt]
    lst_col_names = ['column_numeric', 'Q1', 'Q3', 'IQR', 'lower_bound',
                     'upper_bound', 'cnt_total_values', 'cnt_outliers', 'Comment']

    df_out = pd.DataFrame(list(zip(numeric_cols, lst_Q1, lst_Q3, lst_IQR, lst_lower_b, lst_upper_b, lst_total,
                                   lst_col_outliers, lst_cnt)), columns=lst_col_names)
    df_out['ratio, %'] = round(100 * (df_out['cnt_outliers'] / df_out['cnt_total_values']), 2)

    return df_out


def f_get_unique_val(df, col_name):
    """In: str; Out: list; Step: return unique values in column"""
    if type(col_name) is str:
        return list(df[col_name].unique())


def f_get_data_transform_power(data_array: np.ndarray, power: float = 2.0) -> np.ndarray:
    """ """
    transformed_array = np.power(data_array, power)

    return transformed_array


def f_get_data_transform_log_with_shift(x: np.ndarray, transform_type: str, shift_c: float):
    """ """
    if transform_type == 'forward':
        if (x + shift_c <= 0).any():
            min_val = np.min(x)
            raise ValueError(f"Issue: x + C should be > 0")
        transformed_array = np.log(x + shift_c)
        return transformed_array

    elif transform_type == 'inverse':
        transformed_array = np.exp(x) - shift_c
        return transformed_array


''' Pre-process data '''
# Transform time data to int
lst_time = ['penalty_time', 'penalty_time_opponent', 'game_time_equal_team', 'avg_game_time_equal_team',
            'game_time_wo_keeper', 'avg_game_time_wo_keeper']
for col in lst_time:
    df_temp[col] = df_temp[col].apply(f_get_time_to_seconds)

# Get lists
lst_numeric = f_get_df_types_2(df_temp, is_numeric=True)
lst_numeric.remove('rank')
lst_numeric.remove('year')
# lst_numeric.remove('game_numb')  # Duplicate
lst_numeric_comb = f_get_combination(lst_numeric)

lst_category = f_get_df_types_2(df_temp, is_numeric=False)
lst_category.append('rank')
lst_category.append('year')

# Transform data to int
df_temp = f_change_column_type(df_temp, lst_numeric, int)

''' Analyze data '''
# Check missed, mean values
df_overview_mean = f_get_df_style(f_get_overview(df_temp, lst_numeric))
# df_overview_mean

# Check distribution
f_check_distribution_normality(df_temp, lst_numeric, 0.05)

# Check outliers
df_outliers = f_get_df_style(f_get_outliers_df(df_temp, 1.5))
# df_outliers

# Check correlation
mtx_spearman = df_temp.corr(method='spearman', numeric_only=True)
f_get_heatmap_plot(mtx_spearman, 'Spearman')

# Check number of games for each club
df_group_club_game = df_temp.groupby('club')['game_numb'].sum().reset_index()
mean_val_game = df_group_club_game['game_numb'].mean()
f_get_bar_plot(df_group_club_game, 'game_numb', 'club', 'club', mean_val_game)

# Visualize box plot
for __ in lst_numeric:
    f_get_box_plot(df_temp, __, 'club')

''' Data manipulation '''
# Based on outliers remove raws with number of game less than average
median_game = df_group_club_game['game_numb'].mean()
df_filtered = df_group_club_game[df_group_club_game['game_numb'] < median_game]
lst_club_remove = f_get_unique_val(df_filtered, 'club')
df_temp = df_temp[~df_temp['club'].isin(lst_club_remove)]

# Based correlation remove column 'game_numb' cos of multicolinear
df_temp = df_temp.drop(columns=['game_numb', 'year'], axis=1)

# Based on data distribution transform some columns data to normal distribution
# lst_transform_power = ['avg_game_time_equal_team']
lst_transform_log = ['penalty_time', 'penalty_time_opponent',
                     'game_time_equal_team', 'game_time_wo_keeper']

# for _ in lst_transform_power:
#     df_temp[_] = f_get_data_transform_power(df_temp[_].to_numpy(), 2.0)

for _ in lst_transform_log:
    df_temp[_] = f_get_data_transform_log_with_shift(df_temp[_].to_numpy(), 'forward', 1.0)

# Extract data
df_temp.to_excel(path + 'df_out.xlsx')