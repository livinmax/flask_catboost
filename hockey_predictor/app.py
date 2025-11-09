import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request
from catboost import CatBoostRegressor, Pool

path = 'YOUR PATH'
MODEL_PATH = path + 'catboost_all_features_predictor_cv.cbm'
SCALER_PATH = path + 'robust_scaler.joblib'
AVG_CLUB_VALUES_PATH = path + 'avg_values.joblib'
CLUBS_PATH = path + 'clubs.joblib'

FEATURES = ['club', 'win', 'win_overtime', 'win_bullit',
            'loss_bullits', 'loss_overtime', 'loss', 'score', 'game_wo_scored_goal',
            'game_wo_missed_goal', 'goal_scored', 'goal_missed', 'penalty_time',
            'penalty_time_opponent', 'game_time_equal_team', 'avg_game_time_equal_team',
            'game_time_wo_keeper', 'avg_game_time_wo_keeper']

CATEGORICAL_FEATURES = ['club']
NUMERICAL_FEATURES = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]


def time_to_seconds(time_value):
    if pd.isna(time_value) or time_value == '': return np.nan
    time_str = str(time_value).strip()
    match = re.match(r'(\d+):(\d+)', time_str)
    if match:
        try:
            return int(match.group(1)) * 60 + int(match.group(2))
        except ValueError:
            return np.nan
    else:
        try:
            return float(time_str)
        except ValueError:
            return np.nan


try:
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    AVG_CLUB_VALUES = joblib.load(AVG_CLUB_VALUES_PATH)
    CLUB_OPTIONS = joblib.load(CLUBS_PATH)
    if 'New Club X' not in CLUB_OPTIONS:
        CLUB_OPTIONS.append('New Club X')
    temp_df = pd.DataFrame(AVG_CLUB_VALUES).T
    AVG_OVERALL = temp_df.mean().to_dict()
    print("Data has been loaded.")
except Exception as e:
    print(f"Loading error: {e}")
    exit()

app = Flask(__name__)


def make_prediction(club, goal_scored, goal_missed):
    if club in AVG_CLUB_VALUES:
        new_data_dict = AVG_CLUB_VALUES[club].copy()
    else:
        new_data_dict = AVG_OVERALL.copy()

    club_input = club
    goal_scored_input = time_to_seconds(goal_scored)
    goal_missed_input = time_to_seconds(goal_missed)

    if goal_scored_input is None or goal_missed_input is None or np.isnan(goal_scored_input) or np.isnan(
            goal_missed_input):
        raise ValueError("Incorrect format, input should be int")

    new_data_dict.update({'club': club_input, 'goal_scored': goal_scored_input, 'goal_missed': goal_missed_input})
    input_values = [new_data_dict.get(col) if col != 'club' else club_input for col in FEATURES]
    new_data = pd.DataFrame([input_values], columns=FEATURES)
    new_data_numerical = new_data[NUMERICAL_FEATURES].copy()

    if new_data_numerical.isnull().values.any():
        raise ValueError(
            "Error: There is NaN values in data")

    scaled_data = scaler.transform(new_data_numerical.values)
    new_data[NUMERICAL_FEATURES] = scaled_data

    new_pool = Pool(new_data, cat_features=CATEGORICAL_FEATURES)
    prediction = model.predict(new_pool)[0]

    return int(round(prediction))


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None

    input_data = {
        'club': CLUB_OPTIONS[0] if CLUB_OPTIONS else '',
        'goal_scored': '0',
        'goal_missed': '0'}

    if request.method == 'POST':
        try:
            club = request.form['club'].strip()
            goal_scored = request.form['goal_scored'].strip()
            goal_missed = request.form['goal_missed'].strip()
            input_data.update({
                'club': club,
                'goal_scored': goal_scored,
                'goal_missed': goal_missed})

            predicted_rank = make_prediction(club, goal_scored, goal_missed)

            prediction_result = {
                'club': club,
                'goal_scored': goal_scored,
                'goal_missed': goal_missed,
                'rank': predicted_rank}

        except ValueError as ve:
            prediction_result = {'error': f"Input error: {str(ve)}"}
        except Exception as e:
            prediction_result = {'error': f"Check server's log's"}

    return render_template('index.html',
                           clubs=CLUB_OPTIONS,
                           result=prediction_result,
                           input=input_data)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
