from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, confusion_matrix

app = FastAPI()

# Get the directory of the current script
backend_dir = os.path.dirname(__file__)
model_path = os.path.join(backend_dir, "model.pkl")
data_path = os.path.join(backend_dir, "newdata.csv")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
pipeline = joblib.load(model_path)

# Define the input data model
class UserInput(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    cuisine_top1: str
    spice_tolerance: int
    dietary_pref: str
    eating_out_per_week: int
    food_budget_per_meal: int
    sweet_tooth_level: int
    tea_vs_coffee: str
    movie_genre_top1: str
    series_genre_top1: str
    content_lang_top1: str
    ott_top1: str
    binge_freq_per_week: int
    screen_time_hours_per_week: int
    gaming_days_per_week: int
    gaming_hours_per_week: int
    game_genre_top1: str
    gaming_platform_top1: str
    esports_viewing: str
    social_platform_top1: str
    daily_social_media_minutes: int
    primary_content_type: str
    content_creation_freq: str
    music_genre_top1: str
    listening_hours_per_day: int
    music_lang_top1: str
    live_concerts_past_year: int
    reads_books: str
    book_genre_top1: str
    books_read_past_year: int
    fashion_spend_per_month: int
    shopping_mode_pref: str
    ethical_shopping_importance: int
    travel_freq_per_year: int
    travel_type_top1: str
    budget_per_trip: int
    travel_planning_pref: int
    hobby_top1: str
    club_top1: str
    weekly_hobby_hours: int
    introversion_extraversion: int
    risk_taking: int
    conscientiousness: int
    open_to_new_experiences: int
    teamwork_preference: int

@app.get("/data-summary")
def data_summary():
    df = pd.read_csv(data_path)

    # Replace NaN with None for JSON compatibility
    df = df.replace({np.nan: None})

    # Binarize teamwork preference, handling potential None values
    df['preference'] = df['teamwork_preference'].apply(lambda x: 'Team' if x is not None and x >= 4 else 'Solo')

    preference_distribution = df['preference'].value_counts().to_dict()
    introversion_distribution = df['introversion_extraversion'].dropna().value_counts().to_dict()
    risk_taking_distribution = df['risk_taking'].dropna().value_counts().to_dict()
    recent_submissions = df.tail(5).to_dict(orient='records')

    # Calculate model accuracy and confusion matrix
    # Ensure we only use rows where the target is not null for a fair evaluation
    eval_df = df.dropna(subset=['teamwork_preference', "introversion_extraversion", "risk_taking", "weekly_hobby_hours", "club_top1"])
    X = eval_df[["introversion_extraversion", "risk_taking", "weekly_hobby_hours", "club_top1"]]
    y_true = eval_df['teamwork_preference'].apply(lambda x: 1 if x >= 4 else 0)

    if not X.empty:
        y_pred = pipeline.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        accuracy = 0
        cm = [[0,0],[0,0]]


    return {
        "preference_distribution": preference_distribution,
        "introversion_distribution": introversion_distribution,
        "risk_taking_distribution": risk_taking_distribution,
        "recent_submissions": recent_submissions,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

@app.post("/predict")
def predict(data: UserInput):
    # These are the columns for prediction
    prediction_features = ["introversion_extraversion", "risk_taking", "club_top1", "weekly_hobby_hours"]

    # Create a dataframe for prediction from the input data
    prediction_df = pd.DataFrame([data.dict()], columns=prediction_features)

    # Make a prediction
    prediction = pipeline.predict(prediction_df)[0]
    prediction_proba = pipeline.predict_proba(prediction_df)[0]

    # Create a full dataframe to append to the csv
    # This should match the structure of the original data.csv
    full_data_columns = [
        'timestamp', 'age', 'height_cm', 'weight_kg', 'cuisine_top1', 'cuisine_top2', 'cuisine_top3',
        'spice_tolerance', 'dietary_pref', 'eating_out_per_week', 'food_budget_per_meal', 'sweet_tooth_level',
        'tea_vs_coffee', 'movie_genre_top1', 'movie_genre_top2', 'movie_genre_top3', 'series_genre_top1',
        'series_genre_top2', 'series_genre_top3', 'content_lang_top1', 'content_lang_top2', 'content_lang_top3',
        'ott_top1', 'ott_top2', 'ott_top3', 'binge_freq_per_week', 'screen_time_hours_per_week',
        'gaming_days_per_week', 'gaming_hours_per_week', 'game_genre_top1', 'game_genre_top2', 'game_genre_top3',
        'gaming_platform_top1', 'gaming_platform_top2', 'gaming_platform_top3', 'esports_viewing',
        'social_platform_top1', 'social_platform_top2', 'social_platform_top3', 'daily_social_media_minutes',
        'primary_content_type', 'content_creation_freq', 'music_genre_top1', 'music_genre_top2', 'music_genre_top3',
        'listening_hours_per_day', 'music_lang_top1', 'music_lang_top2', 'live_concerts_past_year',
        'reads_books', 'book_genre_top1', 'book_genre_top2', 'book_genre_top3', 'books_read_past_year',
        'fashion_spend_per_month', 'shopping_mode_pref', 'ethical_shopping_importance', 'travel_freq_per_year',
        'travel_type_top1', 'travel_type_top2', 'travel_type_top3', 'budget_per_trip', 'travel_planning_pref',
        'hobby_top1', 'hobby_top2', 'club_top1', 'club_top2', 'weekly_hobby_hours', 'introversion_extraversion',
        'risk_taking', 'conscientiousness', 'open_to_new_experiences', 'teamwork_preference'
    ]

    # Create a dictionary for the new row, with NaNs for missing values
    new_row_data = {col: np.nan for col in full_data_columns}
    new_row_data.update(data.dict())
    new_row_data['timestamp'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # For columns not in UserInput, set to a default or NaN
    new_row_data['cuisine_top2'] = np.nan
    new_row_data['cuisine_top3'] = np.nan
    new_row_data['movie_genre_top2'] = np.nan
    new_row_data['movie_genre_top3'] = np.nan
    new_row_data['series_genre_top2'] = np.nan
    new_row_data['series_genre_top3'] = np.nan
    new_row_data['content_lang_top2'] = np.nan
    new_row_data['content_lang_top3'] = np.nan
    new_row_data['ott_top2'] = np.nan
    new_row_data['ott_top3'] = np.nan
    new_row_data['game_genre_top2'] = np.nan
    new_row_data['game_genre_top3'] = np.nan
    new_row_data['gaming_platform_top2'] = np.nan
    new_row_data['gaming_platform_top3'] = np.nan
    new_row_data['social_platform_top2'] = np.nan
    new_row_data['social_platform_top3'] = np.nan
    new_row_data['music_genre_top2'] = np.nan
    new_row_data['music_genre_top3'] = np.nan
    new_row_data['music_lang_top2'] = np.nan
    new_row_data['book_genre_top2'] = np.nan
    new_row_data['book_genre_top3'] = np.nan
    new_row_data['travel_type_top2'] = np.nan
    new_row_data['travel_type_top3'] = np.nan
    new_row_data['hobby_top2'] = np.nan
    new_row_data['club_top2'] = np.nan


    # Convert to DataFrame
    full_df = pd.DataFrame([new_row_data], columns=full_data_columns)

    # Append the new data to the csv
    full_df.to_csv(data_path, mode="a", header=False, index=False)

    # Return the prediction
    if prediction == 1:
        preference = "Team"
    else:
        preference = "Solo"

    return {
        "prediction": preference,
        "prediction_probability": {
            "Solo": prediction_proba[0],
            "Team": prediction_proba[1]
        }
    }
