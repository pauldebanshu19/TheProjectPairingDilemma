import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

st.title("Project Pairing Dilemma: Team vs. Solo")

# --- Paths and caching helpers ---
FRONTEND_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(FRONTEND_DIR, os.pardir, "backend"))
MODEL_PATH = os.path.join(BACKEND_DIR, "model.pkl")
DATA_PATH = os.path.join(BACKEND_DIR, "newdata.csv")

@st.cache_resource(show_spinner=False)
def load_pipeline(path: str = MODEL_PATH):
    return joblib.load(path)

# Load once
pipeline = load_pipeline()

# Optional: if you set API_BASE_URL, the app will try the HTTP backend first
API_BASE_URL = os.environ.get("API_BASE_URL")  # e.g., https://your-fastapi-host

# --- Local implementations mirroring the FastAPI backend ---
PREDICTION_FEATURES = [
    "introversion_extraversion", "risk_taking", "club_top1", "weekly_hobby_hours"
]

FULL_DATA_COLUMNS = [
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


def predict_local(payload: dict):
    # Build a dataframe with only the features required by the pipeline
    features = {k: payload.get(k) for k in PREDICTION_FEATURES}
    prediction_df = pd.DataFrame([features])

    pred = pipeline.predict(prediction_df)[0]
    proba = pipeline.predict_proba(prediction_df)[0]

    # Create a full row for persistence (mirrors backend behavior)
    new_row = {col: np.nan for col in FULL_DATA_COLUMNS}
    new_row.update(payload)
    new_row['timestamp'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Fill known optional columns to NaN if not present
    for c in [
        'cuisine_top2','cuisine_top3','movie_genre_top2','movie_genre_top3','series_genre_top2','series_genre_top3',
        'content_lang_top2','content_lang_top3','ott_top2','ott_top3','game_genre_top2','game_genre_top3',
        'gaming_platform_top2','gaming_platform_top3','social_platform_top2','social_platform_top3',
        'music_genre_top2','music_genre_top3','music_lang_top2','book_genre_top2','book_genre_top3',
        'travel_type_top2','travel_type_top3','hobby_top2','club_top2']:
        new_row.setdefault(c, np.nan)

    full_df = pd.DataFrame([new_row], columns=FULL_DATA_COLUMNS)
    # Append to CSV (best-effort; in serverless env this is ephemeral)
    try:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        full_df.to_csv(DATA_PATH, mode="a", header=False, index=False)
    except Exception:
        # Non-fatal if file system is read-only
        pass

    preference = "Team" if pred == 1 else "Solo"
    return {
        "prediction": preference,
        "prediction_probability": {"Solo": float(proba[0]), "Team": float(proba[1])}
    }


def data_summary_local():
    # Load current CSV
    if not os.path.exists(DATA_PATH):
        # If file missing, return empty defaults
        return {
            "preference_distribution": {},
            "introversion_distribution": {},
            "risk_taking_distribution": {},
            "recent_submissions": [],
            "accuracy": 0,
            "confusion_matrix": [[0, 0], [0, 0]]
        }

    df = pd.read_csv(DATA_PATH)
    df = df.replace({np.nan: None})

    # Binarize teamwork preference
    df['preference'] = df['teamwork_preference'].apply(lambda x: 'Team' if x is not None and x >= 4 else 'Solo')

    preference_distribution = df['preference'].value_counts().to_dict()
    introversion_distribution = df['introversion_extraversion'].dropna().value_counts().to_dict()
    risk_taking_distribution = df['risk_taking'].dropna().value_counts().to_dict()
    recent_submissions = df.tail(5).to_dict(orient='records')

    # Evaluate accuracy if we have rows with complete features
    eval_df = df.dropna(subset=['teamwork_preference', "introversion_extraversion", "risk_taking", "weekly_hobby_hours", "club_top1"])  # type: ignore
    if not eval_df.empty:
        X = eval_df[["introversion_extraversion", "risk_taking", "weekly_hobby_hours", "club_top1"]]
        y_true = eval_df['teamwork_preference'].apply(lambda x: 1 if x >= 4 else 0)
        y_pred = pipeline.predict(X)
        accuracy = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        accuracy = 0.0
        cm = [[0, 0], [0, 0]]

    return {
        "preference_distribution": preference_distribution,
        "introversion_distribution": introversion_distribution,
        "risk_taking_distribution": risk_taking_distribution,
        "recent_submissions": recent_submissions,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }


# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'form'


def show_form():
    st.write("Please fill out the form below to get a prediction.")

    with st.form(key='prediction_form'):
        st.header("Personal Information")
        age = st.number_input("Age", 15, 100, 20)
        height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)

        st.header("Food Preferences")
        cuisine_top1 = st.text_input("Favorite Cuisine", "Indian")
        spice_tolerance = st.slider("Spice Tolerance", 1, 5, 3)
        dietary_pref = st.selectbox("Dietary Preference", ["Non-Veg", "Veg", "Eggitarian", "Jain", "Vegan"])
        eating_out_per_week = st.number_input("Eating Out Per Week", 0, 10, 2)
        food_budget_per_meal = st.number_input("Food Budget Per Meal", 50, 10000, 200)
        sweet_tooth_level = st.slider("Sweet Tooth Level", 1, 5, 3)
        tea_vs_coffee = st.selectbox("Tea vs. Coffee", ["Both", "Tea", "Coffee", "Neither Tea nor coffee"])

        st.header("Entertainment")
        movie_genre_top1 = st.text_input("Favorite Movie Genre", "Action")
        series_genre_top1 = st.text_input("Favorite Series Genre", "Crime")
        content_lang_top1 = st.text_input("Primary Content Language", "English")
        ott_top1 = st.text_input("Favorite OTT Platform", "Netflix")
        binge_freq_per_week = st.number_input("Binge Frequency Per Week", 0, 7, 2)
        screen_time_hours_per_week = st.number_input("Screen Time (hours per week)", 0, 100, 10)

        st.header("Gaming")
        gaming_days_per_week = st.number_input("Gaming Days Per Week", 0, 7, 1)
        gaming_hours_per_week = st.number_input("Gaming Hours Per Week", 0, 50, 2)
        game_genre_top1 = st.text_input("Favorite Game Genre", "Strategy")
        gaming_platform_top1 = st.text_input("Primary Gaming Platform", "Mobile")
        esports_viewing = st.selectbox("eSports Viewing", ["Sometimes", "Often", "Never"])

        st.header("Social Media")
        social_platform_top1 = st.text_input("Favorite Social Media Platform", "Instagram")
        daily_social_media_minutes = st.number_input("Daily Social Media (minutes)", 0, 1000, 120)
        primary_content_type = st.text_input("Primary Content Type Consumed", "Memes")
        content_creation_freq = st.selectbox("Content Creation Frequency", ["No", "Occasional", "Regular"])

        st.header("Music")
        music_genre_top1 = st.text_input("Favorite Music Genre", "Bollywood")
        listening_hours_per_day = st.number_input("Music Listening (hours per day)", 0, 24, 2)
        music_lang_top1 = st.text_input("Primary Music Language", "Hindi")
        live_concerts_past_year = st.number_input("Live Concerts in Past Year", 0, 20, 1)

        st.header("Reading")
        reads_books = st.selectbox("Do you read books?", ["Sometimes", "Regularly", "No"])
        book_genre_top1 = st.text_input("Favorite Book Genre", "Fiction")
        books_read_past_year = st.number_input("Books Read in Past Year", 0, 100, 5)

        st.header("Lifestyle")
        fashion_spend_per_month = st.number_input("Fashion Spend Per Month", 0, 10000, 500)
        shopping_mode_pref = st.selectbox("Shopping Mode Preference", ["Mixed", "Mostly Offline", "Mostly Online"])
        ethical_shopping_importance = st.slider("Importance of Ethical Shopping", 1, 5, 3)

        st.header("Travel")
        travel_freq_per_year = st.number_input("Travel Frequency Per Year", 0, 50, 2)
        travel_type_top1 = st.text_input("Favorite Travel Type", "Road Trip")
        budget_per_trip = st.number_input("Budget Per Trip", 1000, 100000, 10000)
        travel_planning_pref = st.slider("Travel Planning Preference", 1, 5, 3)

        st.header("Hobbies & Personality")
        hobby_top1 = st.text_input("Primary Hobby", "Coding")
        club_top1 = st.text_input("Primary Club/Activity", "Coding Club")
        weekly_hobby_hours = st.number_input("Weekly Hobby Hours", 0, 100, 10)
        introversion_extraversion = st.slider("Introversion/Extraversion (1=Introvert, 5=Extravert)", 1, 5, 3)
        risk_taking = st.slider("Risk-Taking (1=Low, 5=High)", 1, 5, 3)
        conscientiousness = st.slider("Conscientiousness", 1, 5, 3)
        open_to_new_experiences = st.slider("Openness to New Experiences", 1, 5, 3)
        teamwork_preference = st.slider("Teamwork Preference (1=Solo, 5=Team)", 1, 5, 3)

        submit_button = st.form_submit_button(label='Predict Preference')

    # Predict button logic
    if submit_button:
        # Create the payload
        payload = {
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "cuisine_top1": cuisine_top1,
            "spice_tolerance": spice_tolerance,
            "dietary_pref": dietary_pref,
            "eating_out_per_week": eating_out_per_week,
            "food_budget_per_meal": food_budget_per_meal,
            "sweet_tooth_level": sweet_tooth_level,
            "tea_vs_coffee": tea_vs_coffee,
            "movie_genre_top1": movie_genre_top1,
            "series_genre_top1": series_genre_top1,
            "content_lang_top1": content_lang_top1,
            "ott_top1": ott_top1,
            "binge_freq_per_week": binge_freq_per_week,
            "screen_time_hours_per_week": screen_time_hours_per_week,
            "gaming_days_per_week": gaming_days_per_week,
            "gaming_hours_per_week": gaming_hours_per_week,
            "game_genre_top1": game_genre_top1,
            "gaming_platform_top1": gaming_platform_top1,
            "esports_viewing": esports_viewing,
            "social_platform_top1": social_platform_top1,
            "daily_social_media_minutes": daily_social_media_minutes,
            "primary_content_type": primary_content_type,
            "content_creation_freq": content_creation_freq,
            "music_genre_top1": music_genre_top1,
            "listening_hours_per_day": listening_hours_per_day,
            "music_lang_top1": music_lang_top1,
            "live_concerts_past_year": live_concerts_past_year,
            "reads_books": reads_books,
            "book_genre_top1": book_genre_top1,
            "books_read_past_year": books_read_past_year,
            "fashion_spend_per_month": fashion_spend_per_month,
            "shopping_mode_pref": shopping_mode_pref,
            "ethical_shopping_importance": ethical_shopping_importance,
            "travel_freq_per_year": travel_freq_per_year,
            "travel_type_top1": travel_type_top1,
            "budget_per_trip": budget_per_trip,
            "travel_planning_pref": travel_planning_pref,
            "hobby_top1": hobby_top1,
            "club_top1": club_top1,
            "weekly_hobby_hours": weekly_hobby_hours,
            "introversion_extraversion": introversion_extraversion,
            "risk_taking": risk_taking,
            "conscientiousness": conscientiousness,
            "open_to_new_experiences": open_to_new_experiences,
            "teamwork_preference": teamwork_preference
        }

        # Try HTTP backend if configured, otherwise run locally
        result = None
        if API_BASE_URL:
            try:
                resp = requests.post(f"{API_BASE_URL.rstrip('/')}/predict", json=payload, timeout=15)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                st.warning(f"HTTP backend failed; falling back to local prediction. Details: {e}")

        if result is None:
            try:
                result = predict_local(payload)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # Display the result
        st.subheader("Prediction Result")
        st.write(f"You are more likely to prefer working in a **{result['prediction']}**")

        # Prediction probability gauge
        st.subheader("Prediction Probability")
        prob_team = result['prediction_probability']['Team']

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_team,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probability of Preferring Teamwork"},
            gauge = {
                'axis': {'range': [0, 1]},
                'steps' : [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1], 'color': "gray"}
                ],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prob_team}
            }
        ))
        st.plotly_chart(fig)

        st.success("Prediction completed successfully!")
        st.session_state.page = 'dashboard'
        if st.button("View Dashboard"):
            st.rerun()


def show_dashboard():
    st.title("📊 Real-Time Project Preference Dashboard")
    st.write("This dashboard visualizes the project preference data in real-time.")

    if st.button("← Back to Prediction Form"):
        st.session_state.page = 'form'
        st.rerun()

    summary_data = None

    # Try HTTP backend if configured, else local
    if API_BASE_URL:
        try:
            summary_response = requests.get(f"{API_BASE_URL.rstrip('/')}/data-summary", timeout=15)
            summary_response.raise_for_status()
            summary_data = summary_response.json()
        except Exception as e:
            st.warning(f"HTTP backend failed; falling back to local summary. Details: {e}")

    if summary_data is None:
        try:
            summary_data = data_summary_local()
        except Exception as e:
            st.error(f"Could not load data visualizations: {e}")
            return

    # --- Visualizations ---
    st.header("Data Visualizations")

    # Overall Preference Distribution
    st.subheader("Overall Preference Distribution")
    pref_df = pd.DataFrame(summary_data['preference_distribution'].items(), columns=['Preference', 'Count'])
    if not pref_df.empty:
        st.bar_chart(pref_df.set_index('Preference'))
    else:
        st.info("No data yet. Make a prediction to populate the dashboard.")

    # Feature Distributions
    st.subheader("Feature Distributions")

    col1, col2 = st.columns(2)

    with col1:
        intro_df = pd.DataFrame(summary_data['introversion_distribution'].items(), columns=['Score', 'Count']).sort_values('Score')
        fig, ax = plt.subplots()
        if not intro_df.empty:
            sns.barplot(x='Score', y='Count', data=intro_df, ax=ax)
        ax.set_title("Introversion/Extraversion Scores")
        st.pyplot(fig)

    with col2:
        risk_df = pd.DataFrame(summary_data['risk_taking_distribution'].items(), columns=['Score', 'Count']).sort_values('Score')
        fig, ax = plt.subplots()
        if not risk_df.empty:
            sns.barplot(x='Score', y='Count', data=risk_df, ax=ax)
        ax.set_title("Risk-Taking Scores")
        st.pyplot(fig)

    # Recent Submissions
    st.subheader("Recent Submissions")
    recent_df = pd.DataFrame(summary_data['recent_submissions'])
    if not recent_df.empty:
        st.dataframe(recent_df)
    else:
        st.caption("No submissions yet.")



# Page router
if st.session_state.page == 'form':
    show_form()
elif st.session_state.page == 'dashboard':
    show_dashboard()