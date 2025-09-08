import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")

st.title("Project Pairing Dilemma: Team vs. Solo")

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

        # Send the request to the backend
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()

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

            st.success("Prediction successful! Redirecting to the dashboard.")
            st.session_state.page = 'dashboard'
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the backend: {e}")

def show_dashboard():
    st.title("📊 Real-Time Project Preference Dashboard")
    st.write("This dashboard visualizes the project preference data in real-time.")

    if st.button("← Back to Prediction Form"):
        st.session_state.page = 'form'
        st.rerun()

    try:
        # Fetch data from the backend
        summary_response = requests.get("http://127.0.0.1:8000/data-summary")
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        # --- Visualizations ---
        st.header("Data Visualizations")

        # Overall Preference Distribution
        st.subheader("Overall Preference Distribution")
        pref_df = pd.DataFrame(summary_data['preference_distribution'].items(), columns=['Preference', 'Count'])
        st.bar_chart(pref_df.set_index('Preference'))

        # Feature Distributions
        st.subheader("Feature Distributions")

        col1, col2 = st.columns(2)

        with col1:
            intro_df = pd.DataFrame(summary_data['introversion_distribution'].items(), columns=['Score', 'Count']).sort_values('Score')
            fig, ax = plt.subplots()
            sns.barplot(x='Score', y='Count', data=intro_df, ax=ax)
            ax.set_title("Introversion/Extraversion Scores")
            st.pyplot(fig)

        with col2:
            risk_df = pd.DataFrame(summary_data['risk_taking_distribution'].items(), columns=['Score', 'Count']).sort_values('Score')
            fig, ax = plt.subplots()
            sns.barplot(x='Score', y='Count', data=risk_df, ax=ax)
            ax.set_title("Risk-Taking Scores")
            st.pyplot(fig)

        # Recent Submissions
        st.subheader("Recent Submissions")
        recent_df = pd.DataFrame(summary_data['recent_submissions'])
        st.dataframe(recent_df)

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load data visualizations: {e}")
    except KeyError:
        st.error("Received invalid data from the backend. Please ensure the backend is running correctly.")

# Page router
if st.session_state.page == 'form':
    show_form()
elif st.session_state.page == 'dashboard':
    show_dashboard()
