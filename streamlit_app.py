import streamlit as st

# Title of the application
st.title("Teamwork Preference Prediction App")

# Introduction
st.write("""
This application helps in predicting suitable team pairings based on individual preferences and criteria.
Please fill in the information below to get started!
""")

# Input Section
st.header("Input Your Preferences")

# Collecting user inputs for teamwork preferences
name = st.text_input("Your Name:")
preference = st.selectbox("Select your preferred role:", ["Developer", "Designer", "Project Manager", "QA Tester"])
experience_level = st.slider("Rate your experience level (1 to 10):", min_value=1, max_value=10)

# Button to submit preferences
if st.button("Predict Team Pairing"):
    st.write(f"Thank you, {name}! Based on your role as a {preference} with an experience level of {experience_level}, we will find the best pairing for you.")
    
    # Here you would typically call a function to perform the prediction
    # For demonstration, we will display a mock result
    st.success("Recommended Pairing: Alice (Developer, Experience Level: 9)")

# Conclusion section
st.header("Conclusion")
st.write("""
This application aims to enhance teamwork by aligning individuals with compatible preferences and skills.
""")
