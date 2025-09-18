# The Project Pairing Dilemma: Team vs Solo Preference Predictor

This project predicts whether a student prefers working "Solo" or in a "Team" based on a small survey. It features a Streamlit web application for user interaction and a FastAPI backend that serves predictions from a trained machine learning model.

## Project Highlights
- **Interactive Web App**: A user-friendly interface built with Streamlit to input personal preferences and receive a prediction.
- **FastAPI Backend**: A robust backend to handle prediction requests, manage data, and serve model insights.
- **Real-Time Dashboard**: A dynamic dashboard that visualizes the distribution of user preferences and other data features.
- **Automated ML Pipeline**: The Jupyter notebook (`project.ipynb`) handles data cleaning, preprocessing, model training, and serialization.

## Repository Structure
- **`model_notebook/`**:
  - `project.ipynb`: The Jupyter notebook for data exploration, model training, and saving the final pipeline.
- **`the-project-pairing-dilemma-app/`**:
  - **`backend/`**:
    - `main.py`: The FastAPI application that serves the model and data.
    - `model.pkl`: The serialized, pre-trained machine learning model.
    - `data.csv`: The original, static dataset.
    - `newdata.csv`: The growing dataset that includes new submissions from the web app.
  - **`frontend/`**:
    - `app.py`: The Streamlit application for user input and displaying predictions.
- **`requirements.txt`**: A single file listing all dependencies for the project.
- **`README.md`**: This file.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Install Dependencies
Install all required Python libraries by running the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Before running the application, you need to train the model and create the `model.pkl` file.
1.  Navigate to the `model_notebook/` directory.
2.  Run the `project.ipynb` notebook. This will train the model and save the `model.pkl` file in the `the-project-pairing-dilemma-app/backend/` directory.

### 3. Run the Application
The application consists of a backend and a frontend, which need to be run separately.

**A. Start the Backend (FastAPI)**
In your terminal, run the following command to start the backend server:
```bash
uvicorn the-project-pairing-dilemma-app.backend.main:app --reload
```
The backend will be available at `http://127.0.0.1:8000`.

**B. Start the Frontend (Streamlit)**
Open a **new** terminal window and run this command to launch the Streamlit app:
```bash
streamlit run the-project-pairing-dilemma-app/frontend/app.py
```
The application will open in your web browser.

## How It Works
1.  **User Input**: The user fills out a form in the Streamlit app.
2.  **Prediction Request**: The frontend sends the user's data to the `/predict` endpoint of the FastAPI backend.
3.  **Data Storage**: The backend appends the new data to `newdata.csv`.
4.  **Prediction**: The backend uses the loaded `model.pkl` to make a prediction.
5.  **Display Results**: The frontend displays the prediction ("Solo" or "Team") and the associated probability.
6.  **Dashboard**: The app provides a link to a dashboard that visualizes the data from `newdata.csv`, providing real-time insights.

## License
This project is unlicensed. You are free to modify and distribute it.
