# The Project Pairing Dilemma: Team vs Solo Preference Predictor

This project predicts whether a student prefers working "Solo" or in a "Team" based on a small survey. It features a Streamlit web application for user interaction and (optionally) a FastAPI backend that serves predictions from a trained machine learning model.

## Project Highlights
- Interactive Web App: A user-friendly interface built with Streamlit to input personal preferences and receive a prediction.
- FastAPI Backend (optional): A backend to handle prediction requests, manage data, and serve model insights. Not required on Streamlit Cloud.
- Real-Time Dashboard: A dynamic dashboard that visualizes the distribution of user preferences and other data features.
- Automated ML Pipeline: The Jupyter notebook (`project.ipynb`) handles data cleaning, preprocessing, model training, and serialization.

## Repository Structure
- `model_notebook/`:
  - `project.ipynb`: The Jupyter notebook for data exploration, model training, and saving the final pipeline.
- `the-project-pairing-dilemma-app/`:
  - `backend/`:
    - `main.py`: The FastAPI application that serves the model and data.
    - `model.pkl`: The serialized, pre-trained machine learning model.
    - `data.csv`: The original, static dataset.
    - `newdata.csv`: The growing dataset that includes new submissions from the web app.
  - `frontend/`:
    - `app.py`: The Streamlit application for user input and displaying predictions.
- `requirements.txt`: A single file listing all dependencies for the project.
- `README.md`: This file.

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1) Install Dependencies
Run:

```cmd
pip install -r requirements.txt
```

### 2) Train or Provide the Model
If you don’t already have `the-project-pairing-dilemma-app\backend\model.pkl`, open and run `model_notebook\project.ipynb` to train and export it.

---

## Running Modes

You can run this project in either of two modes.

### A) Streamlit-only (self-contained) – recommended for Streamlit Cloud
The Streamlit app loads `model.pkl` directly and reads/writes `newdata.csv`. No separate backend process is required.

- Local run:

```cmd
streamlit run the-project-pairing-dilemma-app\frontend\app.py
```

- Streamlit Cloud: Point the app to `the-project-pairing-dilemma-app/frontend/app.py` in this repository. Ensure `the-project-pairing-dilemma-app/backend/model.pkl` exists in the repo. The app will read/write `newdata.csv` in the repo directory (note: this storage is often ephemeral on hosted services).

### B) Split Frontend + Backend (FastAPI)
Use this if you want to keep a separate API service (e.g., deploying FastAPI to Render/Railway/Azure, and the Streamlit app calls that URL).

- Start the backend locally (Windows cmd):

```cmd
cd /d "C:\Users\Mitudru\Documents\ML Project\TheProjectPairing Dilemma\the-project-pairing-dilemma-app\backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- Start the frontend locally:

```cmd
streamlit run the-project-pairing-dilemma-app\frontend\app.py
```

- Deploy the backend to a host (example Render):
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn the-project-pairing-dilemma-app.backend.main:app --host 0.0.0.0 --port $PORT`
  - Ensure CORS allows your Streamlit origin.
  - Then point the frontend to that public API URL (if you keep a backend-calling frontend).

Note: Streamlit Cloud cannot run and expose a second HTTP server (FastAPI) in the same app. Use Mode A on Streamlit Cloud, or deploy FastAPI separately as described above.

---

## Deploy on Streamlit Cloud (single “Main file path”)

When the deployment form asks for “Main file path”, enter:

```
the-project-pairing-dilemma-app/frontend/app.py
```

- No additional “start command” is needed; Streamlit picks this up automatically.
- Optional environment variables:
  - `API_BASE_URL` – If set, the app will call a remote FastAPI (e.g., `https://your-fastapi-host`) for `/predict` and `/data-summary`. If not set or unreachable, the app runs everything locally inside Streamlit using `model.pkl`.

### Recommended repo contents for deployment
- `the-project-pairing-dilemma-app/backend/model.pkl` – required.
- `the-project-pairing-dilemma-app/backend/newdata.csv` – optional; created/appended automatically. Some hosts have read-only or ephemeral filesystems; the app will still work even if writes fail.

---

## How It Works (both modes)
1. User Input: The user fills out a form in the Streamlit app.
2. Prediction: Either the Streamlit app loads the pipeline locally (Mode A) or sends data to FastAPI (Mode B).
3. Data Storage: New submissions are appended to `newdata.csv`.
4. Display Results: The app shows the prediction ("Solo" or "Team") and a probability gauge.
5. Dashboard: Visualizations and basic evaluation metrics computed from `newdata.csv`.

---

## Troubleshooting

- InconsistentVersionWarning when loading `model.pkl`:
  - The model was trained with `scikit-learn==1.7.1` and `requirements.txt` pins to that version. If you retrain the model with a different version, re-pin or rebuild to match.

- Error connecting to backend / HTTPConnectionPool / connection refused:
  - On Streamlit Cloud, use the single-file deployment above. Don’t point the “Main file path” to the FastAPI backend; Streamlit can’t serve a second HTTP server.
  - Locally, ensure FastAPI is running if you set `API_BASE_URL` to a local URL.

- Windows `[WinError 10013]` while starting a server:
  - This usually means the port is blocked by firewall/antivirus/VPN or already in use. For Streamlit-only mode you don’t need to start FastAPI at all; just run the Streamlit command above.

---

## License
This project is unlicensed. You are free to modify and distribute it.
