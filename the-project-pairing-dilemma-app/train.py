import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# Define paths relative to the script's location
app_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(app_dir, "backend")
model_path = os.path.join(backend_dir, "model.pkl")
# The original dataset is now in the root directory, one level above the-project-pairing-dilemma-app
data_path = os.path.join(app_dir, "..", "data.csv")
# The new data file for the app
new_data_path = os.path.join(backend_dir, "newdata.csv")

# Create backend directory if it doesn't exist
os.makedirs(backend_dir, exist_ok=True)

# Load data
raw = pd.read_csv(data_path)

# --- Data processing and model training logic from the notebook ---

def _normalize(s: str) -> str:
    s = str(s)
    s = re.sub(r"\\s+", " ", s).strip()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()

norm_cols = {c: _normalize(c) for c in raw.columns}
norm_to_orig = {}
for orig, norm in norm_cols.items():
    norm_to_orig.setdefault(norm, orig)
norm_list = list(norm_to_orig.keys())

def find_col_by_keywords(keywords):
    candidates = []
    for col in norm_list:
        ok = True
        for kw in keywords:
            if isinstance(kw, (list, tuple, set)):
                if not any(k in col for k in kw):
                    ok = False
                    break
            else:
                if kw not in col:
                    ok = False
                    break
        if ok:
            candidates.append(col)
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]

intro_col_norm = find_col_by_keywords(["introversion", "extraversion"])
risk_col_norm = find_col_by_keywords(["risk", "taking"])
weekly_col_norm = find_col_by_keywords(["weekly", "hobby", "hours"])
club1_col_norm = find_col_by_keywords([["club"], ["top1", "top_1"]])
teamwork_col_norm = find_col_by_keywords(["teamwork", "preference"])

required_map = {
    "introversion_extraversion": intro_col_norm,
    "risk_taking": risk_col_norm,
    "weekly_hobby_hours": weekly_col_norm,
    "club_top1": club1_col_norm,
    "teamwork_preference": teamwork_col_norm,
}

rename_map = {norm_to_orig[v]: k for k, v in required_map.items()}
df = raw.rename(columns=rename_map)[list(required_map.keys())].copy()

for col in ["introversion_extraversion", "risk_taking", "weekly_hobby_hours", "teamwork_preference"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["teamwork_preference"].isin([1, 2, 4, 5])].copy()
df["teamwork_preference_bin"] = (df["teamwork_preference"] >= 4).astype(int)

features = ["introversion_extraversion", "risk_taking", "weekly_hobby_hours", "club_top1"]
num_features = ["introversion_extraversion", "risk_taking", "weekly_hobby_hours"]
cat_features = ["club_top1"]

y = df["teamwork_preference_bin"].astype(int).values
X = df[features].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

final_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=5000, solver="liblinear"))
])

final_model.fit(X_train, y_train)

# Save the model
joblib.dump(final_model, model_path)
print(f"Model saved to {model_path}")

# Copy the original data.csv to newdata.csv in the backend folder to initialize it
import shutil
shutil.copy(data_path, new_data_path)
print(f"Data copied to {new_data_path}")
