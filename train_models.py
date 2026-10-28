# train_models.py
# Trains five classifiers on the hourly bike-sharing data and saves them to model/.
# Target: high_demand = 1 if hourly rental count is above the median, else 0.
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

DATA_PATH = "data/hour.csv"
TARGET_COL = "high_demand"
RANDOM_STATE = 42

raw = pd.read_csv(DATA_PATH)

# Build the classification target from the regression count.
median_cnt = raw["cnt"].median()
df = raw.copy()
df[TARGET_COL] = (df["cnt"] > median_cnt).astype(int)

# Drop identifiers, date string, and columns that leak the target
# (casual + registered = cnt, so keeping them would make the task trivial).
df = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print(f"Features: {X.shape[1]}, rows: {X.shape[0]}, median cnt: {median_cnt}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Held-out test split, uploaded in the Streamlit app
test_df = X_test.copy()
test_df[TARGET_COL] = y_test
test_df.to_csv("test_data.csv", index=False)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True, random_state=RANDOM_STATE),
}

Path("model").mkdir(exist_ok=True)
for name, clf in models.items():
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, f"model/{name}.joblib")
    print(f"Saved {name}")
