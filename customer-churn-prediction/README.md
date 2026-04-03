# 1. Customer Churn Prediction
# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost
from xgboost import XGBClassifier

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =========================
# 2. LOAD DATA
# =========================
# Change the file name to your CSV path
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.shape)
print(df.head())
print(df.info())

# =========================
# 3. BASIC CLEANING
# =========================
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop customerID because it is not useful for prediction
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Make target binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df.isna().sum())


# =========================
# 4. EDA
# =========================
# Target distribution
churn_counts = df["Churn"].value_counts()
print(churn_counts)
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Churn by contract type
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract")
plt.xticks(rotation=20)
plt.show()

# Churn by tenure
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.show()

# Churn by monthly charges
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Correlation among numeric variables
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# =========================
# 5. SPLIT FEATURES AND TARGET
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)


# =========================
# 6. PREPROCESSING PIPELINE
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# =========================
# 7. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)


# =========================
# 8. MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# =========================
# 9. TRAIN AND EVALUATE
# =========================
results = []

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if hasattr(clf.named_steps["model"], "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results.append([name, acc, rec, roc])
    
    print("\n=========================")
    print(name)
    print("Accuracy:", acc)
    print("Recall:", rec)
    print("ROC-AUC:", roc)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "ROC_AUC"])
print(results_df.sort_values("ROC_AUC", ascending=False))


# =========================
# 10. ROC CURVE PLOT
# =========================
plt.figure(figsize=(8, 6))

for name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    
    if hasattr(clf.named_steps["model"], "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()


# =========================
# 11. SAVE RESULTS
# =========================
results_df.to_csv("churn_model_results.csv", index=False)



















