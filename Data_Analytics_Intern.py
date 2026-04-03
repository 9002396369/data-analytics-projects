# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:03:17 2026

@author: harsh
"""


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



















# 2. Sales Forecasting
# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Time series
from statsmodels.tsa.seasonal import seasonal_decompose, SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Prophet
from prophet import Prophet

# LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)




# =========================
# 2. LOAD AND PREPARE DATA
# =========================
# Replace with your file path
df = pd.read_csv("SampleSuperstore.csv")  # Or 'Superstore.xls' with pd.read_excel

# Convert Order Date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Aggregate monthly sales
monthly_sales = df.groupby('Order Date')['Sales'].sum().resample('M').sum()

print(monthly_sales.head())
print(monthly_sales.shape)



# =========================
# 3. EDA & DECOMPOSITION
# =========================
# Plot time series
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.ylabel("Sales")
plt.show()

# Seasonal decomposition (additive model)
decomp = seasonal_decompose(monthly_sales, model='additive', period=12)
decomp.plot()
plt.tight_layout()
plt.show()

# Check stationarity (optional)
from statsmodels.tsa.stattools import adfuller
result = adfuller(monthly_sales)
print("ADF Statistic:", result[0])
print("p-value:", result[1])



# =========================
# 4. SPLIT TRAIN/TEST (last 12 months as test)
# =========================
train = monthly_sales[:-12]
test = monthly_sales[-12:]

print(f"Train: {len(train)}, Test: {len(test)}")



# =========================
# 5. ARIMA/SARIMA MODEL
# =========================
# SARIMA for seasonality (p,d,q)(P,D,Q,s) - tune via auto_arima if needed
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit(disp=False)

# Forecast
sarima_forecast = sarima_fit.get_forecast(steps=12)
sarima_pred = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

# Plot
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, sarima_pred, label="SARIMA Forecast")
plt.fill_between(test.index, sarima_ci.iloc[:,0], sarima_ci.iloc[:,1], alpha=0.3)
plt.title("SARIMA Forecast")
plt.legend()
plt.show()


# =========================
# 6. PROPHET MODEL
# =========================
# Prepare for Prophet
prophet_df = train.reset_index()
prophet_df.columns = ['ds', 'y']

# Fit Prophet (handles trend + seasonality automatically)
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(prophet_df)

# Future dataframe
future = prophet_model.make_future_dataframe(periods=12, freq='M')
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast['yhat'][-12:]

# Plot
fig = prophet_model.plot(prophet_forecast)
plt.plot(test.index, test.values, 'r-', label="Actual")
plt.legend()
plt.title("Prophet Forecast")
plt.show()



# =========================
# 7. LSTM MODEL
# =========================
# Scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1,1))
test_scaled = scaler.transform(test.values.reshape(-1,1))

# Create sequences (window=12 months)
def create_sequences(data, window=12):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(np.concatenate([train_scaled, test_scaled]))

# Build LSTM
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Train
early_stop = EarlyStopping(monitor='loss', patience=5)
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

# Predict
lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()

# Plot
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, lstm_pred, label="LSTM Forecast")
plt.title("LSTM Forecast")
plt.legend()
plt.show()



# =========================
# 8. EVALUATION & COMPARISON
# =========================
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mae, rmse, mape

metrics = {}
for name, pred in [("SARIMA", sarima_pred), ("Prophet", prophet_pred), ("LSTM", lstm_pred)]:
    mae, rmse, mape = evaluate(test.values, pred)
    metrics[name] = [mae, rmse, mape]
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

results_df = pd.DataFrame(metrics).T
results_df.columns = ["MAE", "RMSE", "MAPE"]
print(results_df.sort_values("MAPE"))




# =========================
# 9. FINAL VISUALIZATION: ALL MODELS
# =========================
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train", linewidth=2)
plt.plot(test.index, test, label="Actual", linewidth=2)
plt.plot(test.index, sarima_pred, label="SARIMA")
plt.plot(test.index, prophet_pred, label="Prophet")
plt.plot(test.index, lstm_pred, label="LSTM")
plt.title("Forecast vs Actual Sales (All Models)")
plt.ylabel("Sales")
plt.legend()
plt.show()
