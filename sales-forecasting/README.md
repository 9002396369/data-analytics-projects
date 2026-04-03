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
