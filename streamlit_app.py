# redeploy fix for plotly error
# streamlit_app.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title(" Time Series Forecasting Dashboard")
st.markdown("### Compare ARIMA, Prophet, and LSTM with Confidence Intervals")

# ------------------ Upload Data ------------------
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.warning("Please upload a CSV file with a date column and at least one numeric column.")
    st.stop()

df = pd.read_csv(file)

# ----------- Auto-detect Date column -----------
date_col = None
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
        date_col = col
        break
    except:
        continue

if date_col is None:
    st.error("No date column found. Please check your CSV!")
    st.stop()

df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)

# ----------- Auto-detect numeric column -----------
numeric_col = None
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_col = col
        break

if numeric_col is None:
    st.error("No numeric column found for forecasting!")
    st.stop()

ts = df[numeric_col]

st.write("### Preview of Data")
st.dataframe(df.head())

# ------------------ Parameters ------------------
horizon = st.sidebar.slider("Forecast Horizon (days)", 10, 100, 30)

# ------------------ ARIMA ------------------
model_arima = ARIMA(ts, order=(5,1,0))
fit_arima = model_arima.fit()
forecast_arima = fit_arima.get_forecast(steps=horizon)
mean_arima = forecast_arima.predicted_mean
conf_int_arima = forecast_arima.conf_int()

# ------------------ Prophet ------------------
prophet_df = ts.reset_index().rename(columns={date_col:'ds', numeric_col:'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=horizon)
forecast_prophet = prophet_model.predict(future)

# ------------------ LSTM ------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(ts.values.reshape(-1,1))

X, y = [], []
for i in range(5, len(scaled)):
    X.append(scaled[i-5:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=5, batch_size=1, verbose=0)

last_5 = scaled[-5:].reshape(1,5,1)
lstm_preds = []
for _ in range(horizon):
    pred = model_lstm.predict(last_5, verbose=0)
    lstm_preds.append(pred[0,0])
    last_5 = np.append(last_5[:,1:,:], pred.reshape(1,1,1), axis=1)
lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1,1))

# ------------------ Plot ------------------
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Actual'))

# ARIMA forecast + interval
future_dates_arima = pd.date_range(ts.index[-1], periods=horizon+1, freq='D')[1:]
fig.add_trace(go.Scatter(x=future_dates_arima, y=mean_arima, mode='lines', name='ARIMA Forecast'))
fig.add_trace(go.Scatter(
    x=list(future_dates_arima)+list(future_dates_arima[::-1]),
    y=list(conf_int_arima.iloc[:,0])+list(conf_int_arima.iloc[:,1][::-1]),
    fill='toself', fillcolor='rgba(0,100,250,0.2)',
    line=dict(color='rgba(255,255,255,0)'), name='ARIMA 95% CI'
))

# Prophet forecast + interval
fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], mode='lines', name='Prophet Forecast'))
fig.add_trace(go.Scatter(
    x=list(forecast_prophet['ds'])+list(forecast_prophet['ds'][::-1]),
    y=list(forecast_prophet['yhat_lower'])+list(forecast_prophet['yhat_upper'][::-1]),
    fill='toself', fillcolor='rgba(250,100,100,0.2)',
    line=dict(color='rgba(255,255,255,0)'), name='Prophet 95% CI'
))

# LSTM forecast
future_dates_lstm = pd.date_range(ts.index[-1], periods=horizon+1, freq='D')[1:]
fig.add_trace(go.Scatter(x=future_dates_lstm, y=lstm_preds.flatten(), mode='lines', name='LSTM Forecast'))

fig.update_layout(title="Forecast Comparison", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(fig, use_container_width=True)
