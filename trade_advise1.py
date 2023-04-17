import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

# Fetch data from Binance
print("Fetching data from Binance...")
try:
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Create DataFrame
print("Creating DataFrame...")
df = pd.DataFrame(
    ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Perform seasonal decomposition using STL-LOESS
print("Performing seasonal decomposition...")
decomposition = seasonal_decompose(df['close'], model='stl', period=24)

# Plot the decomposed components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10, 10))
plt.suptitle("Seasonal Decomposition")
plt.subplot(411)
plt.plot(df['close'], label='Original')
plt.legend(loc='upper left')
plt.ylabel("Price")
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.ylabel("Price")
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.ylabel("Price")
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.ylabel("Price")
plt.show()

# Preprocessing for Prophet
print("Preprocessing for Prophet...")
prophet_df = pd.DataFrame({'ds': df['timestamp'], 'y': df['close']})

# Fit Prophet model
print("Fitting Prophet model...")
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Perform cross-validation for Prophet model
print("Performing cross-validation for Prophet model...")
cv_results = cross_validation(
    prophet_model, initial='30 days', period='1 days', horizon='1 hours')
performance = performance_metrics(cv_results)

print("Prophet model performance:")
print(performance.head())

# Make prediction using Prophet model
print("Making predictions using the Prophet model...")
future = prophet_model.make_future_dataframe(periods=1, freq='H')
forecast = prophet_model.predict(future)
next_hour_prediction = forecast[['ds', 'yhat']].iloc[-1]
print(
    f"Predicted price for {next_hour_prediction['ds']}: ${next_hour_prediction['yhat']:.2f}")

# Preprocessing for LSTM
print("Preprocessing for LSTM...")


def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        x.append(data[i: (i + seq_length), 0])
        y.append(data[i + seq_length, 0])

    return np.array(x), np.array(y)


# Scale the data
print("Scaling the data...")
scaler = MinMaxScaler()
close_prices = df['close'].values.reshape(-1, 1)
scaled_close_prices = scaler.fit_transform(close_prices)

# Create input sequences
print("Creating input sequences...")
seq_length = 60  # Choose an appropriate
seq_length = 60  # Choose an appropriate sequence length
x_lstm, y_lstm = create_sequences(scaled_close_prices, seq_length)

# Reshape input data for LSTM (samples, time steps, features)
print("Reshaping input data for LSTM...")
x_lstm = x_lstm.reshape(x_lstm.shape[0], x_lstm.shape[1], 1)

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(
    x_lstm, y_lstm, test_size=0.2, shuffle=False)

# Define LSTM model architecture
print("Defining LSTM model architecture...")
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, 1)))
model.add(Dense(1))

# Compile the model
print("Compiling the model...")
model.compile(optimizer='adam', loss='mse')

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=10, batch_size=64)

# Evaluate the model
print("Evaluating the model...")
score = model.evaluate(x_test, y_test)
print("Test loss:", score)

# Make predictions using the trained model
print("Making predictions using the trained model...")
future_seq = scaled_close_prices[-seq_length:]
future_seq = future_seq.reshape((1, seq_length, 1))
predicted_scaled = model.predict(future_seq)
predicted = scaler.inverse_transform(predicted_scaled)
prediction_time = df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
print(f"Predicted price for {prediction_time}: ${predicted[0][0]:,.2f}")

# Compare Prophet and LSTM predictions
print("\nComparison of predictions:")
print(
    f"Prophet predicted price for {next_hour_prediction['ds']}: ${next_hour_prediction['yhat']:.2f}")
print(f"LSTM predicted price for {prediction_time}: ${predicted[0][0]:,.2f}")
