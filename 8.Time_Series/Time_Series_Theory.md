# ⏳ Time Series Analysis

Time Series Analysis involves analyzing data points collected or recorded at specific time intervals. It is fundamental in various fields like finance, economics, weather forecasting, and engineering, where understanding past trends and predicting future values is crucial.

---

## 📖 What is Time Series Data?

A **Time Series** is a sequence of data points ordered by time. Time series data can be recorded at various frequencies, such as:

- **Hourly** (e.g., electricity demand).
- **Daily** (e.g., stock prices).
- **Monthly** (e.g., sales data).
- **Yearly** (e.g., population growth).

Each observation is a pair: a timestamp and a value. This time-dependence allows time series analysis to capture patterns, trends, and seasonality over time.

---

## 🌟 Key Components of Time Series Data

Time series data can typically be decomposed into four main components:

1. **Trend**: The overall direction or pattern of the data over a long period (e.g., an upward or downward slope).
2. **Seasonality**: Regular, repeating patterns over fixed periods (e.g., higher sales every December).
3. **Cyclic Patterns**: Fluctuations that are not as regular as seasonality but still exhibit periodic behavior (e.g., economic cycles).
4. **Noise**: Random variations that do not follow any pattern.

---

## 🧠 Approaches in Time Series Analysis

There are two main types of methods used in time series analysis:

### 1. **Classical Statistical Methods**

These methods model the time series as a combination of statistical functions based on its components:

- **Autoregressive (AR)**: Relies on previous values to predict future values.
- **Moving Average (MA)**: Uses past forecast errors for predictions.
- **ARIMA** (Autoregressive Integrated Moving Average): A combination of AR and MA, designed for non-stationary data.

**ARIMA Model**: The ARIMA model is defined by three parameters: **(p, d, q)**.
  - **p**: The number of lag observations in the model (AR).
  - **d**: The number of times the data needs differencing to make it stationary (I).
  - **q**: The size of the moving average window (MA).

#### Example Code (ARIMA)
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model
model = ARIMA(time_series_data, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

### 2. **Machine Learning and Deep Learning Models**

With larger datasets and more complex relationships, machine learning models can capture intricate patterns and dependencies in time series data.

- **Long Short-Term Memory (LSTM)**: A type of Recurrent Neural Network (RNN) that is highly effective in learning time-dependent patterns.
- **Prophet**: Developed by Facebook, it’s a time series forecasting model with strong handling of seasonality and holiday effects.
- **Transformers**: In recent years, transformers have gained traction for time series forecasting due to their efficiency in capturing long-term dependencies.

#### Example Code (LSTM)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16)
```

---

## 📏 Key Metrics for Evaluating Time Series Models

To measure the accuracy of time series models, several metrics are widely used:

1. **Mean Absolute Error (MAE)**:
   - Measures the average magnitude of errors.
     ```math
     \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
     ```

2. **Root Mean Squared Error (RMSE)**:
   - Emphasizes larger errors, as it squares the differences.
     ```math
     \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
     ```

3. **Mean Absolute Percentage Error (MAPE)**:
   - Measures error as a percentage of the observed values.
     ```math
     \text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
     ```

These metrics help assess how well a model’s predictions align with actual values, supporting fine-tuning and optimization.

---

## 🔥 Applications of Time Series Analysis

Time Series Analysis is widely used in numerous domains:

- **Finance**: Stock price prediction, asset management, and economic forecasting.
- **Weather Forecasting**: Predicting temperature, rainfall, and other climate variables.
- **Healthcare**: Monitoring patient vitals over time for anomaly detection.
- **Supply Chain**: Demand forecasting and inventory management.

---

## 📐 Implementing a Simple ARIMA Model for Forecasting

Here’s a step-by-step example using the ARIMA model to forecast future values based on historical data.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load time series data
data = pd.read_csv('time_series_data.csv')
time_series_data = data['value_column']

# Fit ARIMA model
model = ARIMA(time_series_data, order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 10 values
forecast = model_fit.forecast(steps=10)
print("Forecasted Values:", forecast)
```
