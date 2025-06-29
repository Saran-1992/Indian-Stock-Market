import yfinance               as yf     #Downloads historical stock data from Yahoo Finance.
import seaborn                as sns    #Visualizes data with attractive statistical graphics, such as heatmaps.
import numpy                  as np     #Handles numerical operations and array manipulations.
import matplotlib.pyplot      as plt    #Creates static and interactive visualizations for data.

from keras.layers                    import Dropout, LSTM, Dense                              #Constructs neural network layers for building LSTM models.
from keras.models                    import Sequential                                        #Defines a linear stack of layers for creating neural networks.
from sklearn.model_selection         import TimeSeriesSplit                                   #Splits time series data for model validation while preserving temporal order.
from sklearn.metrics                 import mean_squared_error, mean_absolute_error, r2_score #Splits time series data for model validation while preserving temporal order.
from sklearn.preprocessing           import MinMaxScaler                                      #Scales data to a specified range for better model performance.
from flask                           import Flask, request, jsonify                           #Develops a web application and API to serve the LSTM model for predictions.

# Download historical data for a list of stocks
stock_symbols     = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
stock_data        = yf.download(stock_symbols, start='2010-01-01', end='2024-01-01')

# Fill missing values
stock_data.fillna(method='ffill', inplace=True)
stock_data.dropna(inplace=True)

# Feature engineering: Adding indicators (SMA, RSI, Bollinger Bands)
stock_data['SMA_50']     = stock_data['Close']['RELIANCE.NS'].rolling(window=50).mean()
stock_data['SMA_200']    = stock_data['Close']['RELIANCE.NS'].rolling(window=200).mean()

# Calculate RSI
delta                = stock_data['Close']['RELIANCE.NS'].diff(1)
gain                 = delta.where(delta > 0, 0)
loss                 = -delta.where(delta < 0, 0)
avg_gain             = gain.rolling(window=14).mean()
avg_loss             = loss.rolling(window=14).mean()
rs                   = avg_gain / avg_loss
stock_data['RSI']    = 100 - (100 / (1 + rs))

# Bollinger Bands
stock_data['Rolling_Mean']      = stock_data['Close']['RELIANCE.NS'].rolling(window=20).mean()
stock_data['Bollinger_High']    = stock_data['Rolling_Mean'] + 2 * stock_data['Close']['RELIANCE.NS'].rolling(window=20).std()
stock_data['Bollinger_Low']     = stock_data['Rolling_Mean'] - 2 * stock_data['Close']['RELIANCE.NS'].rolling(window=20).std()

# Drop any remaining NaNs
stock_data.dropna(inplace=True)

# Prepare data for LSTM (using 30 timesteps)
window_size       = 30
X_train, y_train  = [], []

for i in range(window_size, len(stock_data)):
    X_train.append(stock_data[['SMA_50', 'SMA_200', 'RSI']].iloc[i-window_size:i].values)
    y_train.append(stock_data['Close']['RELIANCE.NS'].iloc[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled= scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[2]).reshape(X_train.shape))
y_train_scaled= scaler_y.fit_transform(y_train.reshape(-1, 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_train_scaled):
    X_train_cv, X_test_cv = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_cv, y_test_cv = y_train_scaled[train_index], y_train_scaled[test_index]

    y_pred    = model.predict(X_test_cv)
    rmse      = mean_squared_error(y_test_cv, y_pred, squared=False)
    mae       = mean_absolute_error(y_test_cv, y_pred)
    r2        = r2_score(y_test_cv, y_pred)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R-squared: {r2}')

# Set up Flask API for model prediction
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data                   = request.json['features']
    scaled_data            = scaler_X.transform(np.array(data).reshape(1, 1, -1))
    prediction             = model.predict(scaled_data)
    original_prediction    = scaler_y.inverse_transform(prediction.reshape(-1, 1))
    return jsonify({'prediction': original_prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

"""FINISHED"""