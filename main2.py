from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import tensorflow as tf
import numpy


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

tf.keras.backend.clear_session()
# init dictionaries
train = dict()
test = dict()
data_dictionary = dict()
diff_values = dict()
supervised = dict()
supervised_values = dict()
train = dict()
test = dict()
train_scaled = dict()
test_scaled = dict()

# load dataset
df = read_csv('train.csv', header=0, index_col=0)
df_district = df.groupby('Zone')
for name, data in df_district:
    data_dictionary[name] = data.values[:, 3]
    diff_values[name] = difference(data_dictionary[name], 1)
    # transform data to be supervised learning
    supervised[name] = timeseries_to_supervised(diff_values[name], 1)
    supervised_values[name] = supervised[name].values
    # split data into train and test-sets
    train[name], test[name] = supervised_values[name][0:-73], supervised_values[name][-73:]
    # transform the scale of the data
    scaler, train_scaled[name], test_scaled[name] = scale(train[name], test[name])

# fit the model
lstm_model = fit_lstm(train_scaled["Financial District South"], 1, 1000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled["Financial District South"][:, 0].reshape(len(train_scaled["Financial District South"]), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled["Financial District South"])):
    # make one-step forecast
    X, y = test_scaled["Financial District South"][i, 0:-1], test_scaled["Financial District South"][i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(data_dictionary["Financial District South"], yhat, len(test_scaled["Financial District South"]) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = data_dictionary["Financial District South"][len(train["Financial District South"]) + i + 1]
    print("Expected: {0}, Predicted: {1}".format(expected, yhat))

# report performance
rmse = sqrt(mean_squared_error(data_dictionary["Financial District South"][-73:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(data_dictionary["Financial District South"][-73:])
pyplot.plot(predictions)
pyplot.show()
