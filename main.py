from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 1:], train[:, 0]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        return model


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [value] + [x for x in X]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, 0]


def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

df = read_csv('train.csv', header=0, index_col=0)
df_district = df.groupby('Zone')

data_dict = dict()
train = dict()
test = dict()
train_scaled = dict()
test_scaled = dict()
for string, district in df_district:
    data_dict[string] = district.values[:, 3:14]
    train[string] = district.values[:-73, 3:14]
    test[string] = district.values[-73:, 3:14]
    scaler, train_scaled[string], test_scaled[string] = scale(train[string], test[string])

lstm_model = fit_lstm(train_scaled['Flushing'], 1, 50, 10)

predictions = list()
i = 0
for scaled in test_scaled['Flushing']:
    # make one-step forecast
    X, y = scaled[1:], scaled[0]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # # invert differencing
    # yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = test['Flushing'][i, 0]
    i = i + 1
    print("Value: {0}, Expected: {1}".format(expected, yhat))

print('a')
