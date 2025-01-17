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

# A regresszióhoz egy LSTM típusú neurális hálót használtunk. Az LSTM típusú háló egy visszacsatolt neurális háló,
# tehát a bemenete nem csak az aktuáls bemenetet, hanem a korábbi bemeneteket is. Az LSTM hálók abban különböznek a
# többi visszacsatolt hálótól, hogy itt szerették volna szabályozni a háló emlékezését a fejlesztők,  így a háló
# tartalmaz egy belső állapotot, amely memóriaként működik és kevésbéérzékeny a back propagation műveletre.

# A következő függvények segédfüggvények

# Az adatsor felügyelt tanulásra alkalmassá tétele
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# Hogy kisebb számokkal kelljen dolgozni, így az időpillanatko közötti változás lesz az input. Ezt a különbséget
# állítja elő a függvény.
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# Visszalakít a különbségi értékekről.
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# [-1, 1] intervallumon skálázza az értékeket, ahol -1 a legkisebb érték, és 1 a legnagyobb érték.
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


# Skála invertálása
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Háló tanítása
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model


# Predikció
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

tf.keras.backend.clear_session()
# init
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
scaler_dict = dict()

# Az adatsor betöltése. Az adatsort egy dictionary-be töltöttük, így a kulcsok megadásával lehet futtatni a programot
# az egyes kerületekre. Az adatsorunk összesen 365 napnyi adat volt, amelyet 80-20%-ban osztottunk ketté. 80%-ával az
# adatoknak tanítottunk, 20%-ával teszteltük a betanított hálót.
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
    scaler_dict[name], train_scaled[name], test_scaled[name] = scale(train[name], test[name])

# A names tupleban szereplő kerületeket prediktáljuk.
names = ("Union Sq", "JFK Airport", "Governor's Island/Ellis Island/Liberty Island")
for name in names:
    # A modell tanítása.
    lstm_model = fit_lstm(train_scaled[name], 1, 500, 4)

    # A háló jellegéből adódóan a tanítás után végig kell futtatni a tanító adatokat, hogy a háló tudjin emlékezni.
    train_reshaped = train_scaled[name][:, 0].reshape(len(train_scaled[name]), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # A prdeikció és a valós értékek összehasonlítása.
    predictions = list()
    for i in range(len(test_scaled[name])):
        # make one-step forecast
        X, y = test_scaled[name][i, 0:-1], test_scaled[name][i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler_dict[name], X, yhat)
        # invert differencing
        yhat = inverse_difference(data_dictionary[name], yhat, len(test_scaled[name]) + 1 - i)
        # store forecast
        predictions.append(yhat)
        expected = data_dictionary[name][len(train[name]) + i]
        # expected = y
        print("Expected: {0}, Predicted: {1}".format(expected, yhat))

    # Az eredmények megjelenítése
    rmse = sqrt(mean_squared_error(data_dictionary[name][-73:], predictions))
    print('Test RMSE: %.3f' % rmse)
    pyplot.title(name)
    pyplot.xlabel("day")
    pyplot.ylabel("income [$]")
    pyplot.plot(data_dictionary[name][-73:])
    pyplot.plot(predictions)
    pyplot.show()

# Az eredményből az látszik, hogy a valamiféle minákat követő kerületek esetében a háló pontosabban megjósolja a
# bevételt, míg azon kerületek esetében amelyeknél a forgalom is kicsi, valamint véletlenszerű, az eredmény is
# pontatlan.
#
# Egyenletes elosztásra a JFK Airportot hoztuk példának, míg a periodikus eloszlásra példa a Union Square, ahol NY
# egyik legnagyobb termelői piaca. A véletlenszerű eloszlásra hoztuk példaként Governor's Islandet, amelyet például nem
# is lehet megközelíteni műóton személygépjárművel, így taxival sem.
