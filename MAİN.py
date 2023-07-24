import warnings
import itertools
from math import sqrt
from datetime import datetime
from numpy import concatenate
import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional, GRU
#from keras.layers.recurrent import LSTM
from sklearn.utils import shuffle
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("Display.max_columns",None)


# "dataset_combined" klasörüne git
folder_path = "/Users/eyupburakatahanli/Desktop/Tubitak_Air_pol/dataset_combined"
os.chdir(folder_path)
dfs = {}
# tüm CSV dosyalarını al ve oku
for filename in os.listdir():
    if filename.endswith(".csv"):
        # dosya adından DataFrame adını ve dosya adını çıkart
        df_name = filename.split(".")[0].replace(" ", "")
        df = pd.read_csv(filename)
        exec(df_name + " = pd.DataFrame(df)") # DataFrame'i kaydet
        dfs[df_name] = df

print(dfs.keys())


seasons = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
    12: "Winter"
}

def categorize_hour(hour):
    if hour < 8:
        return 'Night'
    elif hour < 16:
        return 'Day'
    else:
        return 'Evening'


for district, df in dfs.items():
    #format dönüşümü
    dfs[district]['ReadTime'] = pd.to_datetime(dfs[district]['ReadTime'], format='%Y-%m-%dT%H:%M:%S')
    #gün ay yıl- olarak ayı değişkenlere atanması
    dfs[district]["Year"] = dfs[district]["ReadTime"].dt.year
    dfs[district]["Day"] = dfs[district]["ReadTime"].dt.day
    dfs[district]["Month"] = dfs[district]["ReadTime"].dt.month
    #saat değişkeni ve buna uygun kategorinin eklenmesi
    dfs[district]["Hour"] = dfs[district]["ReadTime"].dt.hour
    dfs[district]['HourCategory'] = dfs[district]['Hour'].apply(categorize_hour)
    #Mevsim değişkeni ve kategorisinin eklenmesi
    dfs[district]['Season'] = dfs[district]['ReadTime'].dt.month.map(seasons)

    #aqı endex değeri nan olan tüm değişkenleri siliyoruz
    dfs[district] = dfs[district].dropna(subset=['AQIINDEX_AQI'])

    #sonraki çalışmalarda 2023 yılına ait bir değişken olduğunu farkettik bunu siliyoruz
    dfs[district].drop(dfs[district][dfs[district]['Year'] == 2023].index,inplace=True)

dfs['Ümraniye1'] = dfs['Ümraniye1'].dropna(subset=['PM10'])



#İndex'e Alınması
dfs['Ümraniye1'].set_index("ReadTime")

#nan değerleri doldurma
dfs['Ümraniye1']['PM10'].interpolate(method='linear', inplace=True)

data = dfs['Ümraniye1']
fig = plt.figure(figsize=(15,8))
data['PM10'].plot(label='PM10')
plt.legend(loc='best')
plt.title('pm10, Buy', fontsize=14)
plt.show()

from sklearn.preprocessing import MinMaxScaler
values = data['PM10'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)


# yüzde 10
TRAIN_SIZE = 0.90
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Gün Sayıları (training set, test set): " + str((len(train), len(test))))


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

import numpy as np
# Verisetlerimizi Oluşturalım
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)
# Yeni verisetinin şekline bakalım.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)


def fit_model(train_X, train_Y, window_size=1):
    model = Sequential()

    # Modelin tek layerlı şekilde kurulacak.
    model.add(LSTM(48,
                   input_shape=(1, window_size)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",
                  optimizer="adam")

    # 30 epoch yani 30 kere verisetine bakılacak.
    model.fit(train_X,
              train_Y,
              epochs=10,
              batch_size=2,
              verbose=1)

    return (model)


# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)


def predict_and_score(model, X, Y):
    # Şimdi tahminleri 0-1 ile scale edilmiş halinden geri çeviriyoruz.
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    # Rmse değerlerini ölçüyoruz.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)



train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict
# Şimdi ise testleri tahminletiyoruz.

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict
# Plot'u oluşturalım.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Hours")
plt.ylabel("PM10 Value")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()