#proje boyunca işimize yarayacak scriptler ve fonksiyonlar





################ Modelleme #########
import pandas as pd
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import legacy


df = df.set_index("ReadTime") #indexi zaman değişkeni olarak ayarla

values = data['PM10'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)
#veriyi standartlaştırma


TRAIN_SIZE = 0.90
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Saat Sayıları (training set, test set): " + str((len(train), len(test))))
#veriyi train test olarak bölme

def create_dataset(dataset, window_size = 1):
    """

    :param dataset: veri seti
    :param window_size: her bir giriş örneğinin ne kadar zaman adımı içereceğini gösterir
    :return: modele hazır eğitim ve test seti olarak çıktı verir

    fonksiyon belirtilen windows size kadar zaman dilimine bakıp hedef değişkeni tahmin
    etmesi için oluşturulmıştur
    """
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))


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

plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Hours")
plt.ylabel("PM10 Value")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()











model = Sequential() #modeli kurma

model.add(LSTM(64, input_shape=(length_opt,train.shape[1]))) #LSTM katmanını modele ekleme
"""
length_opt, kullanılan zaman adımlarının sayısıdır ve train.shape[1], giriş veri setindeki özellik sayısıdır

"""
#özellik tahmini başına bir nöron
model.add(Dense(train.shape[1],activation = 'relu'))

model.summary() #model özeti

cp = ModelCheckpoint('model1/', save_best_only=True) #modeli eğitirken en iyi performans gösteren ağırlıkları kaydetmek
model.compile(loss='mse', optimizer=legacy.Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#modeli derlemeye yarar. optimize algoritması olarak ADAM seçilmiştir.


early_stop = EarlyStopping(patience =2) #overfittingi önlemek için erken durdurma mekanizmasıdır.
#modelin belirli bir epoch sayısından sonra daha iyi bir performans elde etmeyeceği düşünülerek eğitimi durdurur.

validation_generator= TimeseriesGenerator(test,test,length=length_opt,batch_size=batch_size)
#length: Oluşturulacak örneklerin uzunluğu (örneklerin içerdikleri zaman adımları sayısı)
#batch_size: Üretilen örneklerin batch boyutu (her adımda işlenecek örnek sayısı)

model.fit_generator(generator,callbacks=[early_stop], validation_data= validation_generator,epochs=20)