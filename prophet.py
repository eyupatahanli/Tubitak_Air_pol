import pandas as pd
import plotly.io

df = pd.read_csv("/Users/eyupburakatahanli/Desktop/Tubitak_Air_pol/dataset_combined/Ümraniye1.csv")

df['ds'] = pd.to_datetime(df['ReadTime'], format='%Y-%m-%dT%H:%M:%S')
df['y'] = df['AQIINDEX_AQI']

df = df[["ds","y"]]

from prophet import Prophet
model = Prophet()
model.fit(df)

import pandas as pd
from datetime import datetime, timedelta

start_date = datetime(2022, 12, 1, 0, 0, 0)
end_date = start_date + timedelta(days=30)

current_date = start_date
time_intervals = []

while current_date < end_date:
    time_intervals.append(current_date)
    current_date += timedelta(hours=1)

future = pd.DataFrame({'ds': time_intervals})

forecast = model.predict(future)

"""
buraya kadar her şey çok güzel. şimdi son bir ay ı test seti olarak ayarlayıp tahminde bulunalım ve performansımızı ölçelim
"""
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import plotly.io
from datetime import datetime, timedelta


df = pd.read_csv("/Users/eyupburakatahanli/Desktop/Tubitak_Air_pol/dataset_combined/Ümraniye1.csv")
df['ds'] = pd.to_datetime(df['ReadTime'], format='%Y-%m-%dT%H:%M:%S')
df['y'] = df['AQIINDEX_AQI']

df = df[["ds","y"]]

train = df.drop(df.index[-48:])
model = Prophet()
model.fit(train)

from datetime import datetime, timedelta

start_date = datetime(2022, 12,30, 1, 0, 0)
end_date = start_date + timedelta(days=2)

current_date = start_date
time_intervals = []

while current_date < end_date:
    time_intervals.append(current_date)
    current_date += timedelta(hours=1)

future = pd.DataFrame({'ds': time_intervals})
forecast = model.predict(future)

y_true = df['y'][-48:].values
y_pred = forecast['yhat'].values

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(y_true, y_pred)

print('MAE: %.3f' % mae)

mape = mean_absolute_percentage_error(y_true, y_pred)
print('MAPE: %.3f' % mape)


plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


