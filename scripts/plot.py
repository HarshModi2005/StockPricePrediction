import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_csv('../data/HistoricalData.csv')
print(data.head())
high = data['High'].tolist()
date = data['Date'].tolist()
high_float= []
for i in high:
    i= i[1:]
    high_float.append(float(i.replace('$', '')))
import matplotlib.pyplot as plt
#Scale high using this

scaler = MinMaxScaler()
high_float = np.array(high_float).reshape(-1, 1)
high_float = high_float[::-1]
high_scaled = scaler.fit_transform(high_float)


plt.plot(high_scaled)

plt.show()