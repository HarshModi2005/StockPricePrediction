#Importing the Libraries
import pandas as pd
import numpy as np

import matplotlib. pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras. layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils import plot_model
#Get the Dataset
print('hello')

df=pd.read_csv('../data/HistoricalData.csv')

#Set Target Variable
# output_var = pd.DataFrame(df['Close/Last'])
output_var = df['Close/Last'].tolist()

#Selecting the Features
features = ['Volume','Open', 'High', 'Low']
volume = df['Volume'].tolist()
open = df['Open'].tolist()
high = df['High'].tolist()
low = df['Low'].tolist()

# Remove dollar signs from the each list
# Remove dollar signs from the each list

open = [float(o.replace('$', '')) for o in open]
high = [float(h.replace('$', '')) for h in high]
low = [float(l.replace('$', '')) for l in low]
output_var = [float(f.replace('$','')) for f in output_var]

open = np.array(open).reshape(-1, 1)
high = np.array(high).reshape(-1, 1)
low = np.array(low).reshape(-1, 1)
volume = np.array(volume).reshape(-1, 1)
output_var = np.array(output_var).reshape(-1,1)

#Scaling
scaler = MinMaxScaler()

# feature_transform = scaler.fit_transform(df[features])
# feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
# print(feature_transform.head())
volume_scaled = scaler.fit_transform(volume)
open_scaled = scaler.fit_transform(open)
high_scaled = scaler.fit_transform(high)
low_scaled = scaler.fit_transform(low)
output_var_scaled = scaler.fit_transform(output_var)
feature_transform = [np.array([volume_scaled[x], open_scaled[x], high_scaled[x],low_scaled[x]]) for x in range(len(volume_scaled))]
#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var_scaled[:len(train_index)], output_var_scaled[len(train_index): (len(train_index)+len(test_index))]
    
#Process the data for LSTM
trainX =np.array(X_train)

testX =np.array(X_test)
# Your existing code
trainX = np.array(X_train)
testX = np.array(X_test)

# Reshape trainX and testX
X_train = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
X_test = testX.reshape(testX.shape[0], 1, testX.shape[1])
# print(X_train[69])
# print(y_train[69])
# Your existing code
# Building the LSTM Model
#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='linear', return_sequences=True))  # Reshape input to have three dimensions
lstm.add(LSTM(16, activation='relu', return_sequences=False))  # Reshape input to have three dimensions

lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
# plot_model(lstm, show_shapes=True, show_layer_names=True)
y_pred = lstm.predict(X_test)

#Predicted vs True Adj Close Value – LSTM


y_test = y_test[::-1]
y_pred = y_pred[::-1]
plt.plot(y_test, label='True Value')

plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()