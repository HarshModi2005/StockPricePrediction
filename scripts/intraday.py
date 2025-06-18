import pandas as pd
import numpy
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv('../data/data.csv')
stockdata = pd.read_csv('../data/stockdata.csv')
date = data['Date'].tolist()
datesnp = stockdata['date'].tolist()
print(date[-1])
print(datesnp[-1])
opendatasnp = stockdata['open'].tolist()
closedatasnp = stockdata['close'].tolist()
opendata = data['Date'].tolist()
closedata = data['Low'].tolist()
change =[]
changesnp =[]
for i in range(0,365):
    change.append((closedata[i]-opendata[i])/opendata[i])
    changesnp.append((closedatasnp[i]-opendatasnp[i])/opendatasnp[i])

# negative =0 

# plt.plot(changesnp, color='pink')
#change the color of one plot
plt.plot(changesnp, color='pink')
plt.plot(change, color='blue')

# plt.axhline(y=0, color='r', linestyle='-')
plt.show()


