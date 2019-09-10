import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


df = pd.read_csv('perrin-freres-monthly-champagne-.csv')
print(df.tail())


df.drop(106,axis=0,inplace=True)
print(df.tail())

#df.drop(106,axis=0,inplace=True)
df.columns=['Month','Sale_Per_Month']
print(df.head())

df['Month']=pd.to_datetime(df['Month'])
print(df.head())

df.set_index('Month',inplace=True)
print(df.head())
print(df.plot())
model = sm.tsa.statespace.SARIMAX(df['Sales per month'],order=(1,0,0),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales per month','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+DateOffset(months=x)for x in range(0,24)]
future_datest_df = pd.DataFrame(index=future_dates[1:],columns = df.columns)
print(future_datest_df)
future_df=pd.concat([df,future_datest_df])
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)
future_df[['Sales per month', 'forecast']].plot(figsize=(12, 8))