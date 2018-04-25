import pandas as pd
import numpy as np
import pysal as ps
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

import os
os.chdir('C:/Users/husiy/PyProgram/Spatial_Analysis')

lst=pd.read_csv('data/listings.csv/listings.csv')
lst['price']=lst['price'].apply(lambda x: float(x.strip('$').replace(',', '')))
vars=['host_listings_count', 'bathrooms', 'bedrooms']
y=['price']
types=pd.get_dummies(lst['property_type'])
aves=lst[vars+y]
aves=aves.join(types[['Apartment','House','Bed & Breakfast']]).dropna()

y=aves['price']
aves=aves.drop(columns=['price'],axis=1)
#print aves

mmscaler = MinMaxScaler()

x=pd.DataFrame(mmscaler.fit_transform(aves),index=aves.index,columns=aves.columns)
#print x
#print y
y = np.log(y+ 0.000001)

w=ps.knnW_from_array(lst.loc[x.index,['longitude', 'latitude']].values)
w.transform = 'R'

m1 = ps.spreg.OLS(y.values[:, None], x.values,w=w, spat_diag=True,name_x=x.columns.tolist(), name_y='ln(price)')
#print m1.summary

w2=ps.knnW_from_array(lst.loc[x.index,['longitude', 'latitude']].values)
x_w = x.assign(w_x=ps.lag_spatial(w2, x['host_listings_count'].values))

m2 = ps.spreg.OLS(y.values[:, None], x_w.values,w=w, spat_diag=True,name_x=x_w.columns.tolist(), name_y='ln(price)')
#print m2.summary


m3 = ps.spreg.GM_Lag(y.values[:, None], x.values,w=w, spat_diag=True,name_x=x.columns.tolist(), name_y='ln(price)')
#print m3.summary

mses = pd.Series({'OLS': mse(y, m1.predy.flatten()),'X-Lag OLS': mse(y, m2.predy.flatten()),'Y-Lag': mse(y, m3.predy_e)})
print mses.sort_values()