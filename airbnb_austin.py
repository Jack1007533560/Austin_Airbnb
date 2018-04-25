import pandas as pd
import numpy as np
import pysal as ps
import geopandas as gpd
import seaborn as sns
from sklearn import cluster
from sklearn.preprocessing import scale,MinMaxScaler
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from shapely.geometry import Point
import fiona
from matplotlib import colors

import os
os.chdir('C:/Users/husiy/PyProgram/Spatial_Analysis')


lst=pd.read_csv('data/listings.csv/listings.csv')
lst['price']=lst['price'].apply(lambda x: float(x.strip('$').replace(',', '')))
#dy=pd.concat([lst['zipcode'],lst['price']],axis=1).dropna()
#dy['zipcode']=dy['zipcode'].apply(lambda x: str(int(x)))
#print dy

vars=['host_listings_count', 'bathrooms', 'bedrooms', 'beds', 'guests_included']
y=['price']

aves=lst.groupby('zipcode')[vars].mean()
aves_y=lst.groupby('zipcode')[y].mean()
#print aves.info()
#print 'price group', aves_y.info()

types=pd.get_dummies(lst['property_type'])
prop_types=types.join(lst['zipcode']).groupby(['zipcode']).sum()
prop_types_pct = (prop_types * 100.).div(prop_types.sum(axis=1), axis=0)
#print prop_types_pct
mmscaler = MinMaxScaler()
#print mmscaler.fit_transform(prop_types_pct)
#print scale(prop_types_pct)

aves_vx=aves

aves=aves.join(prop_types_pct)

#print mmscaler.fit_transform(aves)

db=pd.DataFrame(mmscaler.fit_transform(aves),index=aves.index,columns=aves.columns).rename(lambda x: str(int(x)))
db_y=pd.DataFrame(aves_y).rename(lambda x: str(int(x)))
db_y.columns=['average_price']
#print db_y


zc=gpd.read_file('data/Zipcodes.geojson')
zdb=zc[['geometry', 'zipcode', 'name']].join(db,on='zipcode').dropna()

zdb_y=zc[['geometry', 'zipcode', 'name']].join(db_y,on='zipcode').dropna()

f, ax = plt.subplots(1, figsize=(9, 9))
zdb_y.plot(column='average_price', categorical=False, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='black', legend=True)
plt.title('Average Airbnb Price by Zipcode in Austin')
ax.set_axis_off()
plt.show()


km5 = cluster.KMeans(n_clusters=5)
km5cls = km5.fit(zdb.drop(['geometry', 'name'], axis=1).values)

f, ax = plt.subplots(1, figsize=(9, 9))
zdb.assign(cl=km5cls.labels_).plot(column='cl',categorical=True,cmap='Accent',legend=True,linewidth=0.1,edgecolor='white',ax=ax)
ax.set_axis_off()
plt.title('5 Clusters of Airbnb in Austin')
plt.show()


aves_mmvx=mmscaler.fit_transform(aves_vx)
aves_mmvx=pd.DataFrame(aves_mmvx,index=aves_vx.index,columns=aves_vx.columns)
cl_pcts = aves_mmvx.rename(lambda x: str(int(x))).reindex(zdb['zipcode']).assign(CLASS=km5cls.labels_).groupby('CLASS').mean()
f, ax = plt.subplots(1, figsize=(16,8))
cl_pcts.plot(kind='barh',stacked=True,ax=ax,cmap='tab20', linewidth=0)
ax.legend(ncol=1, loc='upper right')
plt.title('Distribution Of Features Other Than Property Type In Different Clusters')
plt.show()


cl_pcts2 = prop_types_pct.rename(lambda x: str(int(x))).reindex(zdb['zipcode']).assign(CLASS=km5cls.labels_).groupby('CLASS').mean()
f,ax = plt.subplots(1, figsize=(18,9))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
cl_pcts2.plot(kind='barh',stacked=True,ax=ax,cmap='tab20', linewidth=0)
ax.legend(ncol=1,loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Distribution Of Property Type In Different Clusters')
plt.show()



#geometry = [Point(xy) for xy in zip(lst.longitude, lst.latitude)]
#crs = {'init': 'epsg:3081'} #http://www.spatialreference.org/ref/epsg/3081/
#geo_df = GeoDataFrame(lst, crs=crs, geometry=geometry)
#geo_df.to_file(driver='ESRI Shapefile', filename='lst.shp')


#zdb_y.to_file(driver='ESRI Shapefile', filename='lst_y.shp')


w_y = ps.queen_from_shapefile('lst_y.shp')
w_y.transform = 'r'

price_lag = ps.lag_spatial(w_y, zdb_y.average_price)
price_lag_q5 = ps.Quantiles(price_lag, k=5)

f, ax = plt.subplots(1, figsize=(9, 9))
zdb_y.assign(PRICE_LAG=price_lag_q5.yb).plot(column='PRICE_LAG', categorical=True, k=5, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='white', legend=True)
ax.set_axis_off()
plt.title('Airbnb Price Lag in Austin')
plt.show()


I_price = ps.Moran(zdb_y.average_price.values, w_y)
print 'Global Moran Index:',I_price.I,'Significance:',I_price.p_sim

LMo_price = ps.Moran_Local(zdb_y.average_price.values,w_y,permutations=9999)
#print LMo_price.Is[:], LMo_price.p_sim[:]


average_price = zdb_y.average_price.values

sigs = average_price[LMo_price.p_sim <= 0.05]
W_sigs = price_lag[LMo_price.p_sim <= 0.05]
insigs = average_price[LMo_price.p_sim > 0.05]
W_insigs = price_lag[LMo_price.p_sim > 0.05]



b,a = np.polyfit(average_price,price_lag,1)

f, ax = plt.subplots(1, figsize=(9, 9))
plt.plot(sigs, W_sigs, '.', color='firebrick')
plt.plot(insigs, W_insigs, '.k', alpha=.2)
plt.vlines(average_price.mean(), price_lag.min(), price_lag.max(), linestyle='--')
plt.hlines(price_lag.mean(), average_price.min(), average_price.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(average_price, a + b*average_price, 'r')
plt.text(s='$I = %.3f$' %I_price.I,x=300,y=135,fontsize=18)
plt.text(s='$Significance = %.3f$' %I_price.p_sim,x=300,y=100,fontsize=18)
plt.title('Moran Scatterplot For Airbnb Price In Austin')
plt.ylabel('Spatial Lag of Average Price')
plt.xlabel('Average Price')
plt.show()


sig = LMo_price.p_sim < 0.05
hh=LMo_price.q==1 * sig
ll=LMo_price.q==3 * sig

hcmap = colors.ListedColormap(['blue','grey','red'])
f, ax = plt.subplots(1, figsize=(9, 9))
zdb_y.assign(cl=ll*(-1)+hh*1).plot(column='cl',categorical=True,k=2, cmap=hcmap,linewidth=0.1, ax=ax,edgecolor='black', legend=True)
ax.set_axis_off()
plt.title('Saptial Auto-Correlation HH (1) And LL (-1) Distribution Of Airbnb Average Price In Austin')
plt.show()


#w=ps.knnW_from_array(lst.loc[yxs.index,['longitude','latitude']].values)
#w.transform='R'


