# Austin_Airbnb
Inspired by http://darribas.org/gds_scipy16/, try different scalers and models for spatial analysis

Comparing to Regression models mentioned in http://darribas.org/gds_scipy16/, this project makes feature selection based on Cluster Analysis and using scaled variables to build the models. Again, Spatially Lagged Endogenous Regressors has the best performance, however, with feature selection and max/min scaler processing, the Mean Squared Error (MSE) has improved slightly, which shows the importance of features processing work. 

Results comparison are shown as below:

This project:
Y-Lag:        0.526164,
X-Lag OLS:    0.527902,
OLS:          0.528823,
where Y-Lag stands for spatially lagged endogenous regressors; X-Lag OLS stands for spatially lagged exogenous regressors; OLS stands for baseline (nonspatial) regression.

result on http://darribas.org/gds_scipy16/ipynb_md/08_spatial_regression.html:
Lag:      0.531327,
OLS+W:    0.532402,
OLS:      0.532545,
where Lag stands for spatially lagged endogenous regressors; OLS+W stands for spatially lagged exogenous regressors; OLS stands for baseline (nonspatial) regression.
