#multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int),values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0,1,3,4,5]]   #p of index 2 wass very high i.e. 95%
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0,3,4,5]]    #p of index 1 was very high i.e 94%
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0,3,5]]      #p of index 4 was 65% needs to be removed 
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0,3]]       #p should be less than 0.05 hence p of index 5 has to be removed
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
# this shows that RnD spend affects the companys's profit most
