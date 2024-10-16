'''
APM NOTE:
salam, lotfan tamame import haro be balaye code montaghel konid baraye paksazi data, bad az run kardan
dar entehaye code dar ghesmate final report tozih dahid data ha chi bodand, vorodi chi bodand, khoroji chi bodand
hadaf chi bdoe, chanta model train kardin va kodom model behtarin result ro dashte (natije giri)
moafagh bashid



'''




#-------------Import Libs---------------------------
import sklearn
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#-------------loading data-------------------------

data=fetch_california_housing()
x=data.data
y=data.target

kf=KFold(n_splits=5, shuffle=True, random_state=42 )

'''================================LR model================================'''
model=LinearRegression()
my_params={}
gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of LR model is: ', (1+gs.best_score_)*100,'%')

# accuracy of LR model is:  68.2534402635298 %

'''================================KNN model================================'''
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)

model= KNeighborsRegressor()
list_1=[]
for i in range (1,100):
    list_1.append (i)
my_params={'n_neighbors':list_1, 'metric':['minkowski', 'euclidean', 'manhattan'],
           'weights': ['uniform', 'distance']}

gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x_scaled,y)
gs.best_score_
print('accuracy of KNN model is: ', (1+gs.best_score_)*100,'%')

#accuracy of KNN model is:  78.893 %

'''============================DT=========================================='''
model= DecisionTreeRegressor(random_state=42 )
my_params={'max_depth': [5,10,15,20,25,100,120,150],'min_samples_split':[2,5,10,20,30,40,50], 
           'min_samples_leaf':[2,5,10], 'criterion':['poisson']}

gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of DT model is: ', (1+gs.best_score_)*100,'%')

#accuracy of DT model is:  78.65480063406838 %

'''============================RF=========================================='''
model= RandomForestRegressor()
my_params={'n_estimators': [10,15,20,30]}

gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of RF model is: ', (1+gs.best_score_)*100,'%')

#accuracy of RF model is:  81.74651391923146 %


'''============================SVM=========================================='''
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)

model= SVR()
my_params={'kernel': ['linear','poly','rbf'], 'C':[0.001,0.01,1]}

gs= GridSearchCV (model, my_params, cv=kf, n_jobs=-1, scoring='neg_mean_absolute_percentage_error')
gs.fit(x_scaled,y)
gs.best_score_
print('accuracy of SVM model is: ', (1+gs.best_score_)*100,'%')

#accuracy of SVM model is:  76.28500668759332 %

#========FINAL REPORT======================
'''A sample dataset (containing 20,640 sample) from the scikit-learn library is chosen for this project that contains data on California housing prices. 
8 features of the houses are considered as the input variables, including; MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,Latitude, and Longitude.
The price of the houses is considered as the output (target).
We examin 5 ML methods to find the more accurate model for predicting the value of a house in the desired area.
According to our study, the RF model with 81.75% is the best model.'''
