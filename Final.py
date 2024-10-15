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
#-------------loading data-------------------------

data=fetch_california_housing()
x=data.data
y=data.target


from sklearn.model_selection import KFold
kf=KFold(n_splits=5, shuffle=True, random_state=42 )

'''================================LR model================================'''
from sklearn.linear_model import LinearRegression
model=LinearRegression()
my_params={}
from sklearn.model_selection import GridSearchCV
gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of LR model is: ', (1+gs.best_score_)*100,'%')

# accuracy of LR model is:  68.2534402635298 %

'''================================KNN model================================'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
from sklearn.neighbors import KNeighborsRegressor
model= KNeighborsRegressor()
list_1=[]
for i in range (1,100):
    list_1.append (i)
my_params={'n_neighbors':list_1, 'metric':['minkowski', 'euclidean', 'manhattan'],
           'weights': ['uniform', 'distance']}
from sklearn.model_selection import GridSearchCV
gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x_scaled,y)
gs.best_score_
print('accuracy of KNN model is: ', (1+gs.best_score_)*100,'%')

#accuracy of KNN model is:  78.893 %

'''============================DT=========================================='''

from sklearn.tree import DecisionTreeRegressor
model= DecisionTreeRegressor(random_state=42 )
my_params={'max_depth': [5,10,15,20,25,100,120,150],'min_samples_split':[2,5,10,20,30,40,50], 
           'min_samples_leaf':[2,5,10], 'criterion':['poisson']}
from sklearn.model_selection import GridSearchCV
gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of DT model is: ', (1+gs.best_score_)*100,'%')

#accuracy of DT model is:  78.65480063406838 %

'''============================RF=========================================='''
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
my_params={'n_estimators': [10,15,20,30]}
from sklearn.model_selection import GridSearchCV
gs= GridSearchCV (model, my_params, cv=kf,scoring='neg_mean_absolute_percentage_error')
gs.fit(x,y)
gs.best_score_
print('accuracy of RF model is: ', (1+gs.best_score_)*100,'%')

#accuracy of RF model is:  81.74651391923146 %

x=np.arange (1,30).reshape(-1,1)
y=gs.predict(x)
plt.scatter(x,y)
plt.xlabel(' House age(s)')
plt.ylabel('House value ($)')
plt.show 


'''============================SVM=========================================='''

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)

from sklearn.svm import SVR
model= SVR()
my_params={'kernel': ['linear','poly','rbf'], 'C':[0.001,0.01,1]}
from sklearn.model_selection import GridSearchCV
gs= GridSearchCV (model, my_params, cv=kf, n_jobs=-1, scoring='neg_mean_absolute_percentage_error')
gs.fit(x_scaled,y)
gs.best_score_
print('accuracy of SVM model is: ', (1+gs.best_score_)*100,'%')

#accuracy of SVM model is:  76.28500668759332 %



#========FINAL REPORT======================
