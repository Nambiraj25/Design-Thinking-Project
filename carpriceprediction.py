import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
cars_data=pd.read_csv('cars_sampled.csv')
cars=cars_data.copy()
pd.set_option('display.float_format',lambda x:'%.3f'% x)
pd.set_option('display.max_columns',500)
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=cols,axis=1)
cars.drop_duplicates(keep='first',inplace=True)
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()

price_count=cars['price'].value_counts().sort_index()
power_count=cars['powerPS'].value_counts().sort_index()
cars=cars[
  (cars.yearOfRegistration <=2023)
  &(cars.yearOfRegistration>=1950)
  &(cars.price>=100)
  &(cars.price<=150000)
  &(cars.powerPS<=500)
  &(cars.powerPS>=10)]
cars['monthOfRegistration']/=12
cars['Age']=(2023-cars['yearOfRegistration']+cars['monthOfRegistration'])
cars['Age']=round(cars['Age'],2)
cars=cars.drop(['yearOfRegistration','monthOfRegistration'],axis=1)
fuel_type_counts = cars['fuelType'].value_counts()
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

x=cars_omit.drop(['price'],axis='columns',inplace=False)
y=cars_omit['price']
rf=RandomForestRegressor()
n_estimators=[int(x) for x in np.linspace(start=10,stop=140,num=10)]
max_features=['log2','sqrt']
max_depth=[int(x) for x in np.linspace(start=5,stop=35,num=7)]
min_samples_split=[1,2,5,10,15]
min_samples_leaf=[1,2,4]
bootstrap=[True,False]
param_grid={'n_estimators':n_estimators,
            'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            }
model=RandomizedSearchCV(estimator=rf,param_distributions=param_grid,n_iter=100,cv=5,n_jobs=-1)
model.fit(x,y)
print(model.best_estimator_)
print(model.best_params_)
print(model.best_score_)
model_filename = "random_forest_Regression_model.pkl"
joblib.dump(model, model_filename)
model_dataframe="random_forest_Regression_dataframe.pkl"
x.to_pickle(model_dataframe)
