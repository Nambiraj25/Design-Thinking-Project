import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_absolute_error,mean_squared_error
sns.set(rc={'figure.figsize':(11.7,8.27)})
cars_data=pd.read_csv('cars_sampled.csv')
cars=cars_data.copy()
print(cars.info())
pd.set_option('display.float_format',lambda x:'%.3f'% x)
pd.set_option('display.max_columns',500)
print(cars.describe())
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=cols,axis=1)
cars.drop_duplicates(keep='first',inplace=True)
print(cars.isnull().sum())
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
print(sum(cars['yearOfRegistration']>2023))
print(sum(cars['yearOfRegistration']<1950))
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
sns.regplot(x="Age",y="price",scatter=True,fit_reg=False,data=cars)
plt.show()
sns.regplot(x="powerPS",y="price",scatter=True,fit_reg=False,data=cars)
plt.show()
cars['fueltype_grouped'] = cars['fuelType'].apply(lambda x: 'petrol' if x == 'petrol' else 'diesel' if x=='diesel' else 'others')
fuel_type_counts=cars['fueltype_grouped'].value_counts()
print(fuel_type_counts)
plt.figure(figsize=(8, 8))
plt.pie(fuel_type_counts, labels=fuel_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Fuel Type Distribution')
plt.axis('equal') 
plt.show()
print(cars['vehicleType'].value_counts())
print(pd.crosstab(cars['vehicleType'],columns='count',normalize=True))
print(cars['gearbox'].value_counts())
print(pd.crosstab(cars['gearbox'],columns='count',normalize=True))
plt.figure(figsize=(10, 10))
sns.set(style="whitegrid")
sns.countplot(x="gearbox", data=cars, palette="Set2")
plt.title("Distribution of Gearbox Types")
plt.xlabel("Gearbox Type")
plt.ylabel("Count")
plt.show()
print(cars['vehicleType'].value_counts())
print(cars['model'].value_counts())
print(cars['brand'].value_counts())
col=['seller','offerType','abtest','fueltype_grouped']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
print(round(correlation,3))
cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
base_pred=np.mean(y_test)
print(base_pred)
base_pred=np.repeat(base_pred,len(y_test))
base_mean_absolute_error=mean_absolute_error(y_test,base_pred)
print("mean absolute error for base predict",base_mean_absolute_error)
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print("root mean square error for base predict",base_root_mean_square_error)
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
            'bootstrap':bootstrap}
model=RandomizedSearchCV(estimator=rf,param_distributions=param_grid,n_iter=50,cv=5,n_jobs=-1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("training score for model 1:",model.score(x_train,y_train))
print("test score for model 1:",model.score(x_test,y_test))
print("mean absolute error for model 1:",mean_absolute_error(y_test,y_pred))
print("mean squared error for model 1:",mean_squared_error(y_test,y_pred))
print("root mean squared error for model 1:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Residual for model 1:",np.mean(y_test-y_pred))
print("for model 1:")
print(model.best_estimator_)
print(model.best_index_)
print(model.best_params_)
print(model.best_score_)
vehicletype=input("enter the vehicleType:").lower()
gearbox=input("enter the car gearbox:").lower()
powerPS=int(input("enter the Horsepower:"))
model=input("enter the car model:").lower()
kilometer=int(input("enter the kilometer run:"))
fuelType=input("enter the fuelType:").lower()
brand=input("enter the car brand:").lower()
notReqairedDamage=input("enter whether the damage is not reqaired or not(yes/no):").lower()
Age=float(input("enter the age of the car:"))
user_input=pd.DataFrame({"vehicletype":[vehicletype],"gearbox":[gearbox],"powerPS":[powerPS],"model":[model],"kilometer":[kilometer],"fuelType":[fuelType],
   "brand":[brand],"notReqairedDamage":[notReqairedDamage],"Age":[Age]},index=[0])
missing_col=list(set(x1.columns)-set(user_input.columns))
user_input=pd.get_dummies(user_input,drop_first=True)
missing_data=pd.DataFrame(0,columns=missing_col,index=user_input.index)
user_input=pd.concat([user_input,missing_data],axis=1)
user_input=user_input[x1.columns]
user_prediction = model.predict(user_input)
print("Predicted Price:", user_prediction[0]) 
