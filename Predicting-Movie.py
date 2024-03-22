import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

df_rental = pd.read_csv("rental_info.csv")
#df_rental.info()

df_rental["rental_length"] = pd.to_datetime(df_rental["return_date"]) - pd.to_datetime(df_rental["rental_date"])
df_rental["rental_length_days"] = df_rental["rental_length"].dt.days

df_rental['behind_the_scenes']=np.where(df_rental["special_features"].str.contains("Behind the Scenes"),1,0)
df_rental["deleted_scenes"] =  np.where(df_rental["special_features"].str.contains("Deleted Scenes"), 1, 0)
df_rental.head()

col_drop=["special_features","rental_length_days","rental_length","rental_date", "return_date"]
X=df_rental.drop(col_drop,axis=1)
X.info()
y=df_rental["rental_length_days"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

lasso=Lasso(alpha=0.3,random_state=1)    
lasso.fit(X_train, y_train)

'''
 #-------test---------------
coefficients_before = X_train.mean()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(X_train.columns, coefficients_before)
plt.title('Coefficients Before Lasso')
plt.xticks(rotation=45)
plt.ylabel('Coefficients')
plt.subplot(1, 2, 2)
plt.bar(X_train.columns, lasso.coef_)
plt.title('Coefficients After Lasso')
plt.xticks(rotation=45)
plt.ylabel('Coefficients')
plt.tight_layout()
plt.show()
'''
lasso_coef=lasso.coef_
X_lasso_train=X_train.iloc[:,lasso_coef>0]
X_lasso_test=X_test.iloc[:,lasso_coef>0]

ols=LinearRegression()
ols= ols.fit(X_lasso_train,y_train)
y_test_pred=ols.predict(X_lasso_test)
mse_line_reg_lasso=mean_squared_error(y_test,y_test_pred)

param_dist= {'n_estimators': np.arange(1,101,1),
             'max_depth':np.arange(1,11,1)}

rf= RandomForestRegressor()
rand_search= RandomizedSearchCV(rf,param_distributions=param_dist,cv=5,random_state=1)
rand_search.fit(X_train,y_train)

hyper_params = rand_search.best_params_
#print(hyper_params)
rf=RandomForestRegressor(n_estimators=hyper_params['n_estimators'],max_depth=hyper_params['max_depth'], random_state=9)

rf.fit(X_train,y_train)
rf_pred= rf.predict(X_test)
mse_random_forest=mean_squared_error(y_test,rf_pred)
best_model = rf
best_mse = mse_random_forest
print("Best MSE:", best_mse)