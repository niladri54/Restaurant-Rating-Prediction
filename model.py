import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import gzip

df=pd.read_csv('zomato_df.csv')
df.drop('Unnamed: 0',axis=1, inplace=True)
X=df.drop('rate',axis=1)
y=df['rate']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

rfr=RandomForestRegressor(n_estimators=100)
rfr.fit(X_train,y_train)
# yhat=rfr.predict(X_test)

# print(r2_score(y_test,yhat),mean_squared_error(y_test,yhat))

# import pickle
# pickle.dump(rfr,open('model.pkl','wb'))
# model=pickle.load(open('model.pkl','rb'))

import pickle
import sklearn.externals
import joblib
joblib.dump(rfr,open('model.pkl','wb'),compress=9)
# pickle.dump(rfr, gzip.open("model.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
