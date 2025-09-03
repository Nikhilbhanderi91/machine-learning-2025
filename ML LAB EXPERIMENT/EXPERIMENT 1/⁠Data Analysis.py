import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Load the dataset
data=fetch_openml('titanic',version=1,as_frame=True)
print(data)

data['feature_names']

data_f=data.frame.copy()

#Dropping the empty values
data=data_f[['age','sex','fare','embarked','pclass','survived']].dropna()

set(list(data['embarked']))

#Lable Encoding
le=LabelEncoder()
data['embarked_le']=le.fit_transform(data['embarked'])

data.columns

data['embarked_le']

#One-Hot Encoder
ohe=OneHotEncoder()
# df_ohe=pd.get_dummies(data['sex'])
df_ohe=pd.get_dummies(data,columns=['sex'])
df_ohe

#Loadnthe dataset of diabetes
from sklearn.datasets import load_diabetes

data=load_diabetes(as_frame=True)
data=data.frame
print(data)

data.describe()

import seaborn as sns
import matplotlib.pyplot as plt

#Get coorelation in-between the features
corr=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)
plt.show()

#Feature Importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data.columns

X_train,X_test,y_train,y_test=train_test_split(data.drop('target',axis=1),data['target'],test_size=0.2,random_state=42)

tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X_train,y_train)

importance=pd.Series(tree.feature_importances_,index=X_train.columns)
importance

