#%%
# Breast Cancer Prediction using Logistic Regression(Logistic regression is a classification algorithm unlike its name)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df=pd.read_csv('/Users/sumitsharma/Desktop/data.csv')
df.dropna(axis=1,inplace=True)
# print(df.columns)
#Counting number of malignant and benign tumors
df['diagnosis'].value_counts()
#Encoding Labels
labelencoder=LabelEncoder()
labelencoder.fit_transform(df.iloc[:,1].values)
x=df.iloc[:,2:].values
y=df.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#scaling the standard scaler
standscaler=StandardScaler()
x_train=standscaler.fit_transform(x_train)
x_test=standscaler.fit_transform(x_test)
#building a logistic regression classifier
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
predicted=classifier.predict(x_test)
#we use confusion matrix to predict the actual value and the predicted value
cm=confusion_matrix(y_test,predicted)
sns.heatmap(cm,annot=True)
ac=accuracy_score(y_test,predicted)
print(ac)









# %%
