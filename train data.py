#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


# In[3]:


df = pd.read_excel("Data_Train.xlsx")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


#convert date of jounery into datetime format
df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y')


# In[10]:


df.dtypes


# In[11]:


#extract day,month and year from date of journey using split function
df['Day']=df['Date_of_Journey'].dt.date
df['Month']=df['Date_of_Journey'].dt.month
df['Day']=df['Date_of_Journey'].dt.year


# In[12]:


df.drop('Date_of_Journey',axis=1,inplace=True)


# In[13]:


df.head()


# In[14]:


df['Dep_hour']=df['Dep_Time'].str.split(':').str[0].astype(int)
df['Dep_min']=df['Dep_Time'].str.split(':').str[0].astype(int)


# In[15]:


#first exytact the arrival time and remove the day and month from the arrival time column
df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]


# In[16]:


df.head()


# In[17]:


# extract the arrival hour and arrival min from the arrival time column
df['Arrival_hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
df['Arrival_min'] = df['Arrival_Time'].str.split(':').str[1].astype(int)


# In[18]:


df.head()


# In[19]:


#clean the total stops column
df['Total_Stops'].value_counts()


# In[20]:


#convert the non stop value into 0 stop value
df['Total_Stops']=df['Total_Stops'].replace("non-stop","0 stop")


# In[21]:


df['Total_Stops'].value_counts()


# In[22]:


df.head()


# In[23]:


df['Total_Stops']=df['Total_Stops'].replace("0 stop","0")
df['Total_Stops']=df['Total_Stops'].replace("1 stop","1")
df['Total_Stops']=df['Total_Stops'].replace("2 stops","2")
df['Total_Stops']=df['Total_Stops'].replace("3 stops","3")
df['Total_Stops']=df['Total_Stops'].replace("4 stops","4")


# In[24]:


df.head()


# In[25]:


df['Route'].unique()


# In[26]:


df['Route_1']=df['Route'].str.split('→').str[0].str.strip()
df['Route_2']=df['Route'].str.split('→').str[1].str.strip()
df['Route_3']=df['Route'].str.split('→').str[2].str.strip()
df['Route_4']=df['Route'].str.split('→').str[3].str.strip()
df['Route_5']=df['Route'].str.split('→').str[4].str.strip()


# In[38]:


df.head()


# In[27]:


df['Route_1']=df['Route'].fillna('None')
df['Route_2']=df['Route'].fillna('None')
df['Route_3']=df['Route'].fillna('None')
df['Route_4']=df['Route'].fillna('None')
df['Route_5']=df['Route'].fillna('None')


# In[28]:


df.drop('Route',axis=1,inplace=True)


# In[29]:


plt.figure(figsize=(5,6))
sns.histplot(df['Price'],kde=True)
plt.title('Distribution of Flight Prices')
plt.show()


# In[30]:


#plot the count of airlines
plt.figure(figsize=(5,6))
sns.countplot(x='Airline',data=df,color='red')
plt.title('count of airlines')
plt.xticks(rotation=90)
plt.show()


# In[31]:


#plot the count of airlines in horizontal
plt.figure(figsize=(5,6))
sns.countplot(y=df['Airline'],order=df['Airline'].value_counts().index)
plt.title('count of airlines')
plt.show()


# In[32]:


df.head()


# In[33]:


#check central tendency of the price column
print('Mean:',df['Price'].mean())
print('Median:',df['Price'].median())
print('Mode:',df['Price'].mode().iloc[0])


# In[34]:


#splitting data into 2 parts- x & y
x=df.drop('Price',axis=1)
y=df['Price']


# In[35]:


#splitting data into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[40]:


#train the model
from sklearn.linear_model import LinearRegression
#load the model
model=LinearRegression()


# In[42]:


X_train = pd.get_dummies(X_train, drop_first=True)
model.fit(X_train, Y_train)


# In[43]:


#fit the model
train_model=model.fit(X_train,Y_train)


# In[45]:


X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# In[51]:


model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


# In[52]:


#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,Y_pred)
msa=mean_absolute_error(Y_test,Y_pred)
r2=r2_score(Y_test,y_pred)
print("Mean Squared Error for Linear Regression:",mse)
print("Mean Absolute Error for Linear Regression:",msa)
print("R-squared Score for Linear Regression:",r2)


# In[53]:


#model 2 : Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
#load the model
model=DecisionTreeRegressor()
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred=model.predict(X_test)
y_pred
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred) 
msa=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("Mean Squared Error for Decision Tree Regression:",mse)
print("Mean Absolute Error for Decision Tree Regression:",msa)
print("R-squared Score for Decision Tree Regression:",r2)


# In[54]:


#model 3: SVR
from sklearn.svm import SVR
#load the model
model=SVR(kernel='linear',C=1.0,epsilon=0.1)
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred=model.predict(X_test)
y_pred
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred)
msa=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("Mean Squared Error for SVR:",mse)
print("Mean Absolute Error for SVR:",msa)
print("R-squared Score for SVR:",r2)


# In[55]:


#model 4: KNN
from sklearn.neighbors import KNeighborsRegressor
#load the model
model=KNeighborsRegressor(n_neighbors=5)
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred=model.predict(X_test)
y_pred
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred)
msa=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("Mean Squared Error for KNN:",mse)
print("Mean Absolute Error for KNN:",msa)
print("R-squared Score for KNN:",r2)


# In[56]:


#model 5:random forest
from sklearn.ensemble import RandomForestRegressor
#load the model
model=RandomForestRegressor(n_estimators=100,random_state=9)
train_model=model.fit(X_train,Y_train)
#make predictions
y_pred=model.predict(X_test)
y_pred
#elvaute the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(Y_test,y_pred)
msa=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("Mean Squared Error for Random Forest Regression:",mse)
print("Mean Absolute Error for Random Forest Regression:",msa)
print("R-squared Score for Random Forest Regression:",r2)


# In[ ]:





# In[ ]:




