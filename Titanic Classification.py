#!/usr/bin/env python
# coding: utf-8

# # Task:1 Titanic Classification

# In[182]:


import numpy as np    #for numerical computation
import pandas as pd   #for data manipulation & analysis
import seaborn as sns #for statistical data visualization
import matplotlib.pyplot as plt  #data visualization
from sklearn.model_selection import train_test_split #for splitting data
from sklearn.metrics import confusion_matrix         #for classifcation metrics
from sklearn.metrics import classification_report    #for classifcation metrics
from sklearn.linear_model import LogisticRegression  #for logistic regression


# In[183]:


a = pd.read_csv("T:\Internship\Bharat intern\Titanice dataset/titanic.csv")
df = pd.DataFrame(a)
df.head(5)


# In[193]:


df.shape  #shape and size


# In[194]:


df.columns  #feature of data


# In[195]:


df.describe(include='all')


# In[196]:


# Checking Missing values
missing_values = df.isnull().sum()              
missing_values .sort_values(ascending = False)


# In[197]:


#visulaization the data
def count_plot(feature):    #barplot with count unique value
    sns.countplot(x=feature,data=df)
    plt.show()


# In[198]:


#column visualize
columns= ['Survived','Pclass','Sex','SibSp','Embarked']
for i in columns:
    count_plot(i)


# In[199]:


#data processing
df.head()


# In[200]:


columns_to_drop = ['PassengerId', 'Name', 'Cabin', 'Ticket']
df.drop(columns_to_drop, axis=1, inplace=True)


# In[201]:


#filling missingvalues
df['Age'].fillna(df['Age'].mean(),inplace=True)
df.isnull().sum()


# In[202]:


df


# In[203]:


#duplicate variable for 'sex' column get_dummies()
sex = pd.get_dummies(df['Sex'],drop_first = True)
sex.head()


# In[204]:


#duplicate variable for 'Embarked' column
embark = pd.get_dummies(df['Embarked'],drop_first = True)
embark.head()


# In[205]:


#duplicate variable for 'sex' column
pclass = pd.get_dummies(df['Pclass'],drop_first = True)
pclass.head()


# In[206]:


df.head()


# In[207]:


df.drop(['Sex','Embarked','Pclass'],axis=1, inplace= True)
df.head()


# In[208]:


#concatinating the original with dummy-variable
df= pd.concat([df,sex,embark,pclass],axis=1)
#converting the feature to string datatype
df.columns= df.columns.astype(str)
df.head()


# In[209]:


#Training the model

#creating variable X by dropping 'survived' column from df
X= df.drop(['Survived'], axis=1)

#creating variable Y by dropping 'survived' column from df
Y= df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42)


# In[214]:


#creating and training the logistic regression model
model = LogisticRegression(max_iter= 1000)
model.fit(X_train, Y_train)


# In[211]:


model.score(X_train,Y_train)


# In[212]:


model.score(X_test,Y_test)


# In[216]:


#using the model to predict the labels
Y_predicted = model.predict(X_test)

#Evaluate the performance of modelby calculating teh  confusion matirx
confusion_matrix(Y_test, Y_predicted)


# In[217]:


print(classification_report(Y_test,Y_predicted))


# In[ ]:




