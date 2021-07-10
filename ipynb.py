#!/usr/bin/env python
# coding: utf-8

# <font size="4"> __TASK - 1 Data Science And Business Analytics__ 
# 
# <font size="3"> <b> By Sneha Pisal Intern at The Sparks Foundation

# TASK - Predict the percentage of an student based on the no. of study hours.   

# In[4]:


#import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Reading data 
data=pd.read_csv('C:/Users/DELL/Desktop/file.csv')


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


#Check whether data has null values or not
data.isnull().sum()


# Since , There is no single missing value in the data we can do visualization

# In[10]:


sns.set_style('darkgrid')
sns.scatterplot(x=data.Hours,y=data.Scores)
plt.title('Hours Vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# We can see from above scatterplot that there is a relationship between Hours studied and percentage scored.Lets plot the line to confirm the correlation 

# In[11]:


sns.regplot(x=data.Hours,y=data.Scores)
plt.title('Hours Vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
print(data.corr())


# Thus,It is confirmed that Hours and Scores are positively correlated
# 

# In[12]:


#splitting of data into attributes(Hours) and labels(Scores)
X = data.iloc[:, :1].values  
Y = data.iloc[:, -1].values


# In[13]:


#splitting of data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[14]:


#Using LR model fitting of data
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,Y_train)
print(regression.intercept_)
print(regression.coef_)
print('-----Model Trained------')


# In[15]:


#Making predictions
pred_Y=regression.predict(X_test)
df=pd.DataFrame({'Actual Values':Y_test,'Predicted Values':pred_Y})
df


# In[16]:


#Calculating the accuracy of the model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print('Mean absolute error: ',mean_absolute_error(Y_test,pred_Y))
print('R2 Score:',r2_score(Y_test,pred_Y))


# In[17]:


#To predict score if student studies for 9.25hr/day
Hours=[9.25]
answer=regression.predict([Hours])
print("Score = {}".format(round(answer[0],3)))


# According to model if student studies for 9.25hr/day then,he/she likely to get <b>93.893 score
