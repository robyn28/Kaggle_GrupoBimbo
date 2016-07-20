
# coding: utf-8

# # 03-exploratory-analysis for final project

# I am working on the Kaggle Grupo Bimbo competition dataset for this project. 
# Link to Grupo Bimbo Kaggle competition: [Kaggle-GrupoBimbo](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)

# In[ ]:

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
from sklearn import linear_model

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", font_scale=1)
get_ipython().magic(u'matplotlib inline')


# ![](../assets/images/workflow/data-science-workflow-01.png)
# 
# ## Part 1. Identify the Problem
# 
# 

# **Problem**: Given various sales/client/product data, we want to predict demand for each product at each store on a weekly basis. Per the train dataset, the average demand for a product at a store per week is 7.2 units. However, this does not factor in cases in which store managers under-predict demand for a product which we can see when returns=0 for that week. There are 74,180,464 records in the train data, of which 71,636,003 records have returns=0 or approx 96%. This generally means that managers probably often under predict product demand (unless that are exact on the money, which seems unlikely). 
# 
# **Goals**: The goal is to predict demand for each product at each store on a weekly basis while avoiding under-predicting demand.
# 
# **Hypothesis**: As stated previously, the average product demand at a store per week is 7.2 units per the train data. However, given the likelihood of managers underpredicint product demand, I hypothesize a good model should return a number higher than 7.2 units to more accurately predict demand.

# ![](../assets/images/workflow/data-science-workflow-02.png)
# 
# ## Part 2. Acquire the Data
# 
# Kaggle has provided five files for this dataset:  
# _train.csv_: Use for building a model (contains target variable "Demanda_uni_equil")  
# _test.csv_: Use for submission file (fill in for target variable "Demanda_uni_equil")
# _cliente_tabla.csv_: Contains client names (can be joined with train/test on Cliente_ID)
# _producto_tabla.csv_: Contains product names (can be join with train/test on Producto_ID)
# _town_state.csv_: Contains town and state (can be join with train/test on Agencia_ID)
# 
# 
# **Notes**: I will further split _train.csv_ to generate my own cross validation set. However, I will use all of _train.csv_ to train my final model since Kaggle has already supplied a test dataset. Additionally, I am only using a random 10% of the train data given to me for EDA and model development. Using the entire train dataset proved to be too time consuming for the quick iternations needed for initial modeling building and EDA efforts. I plan to use 100% of the train dataset once I build a model I'm comfortable with. I may have to explore using EC2 for this effort.

# In[ ]:

# Load train data
# Given size of training data, I chose to use only 10% for speed reasons
# QUESTION - how can i randomize with python? i used sql to create the random sample below.
df_train = pd.read_csv("train_random10percent.csv")

# Check head
df_train.head()


# In[ ]:

# Load test data
df_test = pd.read_csv("test.csv")

# Check head. I noticed that I will have to drop certain columns so that test and train sets have the same features. 
df_test.head()


# In[ ]:

#given that i cannot use a significant amount of variables in train data, i created additoinal features using the mean
#i grouped on product id since i will ultimately be predicting demand for each product

df_train_mean = df_train.groupby('Producto_ID').mean().add_suffix('_mean').reset_index()
df_train_mean.head()


# In[ ]:

#from above, adding 2 additional features, the average sales units and the average demand
df_train2 = df_train.merge(df_train_mean[['Producto_ID','Venta_uni_hoy_mean', 'Demanda_uni_equil_mean']],how='inner',on='Producto_ID')
df_train2.sample(5)


# In[ ]:

# Adding features to the test set in order to match train set
df_test2 = df_test.merge(df_train_mean[['Producto_ID','Venta_uni_hoy_mean', 'Demanda_uni_equil_mean']],how='left',on='Producto_ID')
df_test2.head()


# ![](../assets/images/workflow/data-science-workflow-03-05.png)
# 
# ## Part 3. Parse, Mine, and Refine the data
# 
# Perform exploratory data analysis and verify the quality of the data.

# ### Check columns and counts to drop any non-generic or near-empty columns

# In[ ]:

# Check columns
print "train dataset columns:"
print df_train2.columns.values
print 
print "test dataset columns:"
print df_test2.columns.values


# In[ ]:

# Check counts
print "train dataset counts:"
print df_train2.count()
print
print "test dataset counts:"
print df_test2.count()


# ### Check for missing values and drop or impute

# In[ ]:

# Check counts for missing values in each column
print "train dataset missing values:"
print df_train2.isnull().sum()
print
print "test dataset missing values:"
print df_test2.isnull().sum()


# ### Wrangle the data to address any issues from above checks

# In[ ]:

# Drop columns not included in test dataset
df_train2 = df_train2.drop(['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima'], axis=1)

# Check data
df_train2.head()


# In[ ]:

# Drop blank values in test set and replace with mean

# Replace missing values for venta_uni_hoy_mean using mean
df_test2.loc[(df_test2['Venta_uni_hoy_mean'].isnull()), 'Venta_uni_hoy_mean'] = df_test2['Venta_uni_hoy_mean'].dropna().mean()

# Replace missing values for demand using mean
df_test2.loc[(df_test2['Demanda_uni_equil_mean'].isnull()), 'Demanda_uni_equil_mean'] = df_test2['Demanda_uni_equil_mean'].dropna().mean()

print "test dataset missing values:"
print df_test2.isnull().sum()


# ### Perform exploratory data analysis

# In[ ]:

# Get summary statistics for data
df_train2.describe()


# In[ ]:

#RE RUN THIS LAST
# Get pair plot for data
sns.pairplot(df_train2)


# In[ ]:

#show demand by weeks
timing = pd.read_csv('train_random10percent.csv', usecols=['Semana','Demanda_uni_equil'])
print(timing['Semana'].value_counts())
plt.hist(timing['Semana'].tolist(), bins=7, color='blue')
plt.show()

#QUESTION - is this a time series problem since we are predicting demand for weeks 10 and 11? and beyond?


# In[ ]:

#Show box plot of demand by week
sns.factorplot(
    x='Semana',
    y='Demanda_uni_equil',
    data=df_train2,
    kind='box')


# ### Check and convert all data types to numerical

# In[ ]:

# Check data types
df_train.dtypes

#these are all numerical but are not continuous values and therefore don't have relative significant to one another, except for week
#however, creating dummy variables for all these is too memory intensive. as such, might have to explore using a random forest model
#in addition to the linear regression model


# ![](../assets/images/workflow/data-science-workflow-06.png)
# 
# ## Part 4. Build a Model
# 
# Create a cross validation split, select and build a model, evaluate the model, and refine the model

# ### Create cross validation sets

# In[ ]:

#create cross validation sets

#set target variable name
target = 'Demanda_uni_equil'

#set X and y
X = df_train2.drop([target], axis=1)
y = df_train2[target]

# create separate training and test sets with 60/40 train/test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size= .4, random_state=0) 


# ### Build a model

# In[ ]:

#create linear regression object
lm = linear_model.LinearRegression()

#train the model using the training data
lm.fit(X_train,y_train)


# ### Evaluate the model

# In[ ]:

# Check R^2 on test set
print "R^2: %0.3f" % lm.score(X_test,y_test)


# In[ ]:

# Check MSE on test set
#http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
print "MSE: %0.3f" % metrics.mean_squared_error(y_test, lm.predict(X_test))
#QUESTION - should i check this on train set?
print "MSE: %0.3f" % metrics.mean_squared_error(y_train, lm.predict(X_train))


# ![](../assets/images/workflow/data-science-workflow-07.png)
# 
# ## Part 5: Present the Results
# 
# Generate summary of findings and kaggle submission file.
# 
# NOTE: For the purposes of generating summary narratives and kaggle submission, we can train the model on the entire training data provided in _train.csv_.

# ### Load Kaggle training data and use entire data to train tuned model

# In[ ]:

# Set target variable name
target = 'Demanda_uni_equil'

# Set X_train and y_train
X_train = df_train2.drop([target], axis=1)
y_train = df_train2[target]


# In[ ]:

# Build tuned model
#create linear regression object
lm = linear_model.LinearRegression()

#train the model using the training data
lm.fit(X_train,y_train)

# Score tuned model
print "R^2: %0.3f" % lm.score(X_train, y_train)
print "MSE: %0.3f" % metrics.mean_squared_error(y_train, lm.predict(X_train))


# ### Load Kaggle test data, make predictions using model, and generate submission file

# In[ ]:

#create data frame for submission
df_sub = df_test2[['id']]

df_test2 = df_test2.drop('id', axis=1)

#predict using tuned model
df_sub['Demanda_uni_equil'] = lm.predict(df_test2)

df_sub.describe()


# In[ ]:




# In[ ]:

d = df_sub['Demanda_uni_equil']
d[d<0] = 0
df_sub.describe()


# In[ ]:

# Write submission file
df_sub.to_csv("mysubmission3.csv", index=False)


# **Kaggle score** : 0.75682
# 

# In[ ]:

#notes
#want to try to use a classifier like random forest or logistic regression

