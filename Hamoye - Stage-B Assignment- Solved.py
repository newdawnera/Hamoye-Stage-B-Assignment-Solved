#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
datasets = pd.read_csv(r'C:\Users\Tunji\Desktop\Hamoye\energydata_complete.csv')


# In[2]:


datasets.shape


# In[3]:


column_names = {'T2':'Temperature_in_living_room_area', 'T6':'Temperature_outside_the_building_north_side'}
datasets = datasets.rename(columns=column_names)


# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
datasets = datasets.drop(columns=['date', 'lights'])


# In[5]:


normalised_datasets = pd.DataFrame(scaler.fit_transform(datasets), columns=datasets.columns)
X = normalised_datasets['Temperature_in_living_room_area']
y = normalised_datasets['Temperature_outside_the_building_north_side']


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import numpy as np

linear_model = LinearRegression()
linear_model.fit(X_train.to_frame(), y_train)
predicted_values = linear_model.predict(X_test.to_frame())


# In[7]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# In[8]:


new_normalised_datasets = normalised_datasets.drop(columns=['Appliances'])
X = new_normalised_datasets
y = normalised_datasets['Appliances']


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import numpy as np

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
predicted_values = linear_model.predict(X_test)


# In[10]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# In[11]:


rss = np.sum(np.square(y_test - predicted_values))
round(rss, 2)


# In[12]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[13]:


from sklearn.metrics import r2_score
coefficient_of_determination = r2_score(y_test, predicted_values)
round(coefficient_of_determination, 2)


# In[14]:


def get_weights_datasets(model, feat, col_name):
    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_datasets = pd.DataFrame(weights).reset_index()
    weights_datasets.columns = ['Features', col_name]
    weights_datasets[col_name].round(3)
    return weights_datasets


# In[15]:


linear_model_weights = get_weights_datasets(linear_model, X_train, 'Linear_Model_Weight')
print(linear_model_weights)


# In[16]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X_train, y_train)


# In[17]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[18]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train, y_train)


# In[19]:


lasso_weights_df = get_weights_datasets(lasso_reg, X_train, 'Lasso_Weight')
print(lasso_weights_df)


# In[20]:


rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)

