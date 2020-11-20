#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[35]:


# data = pd.read_csv('./cars_v2(ModelReady).csv', index_col='Unnamed: 0')


# In[84]:


class SelectModel:
    
    def __init__(self,make,model,data):
        self.model = model
        self.make = make
        self.data = data
    
    def filter_data(self):
        cond = (self.data['manufacturer']==self.make) & (self.data['model']==self.model)
        return self.data[cond]
        
    def fmtvs(self):
        model_data = self.filter_data()
        self.data_state = model_data['state']
        return (model_data[['age','odometer', 'miles_per_year','condition']],model_data['price'])
    
    def fit_predict(self, sample_case):
        self.X, self.y = self.fmtvs()
        
        model = make_pipeline(
            OneHotEncoder(),
            StandardScaler(),
            RandomForestRegressor(random_state=42, n_jobs=-1))
        model.fit(self.X,self.y)
       
        params = {'randomforestregressor__n_estimators': range(50,151,10),
                  'randomforestregressor__max_depth': range(30,101,10),
                  'randomforestregressor__min_samples_split': range(2,20)}
        search = RandomizedSearchCV( 
                                    model,
                                    params,
                                    n_iter=30,
                                    n_jobs=-1,
                                    cv=5,
                                    verbose=True)
        search.fit(self.X,self.y)
        
        self.model = search.best_estimator_
        self.params = search.best_params_
        self.score = search.best_score_
        self.prediction = int(self.model.predict(sample_case))
        self.mae = int(mean_absolute_error(self.y, self.model.predict(self.X)))
        
        return self.prediction

