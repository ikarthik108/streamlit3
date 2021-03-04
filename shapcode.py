# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:19:03 2021

@author: Karthik
"""

import shap
from sklearn.model_selection import train_test_split
import streamlit as st

def shap_explainer(dataframe,feature_set,model):
    
    
    X=dataframe[feature_set]
    y=dataframe['Default_Status']
    train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=20)
    #model.fit(train_X,train_y)
    

    shap_values = shap.TreeExplainer(model).shap_values(train_X)
    feat_plot=shap.summary_plot(shap_values, train_X, plot_type="bar",show=False)
    summary=shap.summary_plot(shap_values, train_X,show=False)
    
 
    return feat_plot, summary