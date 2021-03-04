# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:56:10 2021

@author: Karthik
"""

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import warnings
import os
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit as st
import chart_studio.plotly as py


def plot_prec_recall_vs_tresh(df,features,classifier,option):
  # generate dataset
  X, y = df[features], df['Default_Status']
# split into train/test sets
  trainX,testX,trainy, testy = train_test_split(X, y, random_state=20,stratify=y)
  #for name,model in models.items():
# fit a model
  model=classifier
    
  warnings.simplefilter("ignore")
  os.environ["PYTHONWARNINGS"] = "ignore"
  
    
# predict probabilities
  yhat = model.predict_proba(testX)
  y_scores=yhat
  precisions, recalls, thresholds = precision_recall_curve(testy, y_scores[:,1], )
  fig, ax = plt.subplots(figsize=(10,6))
  plt.plot(thresholds, precisions[:-1], linestyle="--", label="Precisions")
  plt.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
  plt.ylabel("Level of Precision and Recall of {}".format(option), fontsize=12)
  plt.title("Precision and Recall Scores as a function of the decision threshold", fontsize=12)
  plt.xlabel("Thresholds", fontsize=12)
  plt.legend(loc="best", fontsize=12)
  plt.ylim([0,1])
  plt.axvline(x=0.50, linewidth=3, color="#0B3861")
  #plt.show()
  # Export plot to plotly
  
  return fig
