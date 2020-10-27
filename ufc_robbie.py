# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:03:35 2020

@author: deega
"""

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np



# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("ufc-master.csv")

data.head()

print(data)
data.dtypes['weight_class']

plt.style.use('fivethirtyeight')

data['weight_class'].value_counts().plot(kind="bar")
data['gender'].value_counts().plot(kind="bar")
data['B_stance'].value_counts().plot(kind="bar")
