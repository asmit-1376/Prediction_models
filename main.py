import pandas as pd
import numpy as np

data = pd.read_csv('student_scores.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1]

from sklearn.linear_model import LinearRegression
reg = LinearRegression() 
reg.fit(X,y)