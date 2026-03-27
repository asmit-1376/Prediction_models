import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("student_scores.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(data)