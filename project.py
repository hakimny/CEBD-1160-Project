import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import datasets, linear_model

# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# https://machinelearningmastery.com/linear-regression-for-machine-learning/




# 1 - learn about our data by printing some stats
# 2 - create a heatmap to visualize correlation between features
# 3 - Review machine learning map to determine best practices in regression problems
# 4 - try different regression estimator methods to determine best performance.




def description(ds):
    print(f'Shape of dataset : {ds.shape}')
    print(f'stats of dataset: \n {ds.describe()}')

def performance(reg_model, X,y, y_test, predicted_values, columns_names):
    print(f"Named Coeficients: {pd.DataFrame(reg_model.coef_, columns_names)}")
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, predicted_values))
    print('Variance score: %.2f' % r2_score(y_test, predicted_values))
    print('score: %.2f' % lm.score(X, y))
    print(f"Intercept: {lm.intercept_}\n")



df = pd.read_csv(filepath_or_buffer='data/diabetes.data',
                      sep='\t',
                      header=0)
description(df)

path = 'plots/'
os.makedirs(path, exist_ok=True)
sns.set()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='autumn')
ax.set_xticklabels(df.columns, rotation=45)
ax.set_yticklabels(df.columns, rotation=45)
plt.savefig(f'{path}/diabetes_heatmap.png')
plt.clf()

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


diabetes = load_diabetes()
columns_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lm.predict(X_test)

performance(lm, X,y,y_test,predicted_values,columns_names)
