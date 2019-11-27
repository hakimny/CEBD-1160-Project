import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import datasets, linear_model

# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# https://machinelearningmastery.com/linear-regression-for-machine-learning/

# https://scikit-learn.org/stable/modules/model_evaluation.html

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# 1 - learn about our data by printing some stats
# 2 - create a heatmap to visualize correlation between features
# 3 - Review machine learning map to determine best practices in regression problems
# 4 - try different regression estimator methods to determine best performance.




def description(ds):
    print(f'Shape of dataset : {ds.shape}')
    print(f'stats of dataset: \n {ds.describe()}')

def performance(title, reg_model, X,y, y_test, predicted_values, columns_names):
    print(title)
    if title == "Linear Regression Performance":
        print(f"Named Coeficients: {pd.DataFrame(reg_model.coef_, columns_names)}")
    print("Mean squared error ( close to 0 the better): %.2f"
          % mean_squared_error(y_test, predicted_values))
    print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))
    print('score ( close to 1.0 the better): %.2f' % reg_model.score(X, y))
    if title == "Linear Regression Performance":
        print("Intercept: %.2f" % reg_model.intercept_)
    print("Max Error ( close to 0.0 the better): %.2f" % max_error(y_test, predicted_values))


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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, max_error,explained_variance_score



diabetes = load_diabetes()
columns_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)
performance("Linear Regression Performance",lr, X,y,y_test,predicted_values,columns_names)


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=100)
neigh.fit(X_train, y_train)
predicted_values = neigh.predict(X_test)
performance("K Neighbors Regressor Performance",neigh, X,y,y_test,predicted_values,columns_names)


from sklearn import svm

clf = svm.SVR()
clf.fit(X_train, y_train)
predicted_values = clf.predict(X_test)
performance("Support Vector Regression",clf, X,y,y_test,predicted_values,columns_names)


reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
predicted_values = reg.predict(X_test)
performance("Ridge Regression",reg, X,y,y_test,predicted_values,columns_names)

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
predicted_values = lasso.predict(X_test)
performance("Lasso Regression",lasso, X,y,y_test,predicted_values,columns_names)
