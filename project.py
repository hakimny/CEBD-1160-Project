import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, max_error, explained_variance_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# https://machinelearningmastery.com/linear-regression-for-machine-learning/

# https://scikit-learn.org/stable/modules/model_evaluation.html

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8

# 1 - learn about our data by printing some stats
# 2 - create a heatmap to visualize correlation between features
# 3 - Review machine learning map to determine best practices in regression problems
# 4 - Try different regression estimator methods to determine best performance.
# 5 - In case of KNeighborsRegressor try with # numbers of neighbors to find best performance
# 6 - In case of Ridge & Lasso try # numbers of alpha parameter to determine best performance




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
    if title == "Linear Regression Performance":
        print("Intercept: %.2f" % reg_model.intercept_)
    print("Max Error ( close to 0.0 the better): %.2f" % max_error(y_test, predicted_values))

def draw_heatmap(df, path):
    os.makedirs(path, exist_ok=True)
    sns.set()
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, cmap='autumn')
    ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticklabels(df.columns, rotation=45)
    plt.savefig(f'{path}/diabetes_heatmap.png')
    plt.clf()

def process_models(data):
    names = list()
    scores = list()

    for name, model in data["models"]:
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        scores.append([ mean_squared_error(data["y_test"], predicted_values), r2_score(data["y_test"], predicted_values)])
        names.append(name)
    return pd.DataFrame({'Name': names, 'Score': scores})


diabetes = load_diabetes()
columns_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target
df = pd.DataFrame(data=diabetes['data'],columns=diabetes['feature_names'])

path = 'plots/'
draw_heatmap(df,path)

max_number_of_neighbors = 50
max_number_of_iterations = 20
alpha = 0.1
models = list()
result = list()
models.append(('LR', LinearRegression()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('SVR', svm.SVR()))
models.append(('RR', Ridge()))
models.append(('LS', Lasso()))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
data_dict = {
            "models": models,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
}
for i in range(max_number_of_iterations + 1):
    result.append(process_models(data_dict))
print(result)
# # Linear regression
# lr = LinearRegression()
# lr.fit(X_train, y_train)
#
# # Predicting the results for our test dataset
# predicted_values = lr.predict(X_test)
# performance("Linear Regression Performance",lr, X,y,y_test,predicted_values,columns_names)
#
#
#
# neigh = KNeighborsRegressor(n_neighbors=100)
# neigh
# neigh.fit(X_train, y_train)
# predicted_values = neigh.predict(X_test)
# performance("K Neighbors Regressor Performance",neigh, X,y,y_test,predicted_values,columns_names)
#
#
#
# clf = svm.SVR()
# clf.fit(X_train, y_train)
# predicted_values = clf.predict(X_test)
# performance("Support Vector Regression",clf, X,y,y_test,predicted_values,columns_names)
#
#
# reg = linear_model.Ridge(alpha=.5)
# reg.fit(X_train, y_train)
# predicted_values = reg.predict(X_test)
# performance("Ridge Regression",reg, X,y,y_test,predicted_values,columns_names)
#
# lasso = linear_model.Lasso(alpha=0.1)
# lasso.fit(X_train, y_train)
# predicted_values = lasso.predict(X_test)
# performance("Lasso Regression",lasso, X,y,y_test,predicted_values,columns_names)
