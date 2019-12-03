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
        if name == "KNR":
            scores.append(process_knr(model, data))
            names.append(name)
        elif name == "LS" or name == "LR":
            scores.append(process_ridge_lasso(model, data))
            names.append(name)
        else:
            model.fit(data["X_train"], data["y_train"])
            predicted_values = model.predict(data["X_test"])
            scores.append([ mean_squared_error(data["y_test"], predicted_values), r2_score(data["y_test"], predicted_values)])
            names.append(name)
    return pd.DataFrame({'Name': names, 'Score': scores})


def process_knr(model, data):
    # we will process a loop to find the best performance for KNN for max_number_of_neighbors
    neighbor = 1
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    while neighbor < max_number_of_neighbors:

        model.n_neighbors = neighbor
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if min_mean_sqr_error < mean_sqr_error and max_r2_score > r2_score_calc:
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
        neighbor = neighbor + 1
    return [min_mean_sqr_error, max_r2_score]

def process_ridge_lasso(model, data):
    # We will try to find best alpha coefficient for best performance results for Ridge & lasso Estimators
    c_alpha = 0
    step = 0.001
    max_alpha = 20
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    while c_alpha <= max_alpha:
        model.alpha = c_alpha
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if min_mean_sqr_error < mean_sqr_error and max_r2_score > r2_score_calc:
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
        c_alpha = c_alpha + step
    return [min_mean_sqr_error, max_r2_score]


diabetes = load_diabetes()
columns_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target
df = pd.DataFrame(data=diabetes['data'],columns=diabetes['feature_names'])

path = 'plots/'
# draw_heatmap(df, path)

max_number_of_neighbors = 50
max_number_of_iterations = 20

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

# Get Optimized Number of Neighbors for KNR


def process_optimized_knr(data):
    # we will process a loop to find the best performance for KNN for max_number_of_neighbors
    neighbor = 1
    min_mean_sqr_error = 0
    max_r2_score = 0
    opt_neighbor = 0
    while neighbor < max_number_of_neighbors:
        model = KNeighborsRegressor()
        model.n_neighbors = neighbor
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if max_r2_score < abs(r2_score_calc):
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
            opt_neighbor = neighbor
        neighbor = neighbor + 1
    return opt_neighbor


def process_optimized_lasso(data):
    c_alpha = 0.001
    step = 0.001
    max_alpha = 20
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    opt_alpha = 0
    while c_alpha <= max_alpha:
        model = Lasso()
        model.alpha = c_alpha
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if min_mean_sqr_error < mean_sqr_error and max_r2_score > r2_score_calc:
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
            opt_alpha = c_alpha
            print(f'Lasso | Mean Sqr Error: {min_mean_sqr_error} | R2 Score: {max_r2_score} | alpha: {opt_alpha}')
        c_alpha = c_alpha + step
    return opt_alpha


def process_optimized_ridge(data):
    c_alpha = 0.001
    step = 0.001
    max_alpha = 20
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    opt_alpha = 0
    while c_alpha <= max_alpha:
        model = Lasso()
        model.alpha = c_alpha
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if min_mean_sqr_error < mean_sqr_error and max_r2_score > r2_score_calc:
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
            opt_alpha = c_alpha
            print(f'Ridge | Mean Sqr Error: {min_mean_sqr_error} | R2 Score: {max_r2_score} | alpha: {opt_alpha}')
        c_alpha = c_alpha + step
    return opt_alpha


optimized_neighbor = process_optimized_knr(data_dict)
optimized_lasso_alpha = process_optimized_lasso(data_dict)
optimized_bridge_alpha = process_optimized_ridge(data_dict)

print(optimized_neighbor)

#for i in range(max_number_of_iterations + 1):

# result.append(process_models(data_dict))
# print(process_models(data_dict))
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
