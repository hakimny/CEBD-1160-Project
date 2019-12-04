import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, max_error, explained_variance_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split


# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

# https://machinelearningmastery.com/linear-regression-for-machine-learning/

# https://scikit-learn.org/stable/modules/model_evaluation.html

# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8
# https://realpython.com/pandas-groupby/

# 1 - learn about our data by printing some stats
# 2 - create a heatmap to visualize correlation between features
# 3 - Review machine learning map to determine best practices in regression problems
# 4 - Try different regression estimator methods to determine best performance.
# 5 - In case of KNeighborsRegressor try with # numbers of neighbors to find best performance
# 6 - In case of Ridge & Lasso try # numbers of alpha parameter to determine best performance


def description(ds):
    print(f'Shape of dataset : {ds.shape}')
    print(f'stats of dataset: \n {ds.describe()}')


def performance(title, reg_model, X, y, y_test, predicted_values, columns_names):
    print(title)
    if title == "Linear Regression Performance":
        print(f"Named Coeficients: {pd.DataFrame(reg_model.coef_, columns_names)}")
    print("Mean squared error ( close to 0 the better): %.2f"
          % mean_squared_error(y_test, predicted_values))
    print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))
    if title == "Linear Regression Performance":
        print("Intercept: %.2f" % reg_model.intercept_)
    print("Max Error ( close to 0.0 the better): %.2f" % max_error(y_test, predicted_values))


def draw_heat_map(df, path):
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
            scores.append(
                [mean_squared_error(data["y_test"], predicted_values), r2_score(data["y_test"], predicted_values)])
            names.append(name)
    return pd.DataFrame({'Name': names, 'Score': scores})


def process_lr(data):
    model = LinearRegression()
    model.fit(data["X_train"], data["y_train"])
    predicted_values = model.predict(data["X_test"])
    mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
    r2_score_calc = r2_score(data["y_test"], predicted_values)
    return {"name": "LR", "data": {"coefficients": model.coef_, "intercept": model.intercept_},
            "mean_sqr_err": mean_sqr_error, "r2_score": r2_score_calc}


def process_svr(data):
    model = svm.SVR(gamma='scale')
    model.fit(data["X_train"], data["y_train"])
    predicted_values = model.predict(data["X_test"])
    mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
    r2_score_calc = r2_score(data["y_test"], predicted_values)
    return {"name": "SVR", "data": {},
            'mean_sqr_err': mean_sqr_error, 'r2_score': r2_score_calc}


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
            optimized_neighbor = neighbor
        neighbor = neighbor + 1
    return {"name": "KNR", "data": {"neighbors": optimized_neighbor}, "mean_sqr_err": min_mean_sqr_error,
                    "r2_score": max_r2_score}


def process_optimized_lasso(data):
    c_alpha = 0.001
    step = 0.01
    max_alpha = 20
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    while c_alpha <= max_alpha:
        model = Lasso()
        model.alpha = c_alpha
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if max_r2_score < abs(r2_score_calc):
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
            optimized_lasso_alpha = c_alpha
        c_alpha = c_alpha + step
    return {"name": "LASSO", "data": {"alpha": optimized_lasso_alpha}, "mean_sqr_err": min_mean_sqr_error,
                    "r2_score": max_r2_score}


def process_optimized_ridge(data):
    c_alpha = 0.001
    step = 0.01
    max_alpha = 20
    min_mean_sqr_error = 10000000
    max_r2_score = 0
    while c_alpha <= max_alpha:
        model = Ridge()
        model.alpha = c_alpha
        model.fit(data["X_train"], data["y_train"])
        predicted_values = model.predict(data["X_test"])
        mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
        r2_score_calc = r2_score(data["y_test"], predicted_values)
        if max_r2_score < abs(r2_score_calc):
            min_mean_sqr_error = mean_sqr_error
            max_r2_score = r2_score_calc
            optimized_bridge_alpha = c_alpha
        c_alpha = c_alpha + step
        dict_result = {"name": "RR", 'data': {"alpha": optimized_bridge_alpha}, 'mean_sqr_err': min_mean_sqr_error,
                       'r2_score': max_r2_score}
    return dict_result


diabetes = load_diabetes()
columns_names = diabetes.feature_names
X = diabetes.data
y = diabetes.target
df = pd.DataFrame(data=diabetes['data'], columns=diabetes['feature_names'])

path = 'plots/'
# draw_heat_map(df, path)

max_number_of_neighbors = 50
max_number_of_iterations = 20
optimized_neighbor = 0
optimized_lasso_alpha = 0
optimized_bridge_alpha = 0
models = list()
result = list()
models.append(('LR', LinearRegression()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('SVR', svm.SVR()))
models.append(('RR', Ridge()))
models.append(('LS', Lasso()))

# Run regressors models through iterations to collect data
for i in range(max_number_of_iterations + 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    data_dict = {
        "models": models,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    sub_result = list()
    sub_result.append(process_lr(data_dict))
    sub_result.append(process_svr(data_dict))
    sub_result.append(process_optimized_knr(data_dict))
    sub_result.append(process_optimized_lasso(data_dict))
    sub_result.append(process_optimized_ridge(data_dict))
    result.append(sub_result)

results_data = list()

for item in result:
    index = 0
    while index < len(item):
        temp_list = [item[index]['name'], item[index]['mean_sqr_err'], item[index]['r2_score']]
        results_data.append(temp_list)
        index = index + 1

results_df = pd.DataFrame(data=results_data, columns=['Name', 'Mean Sqr Err', 'R2 Score'])

r2_score_by_name = results_df.groupby('Name')

list_mean_model_r2_score = list()
for key, value in r2_score_by_name:
    list_mean_model_r2_score.append([key, value['R2 Score'].mean()])

df_mean_model_r2_score = pd.DataFrame(data=list_mean_model_r2_score, columns=['Name', 'R2 Score'])
axis = sns.barplot(x='Name', y='R2 Score', data=df_mean_model_r2_score)
axis.set(xlabel='Model', ylabel='Average R2 Score')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")
plt.savefig(f'{path}/diabetes_data_average_r2_score_per_model.png')
plt.close()
list_max_model_r2_score = list()
for key, value in r2_score_by_name:
    list_max_model_r2_score.append([key, value['R2 Score'].max()])

df_max_model_r2_score = pd.DataFrame(data=list_max_model_r2_score, columns=['Name', 'R2 Score'])
axis = sns.barplot(x='Name', y='R2 Score', data=df_max_model_r2_score)
axis.set(xlabel='Model', ylabel='Max R2 Score')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")
plt.savefig(f'{path}/diabetes_data_max_r2_score_per_model.png')

plt.close()
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
