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
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

# 1 - learn about our data by printing some stats
# 2 - create a heatmap to visualize correlation between features
# 3 - Review machine learning map to determine best practices in regression problems
# 4 - Try different regression estimator methods to determine best performance.
# 5 - In case of KNeighborsRegressor try with # numbers of neighbors to find best performance
# 6 - In case of Ridge & Lasso try # numbers of alpha parameter to determine best performance


def description(ds):
    print(f'Shape of dataset : {ds.shape}')
    print(f'stats of dataset: \n {ds.describe()}')


# def performance(title, reg_model, X, y, y_test, predicted_values, columns_names):
#     print(title)
#     if title == "Linear Regression Performance":
#         print(f"Named Coeficients: {pd.DataFrame(reg_model.coef_, columns_names)}")
#     print("Mean squared error ( close to 0 the better): %.2f"
#           % mean_squared_error(y_test, predicted_values))
#     print('Variance score ( close to 1.0 the better ): %.2f' % r2_score(y_test, predicted_values))
#     if title == "Linear Regression Performance":
#         print("Intercept: %.2f" % reg_model.intercept_)
#     print("Max Error ( close to 0.0 the better): %.2f" % max_error(y_test, predicted_values))


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
    global optimized_neighbor
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
    global optimized_lasso_alpha
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
    global optimized_bridge_alpha
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
# X = diabetes.data
# y = diabetes.target
# print(diabetes['target'])

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["Y"] = diabetes.target
X = df.drop("Y", 1)   #Feature Matrix
y = df["Y"]          #Target Variable
df.head()



path = 'plots/'
cor = df.corr()
draw_heat_map(df, path)

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

# Features Selection based on their correlation with Variable Y

# Correlation with output variable
cor_target = abs(cor["Y"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.4]
print(relevant_features)

# Output
# bmi    0.586450
# bp     0.441484
# s4     0.430453
# s5     0.565883
# Y      1.000000

# Potential candidates bmi, bp, s4 & s5
# We will now check how those features relate to each other to eliminate those which are highly correlated.
print(df[["bmi", "bp"]].corr())
print(df[["bmi", "s4"]].corr())
print(df[["bmi", "s5"]].corr())
print(df[["bp", "s4"]].corr())
print(df[["bp", "s5"]].corr())
print(df[["s4", "s5"]].corr())

# Output
#           bmi        bp
# bmi  1.000000  0.395415
# bp   0.395415  1.000000
#           bmi        s4
# bmi  1.000000  0.413807
# s4   0.413807  1.000000
#           bmi        s5
# bmi  1.000000  0.446159
# s5   0.446159  1.000000
#           bp        s4
# bp  1.000000  0.257653
# s4  0.257653  1.000000
#           bp        s5
# bp  1.000000  0.393478
# s5  0.393478  1.000000
#           s4        s5
# s4  1.000000  0.617857
# s5  0.617857  1.000000

# Analysis S4 & S5 are highly correlated and since S5 is highly correlated to Y, we will drop S4 and Keep S5


def process_lr_step2(data):
    model = LinearRegression()
    model.fit(data["X_train"], data["y_train"])
    predicted_values = model.predict(data["X_test"])
    mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
    r2_score_calc = r2_score(data["y_test"], predicted_values)
    return {"name": "LR", "data": {"coefficients": model.coef_, "intercept": model.intercept_},
            "mean_sqr_err": mean_sqr_error, "r2_score": r2_score_calc}


def process_optimized_ridge_step2(data):
    model = Ridge()
    model.alpha = optimized_bridge_alpha
    model.fit(data["X_train"], data["y_train"])
    predicted_values = model.predict(data["X_test"])
    mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
    r2_score_calc = r2_score(data["y_test"], predicted_values)
    dict_result = {"name": "RR", 'data': {"alpha": optimized_bridge_alpha}, 'mean_sqr_err': mean_sqr_error,
                            'r2_score': r2_score_calc}
    return dict_result


def process_optimized_lasso_step2(data):
    model = Lasso()
    model.alpha = optimized_lasso_alpha
    model.fit(data["X_train"], data["y_train"])
    predicted_values = model.predict(data["X_test"])
    mean_sqr_error = mean_squared_error(data["y_test"], predicted_values)
    r2_score_calc = r2_score(data["y_test"], predicted_values)
    return {"name": "LASSO", "data": {"alpha": optimized_lasso_alpha}, "mean_sqr_err": mean_sqr_error,
                    "r2_score": r2_score_calc}


# df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# df["Y"] = diabetes.target
X = df.drop(["s4", "Y"], axis=1)   #Feature Matrix
y = df["Y"]          #Target Variable
df.head()
models = list()
result_step2 = list()
models.append(('LR', LinearRegression()))
models.append(('RR', Ridge(alpha=optimized_bridge_alpha)))
models.append(('LS', Lasso(alpha=optimized_lasso_alpha)))

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
    sub_result.append(process_lr_step2(data_dict))
    sub_result.append(process_optimized_lasso_step2(data_dict))
    sub_result.append(process_optimized_ridge_step2(data_dict))
    result_step2.append(sub_result)

results_data_step2 = list()

for item in result_step2:
    index = 0
    while index < len(item):
        temp_list = [item[index]['name'], item[index]['mean_sqr_err'], item[index]['r2_score']]
        results_data_step2.append(temp_list)
        index = index + 1

results_df = pd.DataFrame(data=results_data_step2, columns=['Name', 'Mean Sqr Err', 'R2 Score'])
r2_score_by_name_step2 = results_df.groupby('Name')
list_mean_model_r2_score = list()

for key, value in r2_score_by_name_step2:
    list_mean_model_r2_score.append([key, value['R2 Score'].mean()])

df_mean_model_r2_score = pd.DataFrame(data=list_mean_model_r2_score, columns=['Name', 'R2 Score'])
axis = sns.barplot(x='Name', y='R2 Score', data=df_mean_model_r2_score)
axis.set(xlabel='Model', ylabel='Average R2 Score')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")

plt.savefig(f'{path}/diabetes_data_average_r2_score_per_model_optimized.png')
plt.close()

list_max_model_r2_score = list()

for key, value in r2_score_by_name_step2:
    list_max_model_r2_score.append([key, value['R2 Score'].max()])

df_max_model_r2_score = pd.DataFrame(data=list_max_model_r2_score, columns=['Name', 'R2 Score'])
axis = sns.barplot(x='Name', y='R2 Score', data=df_max_model_r2_score)
axis.set(xlabel='Model', ylabel='Max R2 Score')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")
plt.savefig(f'{path}/diabetes_data_max_r2_score_per_model_optimized.png')
plt.close()