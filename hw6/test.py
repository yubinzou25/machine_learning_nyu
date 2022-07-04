import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
df = pd.read_csv("VideoGameSales.csv")
df = df.drop(['Name'],axis =1)

##############################
from sklearn.model_selection import train_test_split
y = df[["Global_Sales"]].values.ravel()

# Generate dummies for all catagrical features
X = pd.get_dummies(df.drop(["Global_Sales"], axis=1)).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)


print("Original:", X.shape)
print("Train:   ", X_train.shape,y_train.shape)
print("Test:    ", X_test.shape,y_test.shape)

##############################
from sklearn.metrics import mean_squared_error

constant_mse = mean_squared_error(y_test, np.array([y_train.mean()]*y_test.shape[0]))
print("Constant MSE: %.2f" % constant_mse)

##############################
from sklearn.linear_model import SGDRegressor

#limit the max_iter and set the random seed to fix the out put.
sgd_model = SGDRegressor(
    penalty="l1",            # the type of regularization component to use,
                             # l1 indicates Lasso and l2 indicats Ridge
    max_iter=1000,           # maximal number of epochs
    tol=1e-3,                # tolerance for the stopping condition (stops if it
                             # can't improve the result more than tol), this speeds
                             # up model building and is great for prototyping
    alpha = 0.01,            # regularization strength, low = free model, high = controlled model
    random_state=0           # random seed, fix the output, keep it 0 all the time in this hw
)
##############################


# Note that for k=5 we only do validation for 4(k-1) times
# Input: data, an array inlucding all the train data you want to use for cv
# Output: splits, a list of tuples in the form of (train_index, val_index), just as the kfold() in sklearn
def customized_kfold(data, k):
    n = data.shape[0] # we first get the number of data points we have for future sue
    val_size = n//k
    splits = []
    for step in range(1, k):
        splits.append((np.array(range(val_size*step)), np.array(range(val_size*step+1, val_size*(step+1)))))
    return splits

##############################

# The function for full cross validation
# Input: X_train: full training X
#        y_train: full training y
#        k:       num of folds you want
#        model:   the sklearn model that you would like to do cv on
# Output: a tuple, (mean_train_mse, mean_val_mse)
def kfold_cross_validation(X_train, y_train, k, model):
    train_mse_list, val_mse_list = [], []
    for train_index, val_index in customized_kfold(X_train, k):
        X_train_sub, y_train_sub = X_train[train_index], y_train[train_index]
        X_val_sub, y_val_sub = X_train[val_index], y_train[val_index]
        model.fit(X_train_sub, y_train_sub)
        train_mse_list.append(mean_squared_error(y_train_sub, model.predict(X_train_sub)))
        val_mse_list.append(mean_squared_error(y_val_sub, model.predict(X_val_sub)))
    mean_train_mse = np.mean(train_mse_list)
    mean_val_mse = np.mean(val_mse_list)
    return (mean_train_mse, mean_val_mse)

##############################

model_list_l1 = []
for i in [1,2]:
    for j in [0, 0.001, 0.01, 0.1, 1, 5, 10]:
        model_list_l1.append({
        "pipeline": Pipeline([
            ("ss", StandardScaler()),
            ("poly", PolynomialFeatures(i)),
            ("sgd", SGDRegressor(penalty="l1",alpha=j,random_state=0))
        ]),
        "degree": i,
        "alpha": j
    })

mean_val_mse_list_l1 = []
for model_l1 in model_list_l1:
    mean_val_mse_list_l1.append(kfold_cross_validation(X_train, y_train, 10, model_l1["pipeline"])[1])
min_index_l1 = np.argmin(mean_val_mse_list_l1)
best_params_l1 = model_list_l1[min_index_l1]
best_score_l1 = mean_val_mse_list_l1[min_index_l1]
print(best_params_l1)
print(best_score_l1)

##############################

best_lasso = pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(best_params_l1["degree"])),
    ('sgd',  SGDRegressor(penalty="l1",alpha=best_params_l1["alpha"],random_state=0)),
])
best_lasso.fit(X_train, y_train)
mean_squared_error(best_lasso.predict(X_test), y_test)

##############################
def find_zero_and_nonezero_index(lst):
    zero_index_list = []
    nonzero_index_list = []
    for index, value in enumerate(lst):
        if value == 0:
            zero_index_list.append(index)
        else:
            nonzero_index_list.append(index)

    return (np.array(nonzero_index_list), np.array(zero_index_list))

##############################

# plug in the best params you just find.
pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(best_params_l1["degree"])),
    ('sgd',  SGDRegressor(penalty="l1", alpha=best_params_l1["alpha"],random_state=0)),
])

# Train using the whole pipeline using just 1 call!
pipeline.fit(X_train,y_train)

# Find the coeficients for the SGDRegressor
# Hint: You can retreive the model for each step in the pipeline use pipeline.steps
_,model_coef = pipeline.steps[2]
coef = model_coef.coef_
print(coef)

nonzero_index, zero_index = find_zero_and_nonezero_index(coef[1:])
test_mse_list = []
train_mse_list = []
X_train_selected = X_train[:, nonzero_index]
X_test_selected = X_test[:, nonzero_index]


model_list_l1 = []
for i in [1,2]:
    for j in [0, 0.001, 0.01, 0.1, 1, 5, 10]:
        model_list_l1.append({
        "pipeline": Pipeline([
            ("ss", StandardScaler()),
            ("poly", PolynomialFeatures(i)),
            ("sgd", SGDRegressor(penalty="l1",alpha=j,random_state=0))
        ]),
        "degree": i,
        "alpha": j
    })
mean_val_mse_list_l1 = []
for model_l1 in model_list_l1:
    mean_val_mse_list_l1.append(kfold_cross_validation(X_train_selected, y_train, 10, model_l1["pipeline"])[1])
min_index_l1 = np.argmin(mean_val_mse_list_l1)
best_params_l1 = model_list_l1[min_index_l1]
best_score_l1 = mean_val_mse_list_l1[min_index_l1]
print(best_params_l1)
print(best_score_l1)

best_lasso = pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(best_params_l1["degree"])),
    ('sgd',  SGDRegressor(penalty="l1",alpha=best_params_l1["alpha"],random_state=0)),
])
best_lasso.fit(X_train_selected, y_train)
mean_squared_error(best_lasso.predict(X_test_selected), y_test)