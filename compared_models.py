import numpy as np
import sklearn
import os
from os import listdir
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import keras

########################################################################################################################
# PREPROCCESSING
# read data
data = pd.read_csv('train.csv')
data = data.values
data = data.astype('float32')

# divide features and labels
features_data = data[:, :-1]
labels_data = data[:, -1]
# labels_data = np.reshape(labels_data, (len(labels_data), 1))


def get_best_features(all_features_score):
    sorted_params = np.asarray(sorted(all_features_score, key=abs))
    best_10 = sorted_params[-10:].tolist()
    all_features_score = all_features_score.tolist()
    best_feature_indexes = [all_features_score.index(i) for i in best_10]
    return best_feature_indexes

########################################################################################################################


# REDUCING DIMENSIONS
from sklearn.decomposition import PCA


def reduce_dimensions(train_set, n_components):
    pca = PCA(n_components=n_components)
    train_set = pca.fit_transform(train_set)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return train_set


reduce_dimensions(features_data, int(features_data.shape[0]))

# train test split
x_train, x_test, y_train, y_test = train_test_split(features_data, labels_data, test_size=0.2, random_state=42)


########################################################################################################################
# TRAIN MODELS
# BOOSTING
seed = 7
xgb = XGBClassifier()
xgb.fit(x_train, y_train)


# DISPLAY FEATURES IMPORTANCE
print('Features weights Boosting', xgb.feature_importances_)

# plot feature importance
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.show()

thresholds = np.sort(xgb.feature_importances_)[::-1]
print('Top 10 best features with Boosting:', get_best_features(xgb.feature_importances_))
print('Sorted feature weights: ', thresholds)


def select_best_params_xgb(sorted_features):
    thresholds = sorted_features
    for thresh in thresholds:
        selection = SelectFromModel(xgb, threshold=thresh, prefit=True)
        select_data = selection.transform(features_data)
        selection_model = XGBClassifier()
        selection_model.fit(select_data, labels_data)
        select_test = selection.transform(x_test)
        select_y_pred = selection_model.predict(select_test)
        accuracy = accuracy_score(y_test, select_y_pred)
        print('accuracy', (thresh, select_data.shape[1], accuracy*100))

# # ACCURACY OF SELECTED MODELS
# select_best_params_xgb(thresholds)


# ACCURACY
y_pred_xgb = xgb.predict(x_test)
score_xgb = accuracy_score(y_test, y_pred_xgb)

# CROSS VALIDATION
kfold = KFold(n_splits=10, random_state=seed)
c_v_xgb = cross_val_score(xgb, features_data, labels_data, cv=kfold, scoring='neg_mean_squared_error').mean()
print('Accuracy Boosting:', score_xgb, 'Cross validation Boosting', c_v_xgb)

# TUNING PARAMETERS
def tuning_params():
    param_test = {
        'max_depth': range(1, 10, 1),
        'min_child_weight': range(1, 12, 1),
        'gamma': [i/10.0 for i in range(0, 5)],
        'n_estimators': [50, 100, 150, 200],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
    }

    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                   n_estimators=50,
                                                   max_depth=5,
                                                   min_child_weight=1,
                                                   gamma=0,
                                                   subsample=0.8,
                                                   colsample_bytree=0.8,
                                                   objective='binary:logistic',
                                                   nthread=4,
                                                   scale_pos_weight=1,
                                                   seed=27),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(features_data, labels_data)
    print(gsearch.best_params_, gsearch.best_score_)
    return gsearch
# tuning_params()


########################################################################################################################
# LOGISTIC REGRESSION
reg = LogisticRegression()
reg.fit(x_train, y_train)

# feature importance
print('Features weights Log regression', reg.coef_)

# plot feature importance
plt.bar(range(len(reg.coef_[0])), abs(reg.coef_[0]))
plt.show()

thresholds_reg = np.sort(abs(reg.coef_[0]))[::-1]
print('Top 10 best features with Logistic regression:', get_best_features(reg.coef_[0]))
print(thresholds_reg)


def select_best_params_reg(sorted_features):
    thresholds = sorted_features
    for thresh in thresholds:
        selection = SelectFromModel(reg, threshold=thresh, prefit=True)
        select_data = selection.transform(features_data)
        selection_model = LogisticRegression()
        selection_model.fit(select_data, labels_data)
        select_test = selection.transform(x_test)
        select_y_pred = selection_model.predict(select_test)
        accuracy = accuracy_score(y_test, select_y_pred)
        print('accuracy', (thresh, select_data.shape[1], accuracy*100))

def tuning_reg():
    clf = LogisticRegression()
    param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}
    grid_search = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(features_data, labels_data)
    return grid_search.best_params_

# print(tuning_reg())

# # ACCURACY OF SELECTED MODELS
# select_best_params_reg(thresholds_reg)

y_pred_reg = reg.predict(x_test)
score_reg = accuracy_score(y_test, y_pred_reg)

kfold = KFold(n_splits=10, random_state=seed)
c_v_reg = cross_val_score(reg, features_data, labels_data, cv=kfold, scoring='neg_mean_squared_error').mean()

print('Accuracy Log regression', score_reg, 'Cross validation Log regression: ', c_v_reg)

########################################################################################################################
# SUPPORT VECTOR MACHINES
from sklearn import svm
sup_vec_m = svm.LinearSVC()
sup_vec_m.fit(x_train, y_train)

print('Features weights SVM: ', sup_vec_m.coef_)

# plot feature importance
plt.bar(range(len(sup_vec_m.coef_[0])), abs(sup_vec_m.coef_[0]))
plt.show()

thresholds_svm = np.sort(abs(sup_vec_m.coef_[0]))[::-1]
print('Top 10 best features with SVM: ', get_best_features(sup_vec_m.coef_[0]))
print(thresholds_svm)


def select_best_params_svm(sorted_features):
    thresholds = sorted_features
    for thresh in thresholds:
        selection = SelectFromModel(reg, threshold=thresh, prefit=True)
        select_data = selection.transform(features_data)
        selection_model = svm.LinearSVC()
        selection_model.fit(select_data, labels_data)
        select_test = selection.transform(x_test)
        select_y_pred = selection_model.predict(select_test)
        accuracy = accuracy_score(y_test, select_y_pred)
        print('accuracy', (thresh, select_data.shape[1], accuracy*100))


# # ACCURACY OF SELECTED MODELS
# select_best_params_svm(thresholds_svm)


# TUNING SVM PARAMETERS
def svc_params_selection():
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(svm.LinearSVC(), param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(features_data, labels_data)
    return grid_search.best_params_


print('Grid Search results SVM: ', svc_params_selection())

# ACCURACY FOR SVM
y_pred_svm = sup_vec_m.predict(x_test)
score_svm = accuracy_score(y_test, y_pred_svm)


# CROSS VALIDATION SVM
kfold = KFold(n_splits=10, random_state=seed)
c_v_svm = cross_val_score(svm.LinearSVC(), features_data, labels_data, cv=kfold, scoring='neg_mean_squared_error').mean()

print('Accuracy SVM:', score_svm, 'Cross validation SVM: ', c_v_svm)



