from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime as dt
import csv


def rfr_model(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, 15),
            'n_estimators': (10, 50, 100, 200, 500),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False)

    rfr.fit(X, y)

    return rfr

path = "/Users/Talha/Documents/Master (Big Data)/Semester 3/5001 - Foundations of Data Analytics/Individual Project/"
max_number_of_examples = 10000

columns = ['is_free', 'price', 'genres', 'categories', 'tags', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews']
labels = ['playtime_forever']

columns_train = pd.read_csv(path + "train.csv", usecols = columns, nrows=max_number_of_examples).fillna(0)
columns_train['purchase_date'] = (pd.to_datetime(columns_train['purchase_date']) - dt.datetime(1970,1,1)).dt.total_seconds()
columns_train['release_date'] = (pd.to_datetime(columns_train['release_date']) - dt.datetime(1970,1,1)).dt.total_seconds()
columns_train['purchase_date'] = columns_train['purchase_date'] - columns_train['purchase_date'].mean()
columns_train['release_date'] = columns_train['release_date'] - columns_train['release_date'].mean()
split_columns_train = columns_train['genres'].str.get_dummies(sep=',').join(columns_train['categories'].str.get_dummies(sep=','), lsuffix='_genre', rsuffix='_category').join(columns_train['tags'].str.get_dummies(sep=','), lsuffix='_notTag', rsuffix='_tag').join(columns_train[['is_free', 'price', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews']])
train_features_data = pd.get_dummies(split_columns_train)
train_features = train_features_data.columns
train_labels_data = pd.read_csv(path + "train.csv", usecols = labels, nrows=max_number_of_examples)

columns_test = pd.read_csv(path + "test.csv", usecols = columns, nrows=max_number_of_examples).fillna(0)
columns_test['purchase_date'] = (pd.to_datetime(columns_test['purchase_date']) - dt.datetime(1970,1,1)).dt.total_seconds()
columns_test['release_date'] = (pd.to_datetime(columns_test['release_date']) - dt.datetime(1970,1,1)).dt.total_seconds()
columns_test['purchase_date'] = columns_test['purchase_date'] - columns_test['purchase_date'].mean()
columns_test['release_date'] = columns_test['release_date'] - columns_test['release_date'].mean()
split_columns_test = columns_test['genres'].str.get_dummies(sep=',').join(columns_test['categories'].str.get_dummies(sep=','), lsuffix='_genre', rsuffix='_category').join(columns_test['tags'].str.get_dummies(sep=','), lsuffix='_notTag', rsuffix='_tag').join(columns_test[['is_free', 'price', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews']])
test_features_data = pd.get_dummies(split_columns_test)
test_features = test_features_data.columns

for train_feature in train_features:
    if train_feature not in test_features_data:
        test_features_data[train_feature] = 0

for test_feature in test_features:
    if test_feature not in train_features:
        test_features_data.drop(columns=[test_feature], inplace=True)
        print("dropped " + test_feature)
"""
rfr = rfr_model(train_features_data, train_labels_data)

predictions = rfr.predict(test_features_data)

with open('sample_submission.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id','playtime_forever'])
    id = 0
    for prediction in predictions:
        spamwriter.writerow([id, prediction])
        id+=1
"""
