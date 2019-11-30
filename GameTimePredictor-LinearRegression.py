import pandas as pd
import datetime as dt
import csv
from sklearn.linear_model import LinearRegression

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


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(train_features_data, train_labels_data)  # perform linear regression
predictions = linear_regressor.predict(test_features_data)  # make predictions

print(train_labels_data.to_numpy())
print(predictions)

"""
with open('sample_submission.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id','playtime_forever'])
    id = 0
    for prediction in predictions:
        spamwriter.writerow([id, prediction])
        id+=1
"""