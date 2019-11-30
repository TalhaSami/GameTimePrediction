# Load libraries
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Set random seed
np.random.seed(0)

path = "/Users/Talha/Documents/Master (Big Data)/Semester 3/5001 - Foundations of Data Analytics/Individual Project/"
max_number_of_examples = 10000

columns = ['is_free', 'price', 'genres', 'categories', 'tags', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews']
labels = ['playtime_forever']

columns_train = pd.read_csv(path + "train.csv", usecols = columns, nrows=max_number_of_examples).fillna(0)
columns_train['purchase_date'] = pd.to_datetime(columns_train['purchase_date']).dt.strftime('%Y%m%d').astype(int)
columns_train['release_date'] = pd.to_datetime(columns_train['release_date']).dt.strftime('%Y%m%d').astype(int)
split_columns_train = columns_train['genres'].str.get_dummies(sep=',').join(columns_train['categories'].str.get_dummies(sep=','), lsuffix='_genre', rsuffix='_category').join(columns_train['tags'].str.get_dummies(sep=','), lsuffix='_notTag', rsuffix='_tag').join(columns_train[['is_free', 'price', 'purchase_date', 'release_date', 'total_positive_reviews', 'total_negative_reviews']])
train_features_data = pd.get_dummies(split_columns_train)
train_features = train_features_data.columns
train_labels_data = pd.read_csv(path + "train.csv", usecols = labels, nrows=max_number_of_examples)

train_features_data, val_features_data, train_labels_data, val_labels_data = train_test_split(train_features_data,
                                                                            train_labels_data,
                                                                            test_size=0.33,
                                                                            random_state=0)

columns_test = pd.read_csv(path + "test.csv", usecols = columns, nrows=max_number_of_examples).fillna(0)
columns_test['purchase_date'] = pd.to_datetime(columns_test['purchase_date']).dt.strftime('%Y%m%d').astype(int)
columns_test['release_date'] = pd.to_datetime(columns_test['release_date']).dt.strftime('%Y%m%d').astype(int)
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


# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=32, activation='relu', input_shape=(train_features_data.shape[1],)))

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=32, activation='relu'))

# Add fully connected layer with no activation function
network.add(layers.Dense(units=1))


# Compile neural network
network.compile(loss='mse', # Mean squared error
                optimizer='RMSprop', # Optimization algorithm
                metrics=['mse']) # Mean squared error

# Train neural network
history = network.fit(train_features_data.to_numpy(), # Features
                      train_labels_data.to_numpy(), # Target vector
                      epochs=100000, # Number of epochs
                      verbose=0, # No output
                      batch_size=100, # Number of observations per batch
                      validation_data=(val_features_data, val_labels_data)) # Data for evaluation

print(network.predict(test_features_data.to_numpy()))