import csv
import sys

#Standard data-sci libraries
import numpy as np
import pandas as pd

#SKLearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor, Lars, LassoLars,  ElasticNet, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn import metrics
from sklearn.svm import SVR

#XGBoost
import xgboost as xgb

#Dropping rows with outliers
def clean_outliers(df, col):
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    tolerance_threshold = 2
    low  = df[col].quantile(0.25) - tolerance_threshold*iqr
    high = df[col].quantile(0.75) + tolerance_threshold*iqr
    df_cleaned = df.loc[(df[col] > low) & (df[col] < high)]
    return df_cleaned

def rmse(a,b):
    return metrics.mean_squared_error(a,b)**0.5

model = "xgb"
print("default model is " + model)
training = False
if len(sys.argv) > 1:
    if sys.argv[1] in ("xgb", "knn", "rfr", "sgd"):
        model = sys.argv[1]
    else:
        print("Couldn't find the model you specified. Continuing with default model: "+ model)
    if len(sys.argv) > 2:
        try:
            training = bool(sys.argv[2])
        except:
            print("Couldn't parse the Training mode you specified. Continuing with default mode: " + str(training))

train_data = ("../train.csv")
test_data = ("../test.csv")

#Train data
df = pd.read_csv(train_data)

#cleaning and preprocessing train data
df = pd.read_csv(train_data)

#remove columns with more than a certain proprotion of missing values
missing_value_proportion = 0.75
df = df[df.columns[df.isnull().mean() < missing_value_proportion]]
df = df.loc[df.isnull().mean(axis=1) < missing_value_proportion]

#drop rows with NaNs
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

#replace booleans with ints
df.replace({False: 0, True: 1}, inplace=True)

df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["release_date"] = pd.to_datetime(df["release_date"]) 

df['release_year'] = pd.DatetimeIndex(df['release_date']).year
df['release_month'] = pd.DatetimeIndex(df['release_date']).month
df['release_day'] = pd.DatetimeIndex(df['release_date']).day
df['release_weekday'] = pd.DatetimeIndex(df['release_date']).dayofweek
df['purchase_year'] = pd.DatetimeIndex(df['purchase_date']).year
df['purchase_month'] = pd.DatetimeIndex(df['purchase_date']).month
df['purchase_day'] = pd.DatetimeIndex(df['purchase_date']).day
df['purchase_weekday'] = pd.DatetimeIndex(df['purchase_date']).dayofweek

df["purchase_release_diff"] = df.apply(lambda a: (a["purchase_date"] - a["release_date"]).days, axis=1)

df["purchase_release_diff_year"] = df["purchase_date"].dt.year - df["release_date"].dt.year

df.drop(columns=["purchase_date", "release_date"], inplace=True)

df["positive_ratio"] = df.apply(lambda a: a["total_positive_reviews"]/(a["total_negative_reviews"] + a["total_positive_reviews"])                                               if a["total_negative_reviews"] + a["total_positive_reviews"] != 0                                               else 0.5, axis=1)

df["negative_ratio"] = df.apply(lambda a: a["total_negative_reviews"]/(a["total_negative_reviews"] + a["total_positive_reviews"])                                               if a["total_negative_reviews"] + a["total_positive_reviews"] != 0                                               else 0.5, axis=1)

df["total_reviews"] = df.apply(lambda a: (a["total_negative_reviews"] + a["total_positive_reviews"]),                                               axis=1)

df.drop(columns=["total_positive_reviews", "total_negative_reviews"], inplace=True)

genres = df["genres"].str.get_dummies(",")
genres.columns = ['genre_' + str(col) for col in genres.columns]
categories = df["categories"].str.get_dummies(",")
categories.columns = ['category_' + str(col) for col in categories.columns]
tags = df["tags"].str.get_dummies(",")
tags.columns = ['tag_' + str(col) for col in tags.columns]
preprocessed_df = pd.concat([df, genres, categories, tags], axis=1)

preprocessed_df.drop(columns=["id", "is_free", "genres", "categories", "tags"], inplace=True)

preprocessed_df[["total_reviews", "price"]] = preprocessed_df[["total_reviews", "price"]].astype("int")

preprocessed_df = clean_outliers(preprocessed_df,'price')
preprocessed_df = preprocessed_df[preprocessed_df['playtime_forever'] < 60]
preprocessed_df.reset_index(inplace=True, drop=True)

#Test data
test_df = pd.read_csv(test_data, parse_dates = ['purchase_date', 'release_date'])

#cleaning and preprocessing test data
test_df["purchase_date"] = pd.to_datetime(test_df["purchase_date"])
test_df["release_date"] = pd.to_datetime(test_df["release_date"])

test_df['release_year'] = pd.DatetimeIndex(test_df['release_date']).year
test_df['release_month'] = pd.DatetimeIndex(test_df['release_date']).month
test_df['release_day'] = pd.DatetimeIndex(test_df['release_date']).day
test_df['release_weekday'] = pd.DatetimeIndex(test_df['release_date']).dayofweek
test_df['purchase_year'] = pd.DatetimeIndex(test_df['purchase_date']).year
test_df['purchase_month'] = pd.DatetimeIndex(test_df['purchase_date']).month
test_df['purchase_day'] = pd.DatetimeIndex(test_df['purchase_date']).day
test_df['purchase_weekday'] = pd.DatetimeIndex(test_df['purchase_date']).dayofweek

test_df["purchase_release_diff"] = test_df.apply(lambda a: (a["purchase_date"] - a["release_date"]).days, axis=1)

test_df["purchase_release_diff_years"] = test_df["purchase_date"].dt.year - test_df["release_date"].dt.year

test_df.drop(columns=["purchase_date", "release_date"], inplace=True)

test_df["purchase_release_diff_years"].fillna(test_df["purchase_release_diff_years"].median(), inplace=True)
test_df["purchase_release_diff"].fillna(test_df["purchase_release_diff"].median(), inplace=True)
test_df["total_positive_reviews"].fillna(test_df["total_positive_reviews"].median(), inplace=True)
test_df["total_negative_reviews"].fillna(test_df["total_negative_reviews"].median(), inplace=True)

test_df.reset_index(drop=True, inplace=True)

test_df.replace({False: 0, True: 1}, inplace=True)

test_df["positive_ratio"] = test_df.apply(lambda a: a["total_positive_reviews"]/(a["total_negative_reviews"] + a["total_positive_reviews"])                                               if a["total_negative_reviews"] + a["total_positive_reviews"] != 0                                               else 0.5, axis=1)


test_df["negative_ratio"] = test_df.apply(lambda a: a["total_negative_reviews"]/(a["total_negative_reviews"] + a["total_positive_reviews"])                                               if a["total_negative_reviews"] + a["total_positive_reviews"] != 0                                               else 0.5, axis=1)

test_df["total_reviews"] = test_df.apply(lambda a: (a["total_negative_reviews"] + a["total_positive_reviews"]),                                               axis=1)

test_df.drop(columns=["total_positive_reviews", "total_negative_reviews"], inplace=True)

genres_test = test_df["genres"].str.get_dummies(",")
genres_test.columns = ['genre_' + str(col) for col in genres_test.columns]
categories_test = test_df["categories"].str.get_dummies(",")
categories_test.columns = ['category_' + str(col) for col in categories_test.columns]
tags_test = test_df["tags"].str.get_dummies(",")
tags_test.columns = ['tag_' + str(col) for col in tags_test.columns]
preprocessed_test_df = pd.concat([test_df, genres_test, categories_test, tags_test], axis=1)

preprocessed_test_df.drop(columns=["id","is_free", "genres", "categories", "tags"], inplace=True)

preprocessed_test_df[["total_reviews", "price"]] = preprocessed_test_df[["total_reviews", "price"]].astype("int")


train_x = preprocessed_df.drop(['playtime_forever'], axis=1)
train_y = preprocessed_df['playtime_forever']
test_x = preprocessed_test_df

test_features = test_x.columns.tolist()
for train_feature in train_x.columns.tolist():
    if train_feature not in test_features:
        test_x[train_feature] = 0

train_features = train_x.columns.tolist()
for test_feature in test_x.columns.tolist():
    if test_feature not in train_features:
        test_x.drop(columns=[test_feature],inplace=True)


test_x.fillna(test_x.median(), inplace=True)

print(train_x.shape,test_x.shape,train_y.shape)

# Splitting the dataset

X_train, X_test, Y_train, Y_test = train_test_split(train_x,train_y, test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape,Y_test.shape,Y_train.shape)

mm_scaler = MinMaxScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test = mm_scaler.fit_transform(X_test)

def allmodels():
    classifiers = [
    AdaBoostRegressor(),
    BaggingRegressor(),
    ExtraTreesRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    PassiveAggressiveRegressor(),
    SGDRegressor(),
    TheilSenRegressor(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    ExtraTreeRegressor()
]
    names = [
    "AdaBoostRegressor",
    "BaggingRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
    "PassiveAggressiveRegressor",
    "SGDRegressor",
    "TheilSenRegressor",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "ExtraTreeRegressor"
]
    return classifiers,names
classifiers,names=allmodels()

if training:
    epochs=10
    for i in range(epochs):
        result=[]
        for classifier,name in zip(classifiers,names):
            classifier.fit(X_train, Y_train)
            a=classifier.predict(X_test)
            a[a<0]=0
            result.append(rmse(a,Y_test))
        model_result=pd.DataFrame(data=result,index=names,columns=['rmse']).sort_values(by="rmse" , ascending=True)
        print(model_result)

if (model == "xgb"):

    xgboost_model = xgb.XGBRegressor(learning_rate=0.05
                                     , max_depth=12, n_estimators=10, alpha=10, objective ='reg:linear', colsample_bytree = 0.3
                                    )
    xgboost_model.fit(X_train,Y_train)
    y_pred = xgboost_model.predict(X_test)
    print('rmse',metrics.mean_squared_error(Y_test,y_pred)**0.5)

    xgb_predictions = xgboost_model.predict(test_x.as_matrix())
    xgb_predictions[xgb_predictions<0]=0

    print("XGB", train_y.mean(), xgb_predictions.mean())

    predictions = xgb_predictions

if (model == "knn"):

    classifier=KNeighborsRegressor()
    classifier.fit(X_train,Y_train)
    predictions=classifier.predict(X_test)
    predictions[predictions<0]=0
    result.append(rmse(predictions,Y_test))

    #mm_scaler.fit_transform(test_x)
    knn_predictions=classifier.predict(test_x)
    knn_predictions[knn_predictions<0]=0

    print("KNN", train_y.mean(),knn_predictions.mean())

    predictions = knn_predictions

if (model == "sgd"):

    classifier=SGDRegressor()
    classifier.fit(X_train,Y_train)
    predictions=classifier.predict(X_test)
    predictions[predictions<0]=0
    result.append(rmse(predictions,Y_test))
    print("SGD", rmse(predictions,Y_test))

    mm_scaler.fit_transform(test_x)
    sgd_predictions=classifier.predict(test_x.as_matrix())
    sgd_predictions[sgd_predictions<0]=0

    predictions = sgd_predictions

def rfr_model(X, y, max_depth_gsc):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, max_depth_gsc),
            'n_estimators': (10, 50, 100, 200, 500),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False, max_features="sqrt")

    rfr.fit(X, y)

    return rfr

if (model == "rfr"):

    max_depth_gsc = 15

    rfr = rfr_model(train_x, train_y, max_depth_gsc)

    rfr_predictions = rfr.predict(test_x)
    rfr_predictions[rfr_predictions<0]=0

    print("RFR", train_y.mean(),rfr_predictions.mean())

    predictions=rfr_predictions

with open('sample_submission.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id','playtime_forever'])
    id = 0
    for prediction in predictions:
        spamwriter.writerow([id, prediction])
        id+=1

