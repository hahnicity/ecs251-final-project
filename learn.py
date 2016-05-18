# First try .6335
# With scaling: .6559.
# With stacking 10 breaths .6676 and 33% test set
# With stacking 20 breaths .67855 and 25% test set
# After removing a bunch of pressure features .6627
# Return 'TVe:TVi ratio' .6635
# Remove all pressure features and RR. Only use minF_to_zero .62523
# Remove all stacking; only use minF_to_zero .6278
# Return TVe:TVi ratio, no stacking: .62748
# Remove TVe:TVi, return all pressure data, no stacking: .65504
# Keep pressure data, stacking to 30: .6724
# Remove pressure data, stacking @ 1, Return I:E ratio and eTime: .6291
# Remove eTime: .627
# Add in eTime IE ratio and all pressure data: .7128
# Remove TVi:TVe ratio .7139
# 20 stacked rows with 10000 training rows and 2500 test rows: .7104
# 30 stacked rows with 10000 training rows and 2500 test rows: .6984
# 40 stacked rows with 10000 training rows and 2500 test rows: .6932
# 50 stacked rows with 10000 training rows and 2500 test rows: .6864
# 10 stacked rows with 10000 training rows and 2500 test rows: .7008
# 15 stacked rows with 10000 training rows and 2500 test rows: .704
# 18 stacked rows with 10000 training rows and 2500 test rows: .6776
# 22 stacked rows with 10000 training rows and 2500 test rows: .7096
# 25 stacked rows with 10000 training rows and 2500 test rows: .7276
# Put iTime back in with 25 stacked breaths and all data: .71316
# Remove iTime with 25 stacked breaths and all data: .704
# Put iTime back in with 20 stacked breaths and all data: .7119
# Remove iTime with 20 stacked breaths and all data: .7139
# Use SVM instead of linear regression: .84886
from argparse import ArgumentParser

from numpy import inf, nan
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC

from collate import collate_all_from_breath_meta_to_data_frame

CACHE_SIZE = 5000


def preprocess_x_y(df):
    y = df['y']
    del df['y']
    df = scale(df)
    return df, y


def non_spark(x_train, x_test, y_train, y_test):
    # TODO perform PCA on whole thing.
    clf = SVC(cache_size=CACHE_SIZE)
    param_grid = {"C": list(range(1, 11))}
    gs = GridSearchCV(clf, param_grid)
    res = gs.fit(x_train, y_train)
    print(res.best_score_)
    print(res.best_params_)


def with_spark(x_train, x_test, y_train, y_test):
    from pyspark import SparkConf, SparkContext
    from spark_sklearn import GridSearchCV as SparkGridSearchCV
    conf = SparkConf().setMaster("local").setAppName("ecs251")
    sc = SparkContext(conf=conf)
    param_grid = {"C": [1, 10]}
    gs = SparkGridSearchCV(sc, SVC(cache_size=CACHE_SIZE), param_grid=param_grid)
    res = gs.fit(x_train, y_train)
    print(res.best_score_)
    print(res.best_params_)


def main():
    parser = ArgumentParser()
    parser.add_argument("--with-spark", action="store_true", default=False)
    args = parser.parse_args()
    df = collate_all_from_breath_meta_to_data_frame(20)
    x, y = preprocess_x_y(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
    if args.with_spark:
        with_spark(x_train, x_test, y_train, y_test)
    else:
        non_spark(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
