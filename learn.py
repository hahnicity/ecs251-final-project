# First try with Logistic Regression .6335
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
# Use SVM with C=10 .862928
# Use SVM with C=50 .87754
# Use SVM with C=50 gamma=0.01 .8886
# Use SVM with C=50 gamma=0.02 .89335
from argparse import ArgumentParser
import csv
from multiprocessing import cpu_count
from random import randint

from numpy import append, inf, nan
from pandas import DataFrame
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.preprocessing import scale
from sklearn.svm import SVC

from collate import collate_all_from_breath_meta_to_data_frame

CACHE_SIZE = 1024


def preprocess_x_y(df):
    y = df['y']
    del df['y']
    filenames = df['filename']
    del df['filename']
    start_vent_bns = df['start_vent_bn']
    del df['start_vent_bn']
    # Need to sleep for a second because of some race condition
    import time; time.sleep(1)
    tmp_zip = zip(list(df.index), filenames, start_vent_bns)
    vents_and_files = {idx: [f, bn] for idx, f, bn in tmp_zip}
    df = scale(df)
    return DataFrame(df), DataFrame(y), vents_and_files


def non_spark(x_train, x_test, y_train, y_test, vents_and_files):
    # TODO perform PCA on whole thing.
    for c in [40, 45, 50, 55, 60]:
        for gamma in [.01, .015, .02, .025, .03]:
            clf = SVC(cache_size=CACHE_SIZE, kernel="rbf", C=c, gamma=gamma)
            clf.fit(x_train, y_train)
            print(c, gamma)
            print(clf.score(x_test, y_test))
            predictions = clf.predict(x_test)
            print("Precision: " + str(precision_score(y_test['y'], predictions)))
            print("Recall: " + str(recall_score(y_test['y'], predictions)))
            fpr, tpr, thresh = roc_curve(y_test['y'], predictions)
            print("False pos rate: " + str(fpr[1]))
            print("True post rate: " + str(tpr[1]))
            error = abs(y_test['y'] - predictions)
            failure_idx = error[error == 2]
            with open("failure.test", "w") as f:
                writer = csv.writer(f)
                for idx in failure_idx.index:
                    pt_data = vents_and_files[idx]
                    pt_data = str(pt_data[0]) + " " + str(pt_data[1])
                    actual = y_test.loc[idx].values
                    writer.writerow(pt_data + str(list(actual)))


def with_spark(x_train, x_test, y_train, y_test, vents_and_files, spark_connect_str):
    from pyspark import SparkConf, SparkContext
    from spark_sklearn import GridSearchCV as SparkGridSearchCV
    conf = SparkConf().setMaster(spark_connect_str).setAppName("ecs251")
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.deploy.mode", "cluster")
    sc = SparkContext(conf=conf)
    param_grid = {"C": [40, 45, 50, 55], "gamma": [.01, .02, .03, .04, .05]}
    gs = SparkGridSearchCV(sc, SVC(cache_size=CACHE_SIZE), param_grid=param_grid)
    res = gs.fit(x_train, y_train['y'].values)
    print(res.best_score_)
    print(res.best_params_)
    print("Perform scoring on test set")
    print(res.score(x_test, y_test))


def main():
    parser = ArgumentParser()
    parser.add_argument("--with-spark", action="store_true", default=False)
    parser.add_argument("--train-subset", type=int, default=20000)
    parser.add_argument("--test-subset", type=int, default=10000)
    parser.add_argument("--connect-str", default="local", help="The master connect str for spark")
    args = parser.parse_args()
    df = collate_all_from_breath_meta_to_data_frame(20)
    x, y, vents_and_files = preprocess_x_y(df)
    print(x.info())
    print(y.info())

    for train_idx, test_idx in KFold(n=len(y['y']), n_folds=5):
        x_train = x.iloc[train_idx].sample(n=args.train_subset)
        y_train = y.iloc[train_idx].sample(n=args.train_subset)
        x_test = x.iloc[test_idx].sample(n=args.test_subset)
        y_test = y.iloc[test_idx].sample(n=args.test_subset)
        if args.with_spark:
            with_spark(x_train, x_test, y_train, y_test, vents_and_files, args.conn)
        else:
            non_spark(x_train, x_test, y_train, y_test, vents_and_files)


if __name__ == "__main__":
    main()
