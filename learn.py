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
# Use SVM C=50, gamma=0.02, randomly sampled 10000 items from 20000 items: .9396
# SVM C=10 gamma=0.02, reindexed, full set of 86000 items: 0.9014
from argparse import ArgumentParser
import csv
from random import randint

from numpy import append, inf, nan
from numpy.random import permutation
from pandas import DataFrame
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import scale
from sklearn.svm import SVC

from collate import collate_all_from_breath_meta_to_data_frame

CACHE_SIZE = 1024
C = range(5, 55, 5)
GAMMA = [.02]

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
    return DataFrame(df), DataFrame(y), vents_and_files


def perform_initial_scaling(df, stacked_breaths):
    """
    Since breaths are stacked we look at max and mins across multiple
    stacked rows
    """
    max_mins = {}
    for col in range(df.shape[1]):
        modulo_idx = col % stacked_breaths
        max_mins.setdefault(modulo_idx, {'max': 0, 'min': 0})
        min = df.iloc[:, col].min()
        if min < max_mins[modulo_idx]['max']:
            max_mins[modulo_idx]['min'] = min
        max = df.iloc[:, col].max()
        if max > max_mins[modulo_idx]['min']:
            max_mins[modulo_idx]['max'] = max
    perform_subsequent_scaling(df, max_mins)
    return df, max_mins


def perform_subsequent_scaling(df, max_mins):
    for col in range(df.shape[1]):
        breaths_to_stack = len(max_mins)
        modulo_idx = col % breaths_to_stack
        val = max_mins[modulo_idx]
        df.iloc[:, col] = (df.iloc[:, col] - val['min']) / (val['max'] - val['min'])
    return df


def non_spark(x_train, x_test, y_train, y_test, vents_and_files):
    for c in [12]:
        for gamma in [.02]:
            clf = SVC(cache_size=CACHE_SIZE, kernel="rbf", C=c, gamma=gamma)
            clf.fit(x_train, y_train['y'].values)
            print("Params: ", c, gamma)
            predictions = clf.predict(x_test)
            print("Accuracy: " + str(accuracy_score(y_test['y'], predictions)))
            print("Precision: " + str(precision_score(y_test['y'], predictions)))
            print("Recall: " + str(recall_score(y_test['y'], predictions)))
            fpr, tpr, thresh = roc_curve(y_test['y'], predictions)
            print("False pos rate: " + str(fpr[1]))
            print("True post rate: " + str(tpr[1]))
            error = abs(y_test['y'] - predictions)
            failure_idx = error[error == 2]
            with open("failure.c{}.gam{}.test".format(c, gamma), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "ventbn", "real val"])
                for idx in failure_idx.index:
                    pt_data = vents_and_files[idx]
                    actual = y_test.loc[idx].values
                    writer.writerow(pt_data + list(actual))


def with_spark(x_train, x_test, y_train, y_test, vents_and_files, spark_connect_str):
    from pyspark import SparkConf, SparkContext
    from spark_sklearn import GridSearchCV as SparkGridSearchCV
    conf = SparkConf().setMaster(spark_connect_str).setAppName("ecs251")
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.deploy.mode", "cluster")
    sc = SparkContext(conf=conf)
    param_grid = {"C": C, "gamma": GAMMA}
    gs = SparkGridSearchCV(sc, SVC(cache_size=CACHE_SIZE), param_grid=param_grid)
    res = gs.fit(x_train, y_train['y'].values)
    print("Best Score: ", res.best_score_)
    print("Best params: ", res.best_params_)
    predictions = res.predict(x_test)
    print("Accuracy: " + str(accuracy_score(y_test['y'], predictions)))
    print("Precision: " + str(precision_score(y_test['y'], predictions)))
    print("Recall: " + str(recall_score(y_test['y'], predictions)))
    fpr, tpr, thresh = roc_curve(y_test['y'], predictions)
    print("False pos rate: " + str(fpr[1]))
    print("True post rate: " + str(tpr[1]))


def main():
    parser = ArgumentParser()
    parser.add_argument("--with-spark", action="store_true", default=False)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--connect-str", default="local", help="The master connect str for spark")
    args = parser.parse_args()
    breaths_to_stack = 20
    df = collate_all_from_breath_meta_to_data_frame(breaths_to_stack, args.samples)
    x, y, vents_and_files = preprocess_x_y(df)
    if args.samples:
        x = x.sample(n=args.samples)
        y = y.loc[x.index]
    # Reindex to ensure we don't bias the results
    x = x.reindex(permutation(x.index))
    y = y.loc[x.index]
    print("{} positive samples".format(len(y[y['y'] == 1])))
    print("{} negative samples".format(len(y[y['y'] == -1])))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=randint(0, 100)
    )
    x_train, scaling_factors = perform_initial_scaling(x_train, breaths_to_stack)
    x_test = perform_subsequent_scaling(x_test, scaling_factors)
    if args.with_spark:
        with_spark(x_train, x_test, y_train, y_test, vents_and_files, args.connect_str)
    else:
        non_spark(x_train, x_test, y_train, y_test, vents_and_files)


if __name__ == "__main__":
    main()
