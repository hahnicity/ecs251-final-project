from numpy import inf, nan
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

from collate import collate_all_from_breath_meta_to_data_frame


def preprocess_x_y(df):
    y = df['y']
    del df['y']
    df = scale(df)
    return df, y


def main():
    # TODO perform PCA on whole thing.
    df = collate_all_from_breath_meta_to_data_frame(20)
    x, y = preprocess_x_y(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
    clf = LogisticRegression()
    clf = clf.fit(x_train, y_train)
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
    score = clf.score(x_test, y_test)
    print(score)


if __name__ == "__main__":
    main()
