import json
from random import randint

from flask import Flask, request
from numpy.random import permutation
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC

from collate import collate_all_from_breath_meta_to_data_frame
from learn import preprocess_x_y, perform_initial_scaling, perform_subsequent_scaling
from sms import send_text

TEST_FRACTION = 0.02

app = Flask(__name__)
# Declare global svm with optimal params
clf = SVC(cache_size=1024, C=10, gamma=0.02)


def get_data():
    to_stack = 20
    samples = None
    df = collate_all_from_breath_meta_to_data_frame(to_stack, samples)
    x, y, vents_and_files = preprocess_x_y(df)
    if samples:
        x = x.sample(n=samples)
        y = y.loc[x.index]
    # Reindex to ensure we don't bias the results
    x = x.reindex(permutation(x.index))
    y = y.loc[x.index]
    print("{} positive samples".format(len(y[y['y'] == 1])))
    print("{} negative samples".format(len(y[y['y'] == -1])))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=randint(0, 100)
    )
    x_train, scaling_factors = perform_initial_scaling(x_train, to_stack)
    x_test = perform_subsequent_scaling(x_test, scaling_factors)
    return x_train, x_test, y_train, y_test, scaling_factors


def train(x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train['y'].values)
    predictions = clf.predict(x_test)
    print("Accuracy: " + str(accuracy_score(y_test['y'], predictions)))
    print("Precision: " + str(precision_score(y_test['y'], predictions)))
    print("Recall: " + str(recall_score(y_test['y'], predictions)))
    fpr, tpr, thresh = roc_curve(y_test['y'], predictions)
    print("False pos rate: " + str(fpr[1]))
    print("True post rate: " + str(tpr[1]))


@app.route('/analyze/', methods=["POST"])
def analyze():
    data = json.loads(request.data)
    breath_data = data["breath_data"]
    patient_id = data["patient_id"]
    df = perform_subsequent_scaling(DataFrame([breath_data]), scaling_factors)
    prediction = clf.predict(df)
    print("Prediction was: ", prediction)
    if prediction[0] == 1:
        #send_text("+19083274527", "+15102543918", "Patient {} has ARDS".format(patient_id))
        pass
    return str(prediction[0])


x_train, x_test, y_train, y_test, scaling_factors = get_data()
train(x_train, x_test, y_train, y_test)
app.run(debug=False, port=80, host="0.0.0.0")
