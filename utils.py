import time
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report


def select_binary_classes(X, y):
    mask = np.logical_or(y == 0, y == 1)
    return X[mask], y[mask]


def select_model(estimator, param, X_train, y_train):
    time_start = time.time()
    clf = GridSearchCV(estimator, param, scoring='f1_micro', cv=5, n_jobs=4)
    clf.fit(X_train, y_train)
    time_end = time.time()

    df = pd.DataFrame(clf.cv_results_)
    df.to_csv('result.csv')
    print('best param:', clf.best_params_)
    print('best_score:', clf.best_score_)
    print('Time consumed: %.2f seconds' % (time_end - time_start))


def validate_model(clf, X_train, y_train):
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2)
    time_start = time.time()
    clf.fit(X_train, y_train)
    time_end = time.time()

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    print('acc %.4f' % accuracy_score(y_test, pred))
    print('f1 %.4f' % f1_score(y_test, pred, average='micro'))
    print('Time consumed: %.2f seconds' % (time_end - time_start))


def cross_validate_model(clf, X, y, cv=5):
    time_start = time.time()
    scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=cv)
    time_end = time.time()

    print('mean f1 %.4f' % np.mean(scores))
    print('Time consumed: %.2f seconds' % (time_end - time_start))


def train_and_test(clf, X_train, y_train, X_test, result_file):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    save_result(result_file, pred)


RESULTS_DIR = 'results'


def save_result(file, y_pred):
    with open(os.path.join(RESULTS_DIR, file), 'w', encoding='utf-8') as f:
        f.write('ID,Expected\n')
        for i in range(len(y_pred)):
            f.write('%d,%d\n' % (i, y_pred[i]))


def load_result(file):
    df = pd.read_csv(os.path.join(RESULTS_DIR, file))
    return df['Expected'].values
