import argparse

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

from preprocess import load_train_tfidfdim, load_test_tfidfdim, load_target
from utils import select_model, validate_model, train_and_test, \
    cross_validate_model, select_binary_classes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test by learner svm')
    parser.add_argument('cmd', choices=['val', 'cv', 'pred'],
                        help='sub-commands')
    parser.add_argument('-c', '--C', type=float, default=1.0,
                        help='alpha for multinomial naive bayes')
    parser.add_argument('-f', '--fold', type=int, default=5,
                        help='the number of folds for cross validation')
    parser.add_argument('-o', '--output', default='lsvm_result.csv',
                        help='the location of prediction result')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='set verbosity')
    return parser.parse_args()


def val(args):
    X_train, y_train = load_train_tfidfdim(), load_target()
    svm = LinearSVC(C=args.C, verbose=args.verbose)
    clf = OneVsOneClassifier(svm)
    validate_model(clf, X_train, y_train)


def cv(args):
    X_train, y_train = load_train_tfidfdim(), load_target()
    svm = LinearSVC(C=args.C, verbose=args.verbose)
    clf = OneVsOneClassifier(svm)
    cross_validate_model(clf, X_train, y_train, cv=args.fold)


def pred(args):
    """
    dataset: tfidfdim
    model: LinearSVC(C=1.0), ovo
    cross validate f1-score: 0.1723
    test f1-score: 0.16922
    """
    X_train, y_train = load_train_tfidfdim(), load_target()
    X_test = load_test_tfidfdim()
    svm = LinearSVC(C=args.C)
    clf = OneVsOneClassifier(svm)
    train_and_test(clf, X_train, y_train, X_test, args.output)


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == 'val':
        val(args)
    elif args.cmd == 'cv':
        cv(args)
    elif args.cmd == 'pred':
        pred(args)
