import argparse

from sklearn.naive_bayes import MultinomialNB

from preprocess import load_train_tfidf, load_test_tfidf, load_target
from utils import select_model, validate_model, cross_validate_model, \
    train_and_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test by multinomial naive bayes')
    parser.add_argument('cmd', choices=['val', 'cv', 'pred'],
                        help='sub-commands')
    parser.add_argument('-a', '--alpha', type=float, default=0.05,
                        help='alpha for multinomial naive bayes')
    parser.add_argument('-f', '--fold', type=int, default=5,
                        help='the number of folds for cross validation')
    parser.add_argument('-o', '--output', default='nb_result.csv',
                        help='the location of prediction result')
    return parser.parse_args()


def val(args):
    X_train, y_train = load_train_tfidf(), load_target()
    clf = MultinomialNB(alpha=args.alpha)
    validate_model(clf, X_train, y_train)


def cv(args):
    X_train, y_train = load_train_tfidf(), load_target()
    clf = MultinomialNB(alpha=args.alpha)
    cross_validate_model(clf, X_train, y_train, cv=args.fold)


def pred(args):
    """
    dataset: tfidf
    model: MultinomialNB(alpha=0.05)
    cross validate f1-score: 0.1613
    test f1-score: 0.16181
    """
    X_train, y_train = load_train_tfidf(), load_target()
    X_test = load_test_tfidf()
    clf = MultinomialNB(alpha=args.alpha)
    train_and_test(clf, X_train, y_train, X_test, args.output)


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == 'val':
        val(args)
    elif args.cmd == 'cv':
        cv(args)
    elif args.cmd == 'pred':
        pred(args)
