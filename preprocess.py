import argparse
import os
import re
import time
import random
import pickle

import numpy as np
import jieba
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import torch
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# --------------------------------------------- #
#              Raw data (read only)
# --------------------------------------------- #

EMOJI_DATA_PATH = 'dataset/emoji.data'
TRAIN_DATA_PATH = 'dataset/train.data'
TRAIN_SOLUTION_PATH = 'dataset/train.solution'
TEST_DATA_PATH = 'dataset/test.data'
PRETRAINED_EMBEDDINGS_PATH = 'dataset/sgns.weibo.bigram'


def _check_raw_data_existence(file):
    if not os.path.exists(file):
        print('Error: {} doesn\'t exist. Please unzip raw data'.format(file))
        exit(1)


def open_emoji_data():
    _check_raw_data_existence(EMOJI_DATA_PATH)
    return open(EMOJI_DATA_PATH, 'r', encoding='utf-8')


def open_train_data():
    _check_raw_data_existence(TRAIN_DATA_PATH)
    return open(TRAIN_DATA_PATH, 'r', encoding='utf-8')


def open_train_solution():
    _check_raw_data_existence(TRAIN_SOLUTION_PATH)
    return open(TRAIN_SOLUTION_PATH, 'r', encoding='utf-8')


def open_test_data():
    _check_raw_data_existence(TEST_DATA_PATH)
    return open(TEST_DATA_PATH, 'r', encoding='utf-8')


def open_pretrained_embeddings():
    if not os.path.exists(PRETRAINED_EMBEDDINGS_PATH):
        print('Error: ' + PRETRAINED_EMBEDDINGS_PATH + ' doesn\'t exist.'
              ' Please download pretrained embeddings from '
              'https://github.com/Embedding/Chinese-Word-Vectors')
        exit(1)
    return open(PRETRAINED_EMBEDDINGS_PATH, 'r', encoding='utf-8')


# --------------------------------------------- #
#            Target in form of npz
# --------------------------------------------- #

TARGET_PATH = 'dataset/target.npz'


def _prepare_target():
    """Transform {EMOJI_DATA, TRAIN_SOLUTION} to {TARGET}"""
    print('Preparing target.')
    emoji_map = {}
    with open_emoji_data() as f:
        for line in f:
            s = line.split()
            emoji_map[s[1]] = int(s[0])

    target = []
    pattern = re.compile(r'{(.*)}')
    with open_train_solution() as f:
        for line in f:
            emoji_name = pattern.search(line).group(1)
            target.append(emoji_map[emoji_name])
    np.savez(TARGET_PATH, target=target)


def load_target():
    """Load {TARGET}"""
    if not os.path.exists(TARGET_PATH):
        _prepare_target()
    return np.load(TARGET_PATH)['target']


# --------------------------------------------- #
#                 Stop words
# --------------------------------------------- #

STOP_WORDS_PATH = 'dataset/stop_words.txt'


def load_stop_words():
    stop_words = set(['\t', '\n'])
    if not os.path.exists(STOP_WORDS_PATH):
        return stop_words

    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
        stop_words.update(f.read().splitlines())
    return stop_words


# --------------------------------------------- #
#               Word segmentaion
# --------------------------------------------- #

TRAIN_SEG_PATH = 'dataset/train.seg'
TEST_SEG_PATH = 'dataset/test.seg'


def _segment_stream(istream, ostream):
    stop_words = load_stop_words()
    pattern = re.compile(r'[_a-zA-Z0-9]')
    for line in istream:
        words = [seg for seg in jieba.cut(line)
                 if seg not in stop_words and not pattern.search(seg)]
        ostream.write(' '.join(words))
        ostream.write('\n')


def _prepare_train_seg():
    print('Preparing word segmentaion for train data.')
    with open_train_data() as src:
        with open(TRAIN_SEG_PATH, 'w', encoding='utf-8') as dst:
            _segment_stream(src, dst)


def _prepare_test_seg():
    print('Preparing word segmentaion for test data.')
    with open_test_data() as src:
        with open(TEST_SEG_PATH, 'w', encoding='utf-8') as dst:
            _segment_stream(src, dst)


def open_train_seg():
    if not os.path.exists(TRAIN_SEG_PATH):
        _prepare_train_seg()
    return open(TRAIN_SEG_PATH, 'r', encoding='utf-8')


def open_test_seg():
    if not os.path.exists(TEST_SEG_PATH):
        _prepare_test_seg()
    return open(TEST_SEG_PATH, 'r', encoding='utf-8')


# --------------------------------------------- #
#            Tfidf-vectorized data
# --------------------------------------------- #

FEATURE_NAMES_PATH = 'dataset/feature_names.txt'
TRAIN_TFIDF_PATH = 'dataset/train_tfidf.npz'
TEST_TFIDF_PATH = 'dataset/test_tfidf.npz'


def _prepare_tfidf_data():
    print('Preparing tfidf-vectorized data.')
    vectorizer = TfidfVectorizer(
        encoding='utf-8', token_pattern=r'(?u)\b\w+\b')
    with open_train_seg() as f:
        X_train = vectorizer.fit_transform(f)
    with open_test_seg() as f:
        X_test = vectorizer.transform(f)
    np.savetxt(FEATURE_NAMES_PATH, vectorizer.get_feature_names(),
               fmt='%s', encoding='utf-8')
    save_npz(TRAIN_TFIDF_PATH, X_train)
    save_npz(TEST_TFIDF_PATH, X_test)


def load_train_tfidf():
    if not os.path.exists(TRAIN_TFIDF_PATH):
        _prepare_tfidf_data()
    return load_npz(TRAIN_TFIDF_PATH)


def load_test_tfidf():
    if not os.path.exists(TEST_TFIDF_PATH):
        _prepare_tfidf_data()
    return load_npz(TEST_TFIDF_PATH)


# --------------------------------------------- #
#   Feature-selected tfidfdim-vectorized data
# --------------------------------------------- #

TRAIN_TFIDFIDM_PATH = 'dataset/train_tfidfidm.npz'
TEST_TFIDFIDM_PATH = 'dataset/test_tfidfidm.npz'


def _prepare_tfidfdim_data(dim=50000):
    print('Preparing feature-selected tfidfdim-vectorized data.')
    X_train = load_train_tfidf()
    X_test = load_test_tfidf()
    y_train = load_target()

    selector = SelectKBest(chi2, dim)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    save_npz(TRAIN_TFIDFIDM_PATH, X_train)
    save_npz(TEST_TFIDFIDM_PATH, X_test)


def load_train_tfidfdim():
    if not os.path.exists(TRAIN_TFIDFIDM_PATH):
        _prepare_tfidfdim_data()
    return load_npz(TRAIN_TFIDFIDM_PATH)


def load_test_tfidfdim():
    if not os.path.exists(TEST_TFIDFIDM_PATH):
        _prepare_tfidfdim_data()
    return load_npz(TEST_TFIDFIDM_PATH)


# --------------------------------------------- #
#           Filtered segmented data
# --------------------------------------------- #

FILTERED_TRAIN_SEG_PATH = 'dataset/train.filtered.seg'
FILTERED_TEST_SEG_PATH = 'dataset/test.filtered.seg'


def _vocabulary_in_pretrained_embeddings():
    with open_pretrained_embeddings() as f:
        return set([line.split()[0] for line in f])


def _filter_seg(istream, ostream, whitelist):
    for line in istream:
        words_to_keep = [
            word for word in line.split() if word in whitelist]
        ostream.write(' '.join(words_to_keep))
        ostream.write('\n')


def _perpare_filtered_seg():
    print('Preparing filtered segmented file.')
    whitelist = _vocabulary_in_pretrained_embeddings()
    with open_train_seg() as src:
        with open(FILTERED_TRAIN_SEG_PATH, 'w', encoding='utf-8') as dst:
            _filter_seg(src, dst, whitelist)
    with open_test_seg() as src:
        with open(FILTERED_TEST_SEG_PATH, 'w', encoding='utf-8') as dst:
            _filter_seg(src, dst, whitelist)


def open_filtered_train_seg():
    if not os.path.exists(FILTERED_TRAIN_SEG_PATH):
        _perpare_filtered_seg()
    return open(FILTERED_TRAIN_SEG_PATH, 'r', encoding='utf-8')


def open_filtered_test_seg():
    if not os.path.exists(FILTERED_TEST_SEG_PATH):
        _perpare_filtered_seg()
    return open(FILTERED_TEST_SEG_PATH, 'r', encoding='utf-8')


# --------------------------------------------- #
#        Vocabulary and word embeddings
# --------------------------------------------- #

VACABULARY_PATH = 'dataset/vocalbulary.pkl'


def _add_vocabulary(vocabulary, istream):
    for line in istream:
        for word in line.split():
            vocabulary.setdefault(word, len(vocabulary) + 1)


def _prepare_vacabulary():
    """Vacabulary is a map from words to row indices of embeddings matrix."""
    print('Preparing vacabulary.')
    vocabulary = {}
    with open_filtered_train_seg() as f:
        _add_vocabulary(vocabulary, f)
    with open_filtered_test_seg() as f:
        _add_vocabulary(vocabulary, f)
    with open(VACABULARY_PATH, 'wb') as f:
        pickle.dump(vocabulary, f)
    print('Build a vacabulary with %d words.' % len(vocabulary))


def load_vocabulary():
    if not os.path.exists(VACABULARY_PATH):
        _prepare_vacabulary()
    with open(VACABULARY_PATH, 'rb') as f:
        return pickle.load(f)


EMBEDDINGS_PATH = 'dataset/embeddings.npz'


def _prepare_embeddings():
    print('Preparing embeddings from the pretrained one.')
    vocabulary = load_vocabulary()
    embeddings = np.random.randn(len(vocabulary) + 1, 300)
    with open_pretrained_embeddings() as f:
        ctr = 0
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if word in vocabulary:
                vector = np.array(list(map(float, tokens[1:])))
                embeddings[vocabulary[word]] = vector
                ctr += 1
    np.savez(EMBEDDINGS_PATH, embeddings=embeddings)
    print('Reused %d existing embeddings.' % ctr)


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        _prepare_embeddings()
    return np.load(EMBEDDINGS_PATH)['embeddings']


def load_embeddings_as_tensor():
    return torch.FloatTensor(load_embeddings())


# --------------------------------------------- #
#               Dataset for cnn
# --------------------------------------------- #

CNN_TRAINSET_PATH = 'dataset/cnn_trainset.npz'
CNN_VALSET_PATH = 'dataset/cnn_valset.npz'
CNN_TESTSET_PATH = 'dataset/cnn_testset.npz'


def _seg_to_index(istream, dim):
    vocabulary = load_vocabulary()
    index_mat = []
    for line in istream:
        indices = [vocabulary[word] for word in line.split()]
        if len(indices) < dim:
            indices += [0] * (dim - len(indices))
        elif len(indices) > dim:
            random.shuffle(indices)
            indices = indices[:dim]
        index_mat.append(indices)
    return np.array(index_mat)


def _prepare_cnn_data(validate_size=100000):
    with open_filtered_train_seg() as f:
        train_index = _seg_to_index(f, 30)
    with open_filtered_test_seg() as f:
        test_index = _seg_to_index(f, 30)
    target = load_target()
    train_index, val_index, train_target, val_target = train_test_split(
        train_index, target, test_size=validate_size)
    np.savez(CNN_TRAINSET_PATH, X=train_index, y=train_target)
    np.savez(CNN_VALSET_PATH, X=val_index, y=val_target)
    np.savez(CNN_TESTSET_PATH, X=test_index)


def load_cnn_trainset():
    if not os.path.exists(CNN_TRAINSET_PATH):
        _prepare_cnn_data()
    data = np.load(CNN_TRAINSET_PATH)
    return data['X'], data['y']


def load_cnn_valset():
    if not os.path.exists(CNN_VALSET_PATH):
        _prepare_cnn_data()
    data = np.load(CNN_VALSET_PATH)
    return data['X'], data['y']


def load_cnn_testset():
    if not os.path.exists(CNN_TESTSET_PATH):
        _prepare_cnn_data()
    data = np.load(CNN_TESTSET_PATH)
    return data['X']


# --------------------------------------------- #
#                  Preprocess
# --------------------------------------------- #

def preprocess_for_nb():
    _prepare_target()
    _prepare_tfidf_data()


def preprocess_for_lsvm():
    _prepare_target()
    _prepare_tfidfdim_data()


def preprocess_for_cnn():
    _perpare_filtered_seg()
    _prepare_vacabulary()
    _prepare_embeddings()
    _prepare_cnn_data()


# --------------------------------------------- #
#                   Entrance
# --------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessor for learning algorithms')
    parser.add_argument(
        'alg', choices=['nb', 'lsvm', 'cnn'],
        help='set the algorithm for which the data will be preprocessed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.alg == 'nb':
        preprocess_for_nb()
    elif args.alg == 'lsvm':
        preprocess_for_lsvm()
    elif args.alg == 'cnn':
        preprocess_for_cnn()
