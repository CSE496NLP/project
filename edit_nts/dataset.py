import os
import pickle

WIKILARGE_PATH = '/common/cse896/asturtz/data-simplification/wikilarge'
WIKISMALL_PATH = '/common/cse896/asturtz/data-simplification/wikismall'


def load_raw_lines(path_prefix):
    src_path = path_prefix + '.src'
    dst_path = path_prefix + '.dst'

    src_lines = []
    dst_lines = []

    with open(src_path, 'r') as src_file:
        src_lines = src_file.readlines()

    with open(dst_path, 'r') as dst_file:
        dst_lines = dst_file.readlines()

    return src_lines, dst_lines


def serialize_tokens(path_prefix, tokens):
    token_path = path_prefix + '.pickle'

    with open(token_path, 'wb') as token_file:
        pickle.dump(tokens, token_file)

    return None


def deserialize_tokens(path_prefix):
    token_path = path_prefix + '.pickle'

    tokens = None
    with open(token_path, 'rb') as token_file:
        tokens = pickle.load(token_file)

    return tokens


def wikilarge():
    TRAIN_FILE_PREFIX = 'wiki.full.aner.ori.train'
    train_path_prefix = os.path.join(WIKILARGE_PATH, TRAIN_FILE_PREFIX)

    return load_raw_lines(train_path_prefix)


def wikismall():
    SMALL_TRAIN_FILE_PREFIX = 'PWKP_108016.tag.80.aner.ori.train'
    small_train_path_prefix = os.path.join(WIKISMALL_PATH, SMALL_TRAIN_FILE_PREFIX)

    return load_raw_lines(small_train_path_prefix)
