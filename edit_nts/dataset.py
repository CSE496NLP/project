import os
import pickle

WIKILARGE_PATH = '/common/cse896/asturtz/data-simplification/wikilarge'
WIKILARGE_TRAIN_PREFIX = 'wiki.full.aner.ori.train'
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

def serialize_tokens(file_prefix, tokens):
    for key, value in tokens.items():
        with open(os.path.join('output', file_prefix + '.' + key + '.pickle'), 'wb') as file:
            pickle.dump(value, file)
    
    return None

def deserialize_tokens(file_prefix):
    token_keys = ['complex_tokens', 'complex_pos', 'simple_tokens', 'edit_labels']

    tokens = {}
    for key in token_keys:
        with open(os.path.join('output', file_prefix + '.' + key + '.pickle'), 'rb') as file:
            tokens[key] = pickle.load(file)
            
    return tokens

def wikilarge():
    train_path_prefix = os.path.join(WIKILARGE_PATH, WIKILARGE_TRAIN_PREFIX)
    
    return load_raw_lines(train_path_prefix)

