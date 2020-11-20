#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import torch
from checkpoint import Checkpoint
from editnts import EditNTS
from label_edits import edit2sent
from data import id2edits, Vocab, POSvocab

from nltk import pos_tag

import sys
import os

def to_token(word, vocab):
    return vocab.w2i[word] if word in vocab.w2i.keys() else vocab.w2i['UNK']

def tokenize_line(line, vocab):
    return list(map(lambda word: to_token(word, vocab), line.lower().split()))

def pos_line(line, pos_vocab):
    return list(map(lambda word: to_token(pos_tag(word), vocab)), line.lower().split())

def translate(line, vocab, pos_vocab, model):
    input_tokens = line.lower().split()
    input_ids = tokenize_line(line, vocab)
    input_pos = pos_line(line, pos_vocab)
    output = model(input_ids, [], input_ids, input_pos, [], 0.0)
    actions = torch.argmax(output, dim=1).view(-1).data.cpu().numpy()
    edit_list = id2edits(actions, vocab)
    new_sentence_compl = ' '.join(edit2sent(input_tokens, edit_list))
    new_sentence = ' '.join(new_sentence_compl.split('STOP')[0].split(' '))
    return new_sentence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, dest='checkpoint',
                        help="Path to Model checkpoint")
    parser.add_argument("--vocabulary-store", type=str, dest='vocabulary_store',
                        help="Path to Vocabulary store")
    parser.add_argument('--vocabulary-size', dest='vocabulary_size', default=30000, type=int)
    parser.add_argument('--device', type=int, default=0, help='Which GPU to use')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    if args.vocabulary_store == None:
        print("No vocabulary store path provided.", file=sys.stderr)
        sys.exit(1)
    if args.checkpoint == None:
        print("No Checkpoint provided.", file=sys.stderr)
        sys.exit(1)

    print("Loading vocabulary...")
    vocab = Vocab()
    vocab.add_vocab_from_file(os.path.join(args.vocabulary_store, "vocab.txt"), args.vocabulary_size)
    vocab.add_embedding(os.path.join(args.vocabulary_store, "glove.6B.100d.txt"))
    pos_vocab = POSvocab(args.vocabulary_store)
    print("Done.")

    print("Loading checkpoint...")
    checkpoint = Checkpoint.load(args.checkpoint)
    editnet = checkpoint.model
    print("Moving Checkpoint to GPU...")
    print(editnet.cuda())
    print("Done.")

    print()
    in_str = input("Type a sentence (quit to exit): ")
    while(in_str != "quit"):
        print(translate(in_str, vocab, pos_vocab, editnet))
        print()
        in_str = input("Type a sentence (quit to exit): ")


if __name__ == "__main__":
    main()
