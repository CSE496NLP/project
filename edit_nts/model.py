#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout

def encoder(vocab_size, embedding_dim):
    x = Input(shape=(None,))
    embedding = Embedding(vocab_size, embedding_dim)(x)
    lstm = LSTM(embedding_dim)
    bi_lstm = Bidirectional(lstm)(embedding)
    dense = Dense(embedding_dim, activation='relu')(bi_lstm)
    dropout = Dropout(0.5)(dense)
    return Model(inputs=x, outputs=dropout)

def main():
    print('Creating encoder')
    enc = encoder(1000, 64)
    enc.summary()
    

if __name__ == "__main__":
    main()
