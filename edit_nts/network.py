import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EncoderLSTM:
    
    def __init__(self, hidden_states_size,
                 embedding_size,
                 vocabulary_size,
                 pos_vocabulary_size,
                 pos_embedding_size,
                 embedding=None,
                 embedding_pos=None,
                 number_layers=1,
                 dropout = 0.3):
        # embedding_size = embedding_dim
        self.number_layers = number_layers
        self.hidden_state_size = hidden_states_size
        self.dropout = dropout

        # Save the embeddings: note, they shouldn't be None, but for
        # the sake of implementation, they're optional
        self.embedding = embedding
        self.embedding_pos = embedding_pos

        # Define what input is going to look like (hint, it's an embedding)
        self.inputs = keras.Input(shape=(None,embedding_size+vocabulary_size))

        # Start creating layers
        self.layers = [layers.Bidirectional(layers.LSTM(self.hidden_state_size, return_sequences=True))(self.inputs)]
        self.dropout_layers = [layers.Dropout(rate=self.dropout)(self.layers[0])]

        # Loop to create remaining hidden layers
        for i in range(1, number_layers):
            last_layer = self.dropout_layers[i - 1]
            new_layer = layers.Bidirectional(layers.LSTM(self.hidden_state_size, return_sequences=True))(last_layer)
            self.layers.append(new_layer)
            new_dropout = layers.Dropout(rate=self.dropout)(self.layers[i])
            self.dropout_layers.append(new_dropout)

        # Get the final, output layer
        self.output = self.dropout_layers[number_layers - 1]
        # Create the model
        self.model = keras.Model(inputs = self.inputs, outputs = self.output)
        # Print a summary
        print(self.model.summary())
        
