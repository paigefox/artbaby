# lstm autoencoder recreate sequence
from numpy import array
import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model


class IdentityConstraint(Constraint):
    """Contains weight tensors for `ref_value`."""

    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        return self.ref_value

    def get_config(self):
        return {'ref_value': self.ref_value}

# define the embeddings
embedding_dim = 5
vocab_size = 10
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
# define input sequence
# reserved ids in vocab: EOL, space, GO/CLS, EOS, PAD
id_sequence = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# reshape input into [samples, timesteps, features]
n_in = len(id_sequence)
id_sequence = tf.reshape(id_sequence, (1, n_in))
sequence = embedding_layer(id_sequence)
# sequence = sequence.reshape((1, n_in, embedding_dim))
# define model
model = Sequential()
hidden_units = 100
bottleneck_units = 10
model.add(LSTM(hidden_units, activation='relu',
               input_shape=(n_in, embedding_dim)))
# define the bottleneck with size. Sigmoid to keep 0 <= output <= 1
model.add(Dense(bottleneck_units, activation="sigmoid", name="bottleneck"))

# The output of the previous LSTM encoder is the final state of the LSTM,
# and will be the size of the bottleneck. Repeat it n_in times for the decoder
# to output the correct sequence length
model.add(RepeatVector(n_in))
# TODO: turn the bottleneck into an image, make this a CNN
model.add(LSTM(hidden_units, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(embedding_dim)))
model.add(Dense(vocab_size, kernel_constraint=IdentityConstraint(
    tf.transpose(embedding_layer.embeddings))))
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True))
# fit model
for i in range(300):
    model.train_on_batch(x=sequence, y=id_sequence)
for layer in model.layers:
    print(layer.output_shape)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
logits = model.predict(sequence, verbose=0)
encoder = Model(inputs=model.input,
                outputs=model.get_layer("bottleneck").output)
print(encoder(sequence))
print(np.argmax(logits, axis=2))
print(sequence)
print(id_sequence)
