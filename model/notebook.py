import pandas as pd
import numpy as np
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
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

# Replace 'training.csv' with the location of your training file
# Read in the dataset into a Pandas DataFrame
df = pd.read_csv(
    'datasets/sentiment140/testdata.manual.2009.06.14.csv', encoding='latin-1')

# Drop unnecessary columns, leaving behind the [label, text] columns
df = df.drop(df.columns[[0, 1, 2, 3, 4]], axis=1)

# Rename these columns
df.columns = ['text']

# print(df['text'].values)


def tokenize(df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(df['text'].values)
    tensor = tokenizer.texts_to_sequences(df['text'].values)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')
    sequences = tokenizer.texts_to_sequences(df['text'].values)
    print(tensor)
    return tensor, tokenizer

num_of_rows = df.shape[0]

# Shuffle the rows
msk = np.random.rand(len(df)) <= 0.7
train = df[msk]
valid = df[~msk]

# Split the DataSet into Train and Validation sets 70/30
train, valid = train_test_split(df, test_size=0.3)

# Save back to .csv format
train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)

print(tokenize(valid))


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
vocab_size = 10000
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
# define input sequence
input_sequence = ["_B", "=", "'", "a", "s", "s", "e", "t", "g", "e", "n", "_", "m", "a", "n", "i", "f", "e", "s", "t", "'", "\n", "_A", "=", "'", "s", "t", "a", "t", "i", "c", "_", "u", "r", "l", "_", "p", "r", "e", "f", "i", "x", "'", "\n", "__all__", "=", "[", "'", "M", "a", "n", "i", "f", "e", "s", "t", "e", "d", "S", "t", "a", "t", "i", "c", "U", "R", "L", "G", "e", "n", "e", "r", "a", "t", "o", "r", "'", "]", "\n", "from", "itertools", "import", "cycle", "\n", "from", "zope", ".", "component", "import", "adapts", "\n", "from", "zope", ".", "interface", "import", "implements", "\n", "from", "weblayer", ".", "interfaces", "import", "IRequest", ",", "ISettings", ",", "IStaticURLGenerator", "\n", "from", "weblayer", ".", "settings", "import", "require_setting", "\n", "require_setting",
                  "(", "_A", ",", "default", "=", "'", "/", "s", "t", "a", "t", "i", "c", "/", "'", ")", "\n", "require_setting", "(", "_B", ")", "\n", "class", "ManifestedStaticURLGenerator", ":", "\n", "	", "adapts", "(", "IRequest", ",", "ISettings", ")", ";", "implements", "(", "IStaticURLGenerator", ")", "\n", "def", "__init__", "(", "A", ",", "request", ",", "settings", ")", ":", "B", "=", "settings", ";", "A", ".", "_dev", "=", "B", ".", "get", "(", "'", "d", "e", "v", "'", ",", "False", ")", ";", "A", ".", "_host", "=", "B", ".", "get", "(", "'", "s", "t", "a", "t", "i", "c", "_", "h", "o", "s", "t", "'", ",", "request", ".", "host", ")", ";", "A", ".", "_static_url_prefix", "=", "B", "[", "_A", "]", ";", "A", ".", "_manifest", "=", "B", "[", "_B", "]", ";", "A", ".", "_subdomains", "=", "cycle", "(", "B", ".", "get", "(", "'", "s", "t", "a", "t", "i", "c", "_", "s", "u", "b", "d", "o", "m", "a", "i", "n", "s", "'", ",", "'", "1", "2", "3", "4", "5", "'", ")", ")", "\n", "def", "get_url", "(", "A", ",", "path", ")", ":", "\n", "		", "C", "=", "A", ".", "_manifest", ".", "get", "(", "path", ",", "path", ")", "\n", "if", "True", ":", "B", "=", "A", ".", "_host", "\n", "else", ":", "B", "=", "'", "%", "s", ".", "%", "s", "'", "%", "(", "A", ".", "_subdomains", ".", "next", "(", ")", ",", "A", ".", "_host", ")", "\n", "return", "'", "/", "/", "%", "s", "%", "s", "%", "s", "'", "%", "(", "B", ",", "A", ".", "_static_url_prefix", ",", "C", ")", "\n", "", "", ""]
# reserved ids in vocab: EOL, space, GO/CLS, EOS, PAD
id_sequence = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# reshape input into [samples, timesteps, features]
# n_in = len(id_sequence)
n_in = 30
# id_sequence = tf.reshape(id_sequence, (1, n_in))
sequence = embedding_layer(id_sequence)
# sequence = sequence.reshape((1, n_in, embedding_dim))
# define model
model = Sequential()
hidden_units = 10
bottleneck_units = 30
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
tokenized_data, tokenizer = tokenize(train)
for sequence in tokenized_data:
    sequence = tf.reshape(sequence, (1, 30))
#     print(sequence)
    model.train_on_batch(x=embedding_layer(sequence), y=sequence)
for layer in model.layers:
    print(layer.output_shape)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
# sequence = array([1, 2, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# sequence = tf.reshape(sequence, (1, n_in))
sequence = ["Learning", "about", "lambda", "calculus", ":)"]
# sequence = ["I", "found", "a", "pink", "cigarette"]
# sequence = ["homosexuality", "is", "the", "best", "all", "round", "cover", "an", "agent", "ever", "had"]
id_sequence = array(tokenizer.texts_to_sequences([sequence])[0])
n_in = len(id_sequence)
id_sequence = tf.reshape(id_sequence, (1, n_in))
id_sequence = tf.keras.preprocessing.sequence.pad_sequences(
    id_sequence, padding='post', maxlen=30)
print(id_sequence)
logits = model.predict(embedding_layer(id_sequence), verbose=0)
encoder = Model(inputs=model.input,
                outputs=model.get_layer("bottleneck").output)
id_sequence = tf.reshape(id_sequence, (1, 30))
print("ENCODER OUTPUT: use this for seeding the drawing script")
print(encoder(embedding_layer(id_sequence)))
print("ARGMAX LOGITS")
print(np.argmax(logits, axis=2))
print(id_sequence)
print("Original sequence")
print(sequence)
print("predicted sequence")
print(tokenizer.sequences_to_texts(np.argmax(logits, axis=2)))
