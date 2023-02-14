import pylab as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


#Download the subworde encoded pretokenized dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

#Get the train and test splits
train_data, test_data = imdb['train'], imdb['test']

# Initialise sentences and labels list
training_sentences = list()
training_labels = list()

testing_sentences = list()
testing_labels = list()

#Loop over all test examples and save the sentences and labels
for sentence, label in train_data:
    training_sentences.append(sentence.numpy().decode('utf-8'))
    training_labels.append(label.numpy())

for sentence, label in test_data:
    testing_sentences.append(sentence.numpy().decode('utf-8'))
    testing_labels.append(label.numpy())

# convert labels list to numpy array
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

# Initialize the Tokenizer class
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

#Generate the word index dictionary for training sentenses
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index

# Generate and pad the training sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

# Grenerate and pad the test sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = tf.keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_length)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

#Flatten
def model_Flatten():
    # Parameters
    embedding_dim = 16
    dense_dim = 6
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Model definition with Flatten Layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model sumary
    model.summary()
    history = model.fit(training_padded, training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    return model, history

# LSTM: Long Short Memory Term
def model_LSTM():
    # Parameters
    embedding_dim = 16
    lstm_dim = 32
    dense_dim = 6
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Model definition with LSTM Layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model sumary
    model.summary()
    history = model.fit(training_padded, training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    return model, history

#Gated Recurrent Unit
def model_GRU():
    # Parameters
    embedding_dim = 16
    gru_dim = 32
    dense_dim = 6
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Model definition with GRU Layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model sumary
    model.summary()
    history = model.fit(training_padded, training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    return model, history

# CONVOLUTION
def model_Conv1D():
    # Parameters
    embedding_dim = 16
    filters = 128
    kernel_size = 5
    dense_dim = 6
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # Model definition with GRU Layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print model sumary
    model.summary()
    history = model.fit(training_padded, training_labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    return model, history

if __name__ == '__main__':

    #Flatten: Plot accuracy and loss history:
    model_Flatten, history_Flatten = model_Flatten()
    plot_graphs(history_Flatten, 'accuracy')
    plot_graphs(history_Flatten, 'loss')

    # LSTM: Plot accuracy and loss history:
    model_LSTM, history_LSTM = model_LSTM()
    plot_graphs(history_LSTM, 'accuracy')
    plot_graphs(history_LSTM, 'loss')

    # GRU: Plot accuracy and loss history:
    model_GRU, history_GRU = model_GRU()
    plot_graphs(history_GRU, 'accuracy')
    plot_graphs(history_GRU, 'loss')

    # Conv1D: Plot accuracy and loss history:
    model_Conv1D, history_Conv1D = model_Conv1D()
    plot_graphs(history_Conv1D, 'accuracy')
    plot_graphs(history_Conv1D, 'loss')



