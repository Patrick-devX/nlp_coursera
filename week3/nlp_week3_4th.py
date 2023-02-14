import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

datastore = [json.loads(line) for line in open(r'C:\Users\tchuentep\PycharmProjects\coursera_nlp\data\sarcasm_dataset.json', 'r')]

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Preparation parameters
TRAINING_SIZE = 20000
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"
padding_type = 'post'

#Split sentences
training_sentences = sentences[0:TRAINING_SIZE]
testing_sentences = sentences[TRAINING_SIZE:]

#Split Labels
training_labels = labels[0:TRAINING_SIZE]
testing_labels = labels[TRAINING_SIZE:]

#Data Processing
# Initialize the Tokenizer class
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

#Generate the word index dictionary
tokenizer.fit_on_texts(training_sentences)

#Generate word index dictionary
word_index = tokenizer.word_index

#Generate and pad the training sequences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Generate an pad testing sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = tf.keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Convert the label list to numpy array
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

def build_model_LSTM():
    #Hyperparameter
    embedding_dim = 16
    lstm_dim = 32
    dense_dim = 24

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    #Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Pront the model sommary
    model.summary()

    return model

def build_model_Conv1D():
    #Hyperparameter
    embedding_dim = 16
    filters = 128
    kernel_size = 5
    dense_dim = 24

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    #Set the training parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Pront the model sommary
    model.summary()

    return model

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

if __name__ == '__main__':

    NUM_EPOCHS = 10

    model_LSTM = build_model_LSTM()
    history_LSTM = model_LSTM.fit(training_padded, training_labels, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    plot_graphs(history_LSTM, 'accuracy')
    plot_graphs(history_LSTM, 'loss')

    # model_Conv1D = build_model_Conv1D()
    # history_Conv1D = model_Conv1D.fit(training_padded, training_labels, epochs=NUM_EPOCHS, validation_data=(testing_padded, testing_labels))
    # plot_graphs(history_Conv1D, 'accuracy')
    # plot_graphs(history_Conv1D, 'loss')







