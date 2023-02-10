import pylab as pl
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

#Download the subworde encoded pretokenized dataset
dataset_subword, info_subword = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

#Get the tokenizer
tokenizer = info_subword.features['text'].encoder

############ Prepare the dataset #############

#parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 256

embedding_dim = 64
filters = 128
kernel_size = 5
DENSE_DIM = 64

#Get the train and test splits
train_data, test_data = dataset_subword['train'], dataset_subword['test']

#shuffle the training data
train_data = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the dataset to the maximum length of sequences
train_data = train_data.padded_batch(BATCH_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_dim),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.GlobalAvgPool1D(), # As alternative to Flatten but with the same output size as the embedding layer
        tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def build_model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        #tf.keras.layers.GlobalAvgPool1D(),
        # As alternative to Flatten but with the same output size as the embedding layer
        tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def plot_graphs(history, string):
    pl.plot(history.history[string])
    pl.plot(history.history['val_' + string])
    pl.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def conv1D_vs_globalMaxPooling1D():

    # Hyperparameters
    batch_size = 1
    time_steps = 20
    features = 16
    filters = 8
    kernel_size = 5

    print(f'batch_size: {batch_size}')
    print(f'time steps (sequence length): {time_steps}')
    print(f'features (embedding size): {features}')
    print(f'filters : {filters}')
    print(f'kernel_size : {kernel_size}')

    # Define array input with random values
    random_input = np.random.rand(batch_size, time_steps, features)
    print(f'shape of input array: {random_input.shape}')

    #Define Conv1D Layer and pass the input_array to it
    lstm = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
    result_conv1D = lstm(random_input)
    print(f'shape of Conv1D output (return_sequences False): {result_conv1D.shape}') # [(batch_size, Input_dim - kernel_size) + 1, ]

    #Define LSTM tha return a sequence
    lstm = tf.keras.layers.GlobalAvgPool1D()
    result_globalMaxPooling1D = lstm(random_input)
    print(f'shape of GlobalMaxPooling1D output (return_sequences True): {result_globalMaxPooling1D.shape}')

    return random_input, result_conv1D, result_globalMaxPooling1D


if __name__ == '__main__':

    model = build_model()

    random_input, result_conv1D, result_globalMaxPooling1D = conv1D_vs_globalMaxPooling1D()
    #print(result_conv1D)

    NUM_EPOCHS = 10 #50

    #train the model
    history = model.fit(train_data, epochs=NUM_EPOCHS, validation_data=test_data)

    #plot_graphs(history, 'accuracy')
    #plot_graphs(history, 'loss')
