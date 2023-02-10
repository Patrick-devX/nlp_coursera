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
max_length = 100
embedding_dim = 64
BUFFER_SIZE = 10000
BATCH_SIZE = 256
LSTM_DIM = 64
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
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.GlobalAvgPool1D() # As alternative to Flatten but with the same output size as the embedding layer
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
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

def lstm_layer():

    # Hyperparameters
    batch_size = 1
    time_steps = 20
    features = 16
    lstm_dim = 8

    print(f'batch_size: {batch_size}')
    print(f'time steps (sequence length): {time_steps}')
    print(f'features (embedding size): {features}')
    print(f'lstm output units : {lstm_dim}')

    # Define array input with random values
    random_input = np.random.rand(batch_size, time_steps, features)
    print(f'shape of input array: {random_input.shape}')

    #Define LSTM tha return a single output
    lstm = tf.keras.layers.LSTM(lstm_dim, return_sequences=False)
    result_rsf = lstm(random_input)
    print(f'shape of lstm output (return_sequences False): {result_rsf.shape}')

    #Define LSTM tha return a sequence
    lstm = tf.keras.layers.LSTM(lstm_dim, return_sequences=True)
    result_rst = lstm(random_input)
    print(f'shape of lstm output (return_sequences True): {result_rst.shape}')

    return random_input, result_rsf, result_rst




if __name__ == '__main__':

    model = build_model()

    #random_input, result_rsf, result_rst = lstm_layer()
    #print(result_rst)

    NUM_EPOCHS = 10 #50

    #train the model
    history = model.fit(train_data, epochs=NUM_EPOCHS, validation_data=test_data)

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
