

import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

#parameters
training_size=20000
vocab_size = 10000
max_length = 32
embedding_dim = 16
trunc_type = 'post'
padding_type='post'
oov_token = '<OOV>'
num_epochs = 10

datastore = [json.loads(line) for line in open('.\data\Sarcasm_Headlines_Dataset.json', 'r')]

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Split Train and Test
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:0]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:0]

#Initialize the Tokenizer Class
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_token)

#Generate word_index dictionary for training sentences
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

#Generate and pad the training sequences
sequences_train = tokenizer.texts_to_sequences(training_sentences)
padded_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length, truncating=trunc_type)

#Generate and pad the Testing sequences
sequences_test = tokenizer.texts_to_sequences(testing_sentences)
padded_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length, truncating=trunc_type)

training_labels = numpy.array(training_labels)
testing_labels = numpy.array(testing_labels)

##### Build and Compile The model #####
def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.GlobalAvgPool1D(), # As alternative to Flatten but with the same output size as the embedding layer
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def plotter(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history[string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, string])
    plt.show()

if __name__ == '__main__':

    model = build_model()
    model.summary()
    history = model.fit(padded_train, training_labels, epochs=num_epochs, validation_data=(padded_test, testing_labels), verbose=2)
    plotter(history, 'accuracy')
    plotter(history, 'loss')

    ##### Visualize Embedding Layer #######
    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()[0]
    print(embedding_weights.shape)  # (vocab_size, embedding_dim)

    import io

    out_vector = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('meta.tsv', 'w', encoding='utf-8')

    reserve_word_index = tokenizer.index_word

    for word_num in range(1, vocab_size):
        # Get the word associed with the current index
        word_name = reserve_word_index[word_num]

        # Get the embedding weights associated with the current index
        word_embedding_weights = embedding_weights[word_num]

        # Write the word name
        out_metadata.write(word_name + '\n')

        # Write the word embedding
        out_vector.write('\t'.join([str(x) for x in word_embedding_weights]) + '\n')

    # close files
    out_vector.close()
    out_metadata.close()

    ####------------> Go to Tensorflow Embedding Projector and load the 2 files to see the cisualization: https://projector.tensorflow.org/

