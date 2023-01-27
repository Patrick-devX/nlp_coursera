import numpy
import tensorflow as tf
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

#Train Test
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels= []

testing_sentences = []
testing_labels = []

#loop over all training examples and save the sentences and labels
for sent, lab in train_data:
    training_sentences.append(sent.numpy().decode('utf8'))
    training_labels.append(lab.numpy())

#loop over all test examples and save the sentences and labels
for sent, lab in test_data:
    testing_sentences.append(sent.numpy().decode('utf8'))
    testing_labels.append(lab.numpy())

training_labels_final = numpy.array(training_labels)
test_labels_final = numpy.array(testing_labels)


###### Generate Pad Sequences ######

#parameters
vocab_size = 10000
max_length = 100
embedding_dim = 16
trunc_type = 'post'
oov_token = '<OOV>'
num_epochs = 10

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

##### Build and Compile The model #####
def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=100),
        #tf.keras.layers.Flatten(),
        tf.keras.layers.GlobalAvgPool1D(), # As alternative to Flatten but with the same output size as the embedding layer
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    model = build_model()
    model.summary()
    model.fit(padded_train, training_labels_final, epochs=num_epochs, validation_data=(padded_test, test_labels_final))

    ##### Visualize Embedding Layer #######
    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()[0]
    print(embedding_weights.shape) # (vocab_size, embedding_dim)

    import io

    out_vector = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('meta.tsv', 'w', encoding='utf-8')

    reserve_word_index = tokenizer.index_word

    for word_num in range(1, vocab_size):
        #Get the word associed with the current index
        word_name = reserve_word_index[word_num]

        #Get the embedding weights associated with the current index
        word_embedding = embedding_weights[word_num]

        #Write the word name
        out_metadata.write(word_name + '\n')

        # Write the word embedding
        out_vector.write('\t'.join([str(x) for x in word_embedding]) + '\n')

    #close files
    out_vector.close()
    out_metadata.close()

    ####------------> Go to Tensorflow Embedding Projector and load the 2 files to see the cisualization: https://projector.tensorflow.org/

