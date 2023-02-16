import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# sample phrase
data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ..."
corpus = data.lower().split("\n")

print(f' The corpus is: {corpus}')

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1

print(f' The number of used words is: {tokenizer.word_index}')

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Find the max length of sentences in the corpus
max_sequence_len = max([len(x) for x in input_sequences])

#Padding
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
#labels = input_sequences[:, -1:]
labels = input_sequences[:, -1]

print(input_sequences)
print(xs)
print(labels)

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words, dtype='int32')
#yss = tf.keras.utils.to_categorical(l, num_classes=total_words)
print(ys.shape)
#print(yss.shape)

def build_model():
    #Hyperparameter
    embedding_dim = 64
    lstm_dim = 20
    EPOCHS = 500

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_len-1),
        tf.keras.layers.LSTM(lstm_dim),
        tf.keras.layers.Dense(total_words, activation='relu'),
    ])
    #Set the training parameter
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Pront the model sommary
    model.summary()

    history = model.fit(xs, ys, epochs=EPOCHS, verbose=1)

    return model

def plot_graphs(history, string):
    plt.plot(history.history[string])
    #plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    #plt.legend([string, 'val_' + string])
    plt.show()

#Genarating text by addind the predicted word with the max probabilitie
def generating_text(seed_text, next_words, model):
    """
    :param seed_text: (string)
    :param next_words: total number of words to predict
    :param model: (object ) trained tensorflow model
    :return: (string) the seed text combined with the generated text
    """
    # loop until desired length is reached
    for _ in range(next_words):
        #convert the seed text to a token sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        #pad the sequence
        token_list_padded = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        #fee to the trained mdel an get the probabilities for each index
        probabilities = model.predict(token_list_padded)

        #Get the index with the highest probability
        predicted = np.argmax(probabilities, axis=-1)[0]

        # Ignore the index o because that is just the padding
        if predicted != 0:
            # Look up the word associated with the index
            output_word = tokenizer.index_word[predicted]

            #combine with the seed tet
            seed_text += " " + output_word
    return seed_text

def test_function(tokenizer, element_number, xs, ys):
    #print token list
    print(f'tocken list: {xs[element_number]}')
    print(f'decoded to text: {tokenizer.sequences_to_texts([xs[element_number]])}')

    # Print label
    print(f'one hot label: {ys[element_number]}')
    print(f'index of label: {np.argmax([ys[element_number]])}')



# Genarating text by addind the predicted word with the max probabilitie
def generating_text_x(seed_text, next_words, model):
    """
    :param seed_text: (string)
    :param next_words: total number of words to predict
    :param model: (object ) trained tensorflow model
    :return: (string) the seed text combined with the generated text
    """
    # loop until desired length is reached
    for _ in range(next_words):
        # convert the seed text to a token sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # pad the sequence
        token_list_padded = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1,
                                                                          padding='pre')
        # fee to the trained mdel an get the probabilities for each index
        probabilities = model.predict(token_list_padded)

        # Pick a random number from [1, 2, 3]
        choice = np.random.choice([1, 2, 3])

        #sort the probabilities in ascending order
        # an get the random choice from the end of the array
        predicted = np.argsort(probabilities)[0][-choice]

        # Get the index with the highest probability
        predicted = np.argmax(probabilities, axis=-1)[0]

        # Ignore the index o because that is just the padding
        if predicted != 0:
            # Look up the word associated with the index
            output_word = tokenizer.index_word[predicted]

            # combine with the seed tet
            seed_text += " " + output_word
    return seed_text




if __name__ == '__main__':

    history = build_model()
    # plot_graphs(history, 'accuracy')
    # plot_graphs(history, 'loss')

