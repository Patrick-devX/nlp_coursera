import tensorflow as tf
import numpy as np



#Load dataset
data = open(r'C:\Users\tchuentep\PycharmProjects\coursera_nlp\data\corpus_nlp.txt').read()

#Lowercase and split the text
corpus = data.lower().split('\n')

#print(corpus)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1

print(f'word index dictionary: {tokenizer.word_index}')
print(f'total number of words: {tokenizer}')

#Processing the dataset
input_sequences = list()

#loop for every line
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    #loop over the tokernized line several times to generate subphrases
    for i in range(1, len(token_list)):

        #generate subphrase
        n_gram_sequence = token_list[:i+1]

        # Append the subphrase to the sequence list
        input_sequences.append(n_gram_sequence)

# Get the length of the longest token_list
max_sequence_length = max([len(x) for x in input_sequences])

# pad all sequences
input_sequences_padded = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

#create input and label by splitting the last token in subphrases
xs = input_sequences_padded[:, :-1]
ys = input_sequences_padded[:, -1]

#Convert the label into one hot array
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)

print(xs)
print(ys)

def build_model():
    #Hyperparameter
    embedding_dim = 100
    lstm_dim = 150
    EPOCHS = 500
    learning_rate = 0.01

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_sequence_length-1),
        tf.keras.layers.LSTM(lstm_dim),
        tf.keras.layers.Dense(total_words, activation='softmax'),
    ])
    #Set the training parameter
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    #Pront the model sommary
    model.summary()

    history = model.fit(xs, ys, epochs=EPOCHS, verbose=1)

    return model, history


if __name__ == '__main__':

    model, history = build_model()
