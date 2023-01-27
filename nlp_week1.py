# This is a sample Python script.
import tensorflow as tf


sentences = [
             'I love my dog',
             'I love my cat',
             'You love my dog',
             'Do you think my dog is amazing?']

sentences_x = [
             'I love my dog',
             'I love my cat',
             'You love my dog',
             'Do you think my dog is amazing?',
             'I really love my animals']

# Initialize the Tokenizer class
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>') #OOV out of vocabulary

# Tokenize zhe inout sentences
tokenizer.fit_on_texts(sentences)

#Het the word index dictionary
word_index = tokenizer.word_index
print(word_index)

#Generate list of token sequences
sequences = tokenizer.texts_to_sequences(sentences)
#print(sequences)

sequences_ = tokenizer.texts_to_sequences(sentences_x)
print(sequences_)

# pad the sequences to a uniform length
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post')
#print(padded)

#Print the results
#print('\nWord Index = ', word_index)
#print('\nsequences = ', sequences)
#print('\npadded sequences= ', padded)



