import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
#tensorboard = TensorBoard(log_dir='log/{}'.format(time()))

#Download the plain text default config
imdb_plaintext, info_plaintext = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
#Download the subword encoded oretokenized dataset
imdb_subwords, info_subwords = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

#Train Test
train_data, test_data = imdb_subwords['train'], imdb_subwords['test']
#Acces to tokenizer
tokenizer_subwords= info_subwords.features['text'].encoder

"""The Data returned by the 2 datasets are different. For the default, it will be strings as il the last Lab."""
print(info_plaintext.features) #'label': ClassLabel(shape=(), dtype=int64, num_classes=2), 'text': Text(shape=(), dtype=string)
print(info_subwords.features)  #'label': ClassLabel(shape=(), dtype=int64, num_classes=2), 'text': Text(shape=(None,), dtype=int64, encoder=<SubwordTextEncoder vocab_size=8185>


#Take 2 Training examples and print the test feature
for example in imdb_plaintext['train'].take(2):
    print('plain Text')
    print(example[0].numpy().decode('utf-8'))

for example in imdb_subwords['train'].take(2):
    print('print subword')
    print(example[0])

#Take 2 Training examples an decode the text feature ( code to text)
for example in imdb_subwords['train'].take(2):
    print('subword coded in words')
    print(tokenizer_subwords.decode(example[0]))


#Get the Train Data
train_data = imdb_plaintext['train']

train_sentences = list()

for s, t in train_data:
    train_sentences.append(s.numpy().decode('utf-8'))

vocab_size = 10000
oov_tok = '<OOV>'

#Initialize The Tokenizer
tokenizer_plaintext = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

#Generate word_index dictionary for training sentences
tokenizer_plaintext.fit_on_texts(train_sentences)

#Generate  the training sequences
sequences_train = tokenizer_plaintext.texts_to_sequences(train_sentences)

#Decode the first sequence using the Tokenizer class
decoded_train_first_sequence = tokenizer_plaintext.sequences_to_texts(sequences_train[0:1])
print('decoded_train_first_sequence')
print(decoded_train_first_sequence)

#print the subwords
print(len(tokenizer_plaintext.word_index))

"""Subword text encoding gets around the problem of small vocabular size. With small vocabular size , we got a lot of oov in the sentenses. Subword text encoding gets around this problem by using
parts of the word to compose whole words. This makes it more flexible when it encounters uncommon words. See how thes subwords look like for particular encoder:"""

#print(tokenizer_subwords.subwords)

"""Now we will use it on the previous plain text sentence, you will see that it won't have any OOVs even if it has a smaller vocab size (only 8k compared to 10k above)"""

#Encode the first plaintext sentence using the subword text encoder
tokenized_string = tokenizer_subwords.encode(train_sentences[0])
#print(tokenizer_plaintext.sequences_to_texts(sequences_train[0]))
print(train_sentences[0])
print(tokenized_string)

#Decode the sequence
original_string = tokenizer_subwords.decode(tokenized_string)
print(original_string)

""" Subword encoding can even perform well on words that are not commonly found on movie reviews. See first the result when using the plain text tokenizer. As expected, it will show many OOVd"""
print("####################################################################")
#Define sample sentence
sample_string = "TensorFlow, from basic to mastery"

#Encode using the plain text tokenizer
tokenized_string = tokenizer_plaintext.texts_to_sequences([sample_string])
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_plaintext.sequences_to_texts(tokenized_string)
print('Oroginal string is {}'.format(original_string))

#========== Compare to subword encoder =================
tokenized_string_sw = tokenizer_subwords.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string_sw))

original_string_sw = tokenizer_subwords.decode(tokenized_string_sw)
print('Tokenized string is {}'.format(original_string_sw))

# Show token to subword mapping
# for ts in tokenized_string:
#     print('{} -----> {}'.format(ts, tokenizer_subwords.decode([ts])))

#Training model
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 64
NUM_EPOCHS = 10

#Get the train and test split
train_data, test_data = imdb_subwords['train'], imdb_subwords['test']

#shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

#Batch and pad the dataset to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer_subwords.vocab_size, output_dim=EMBEDDING_DIM),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Print model sumary
model.summary()

#Set the training parameter
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='log/{}'.format(time()))
#Start training
history = model.fit(train_dataset, validation_data=test_dataset, epochs=NUM_EPOCHS) #callbacks=[TensorBoard]

