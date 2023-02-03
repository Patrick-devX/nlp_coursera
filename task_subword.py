import numpy
import tensorflow as tf
import tensorflow_datasets as tfds
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='log/{}'.format(time()))

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
    print(example[0].numpy().decode('utf-8'))

for example in imdb_subwords['train'].take(2):
    print(example[0])

#Take 2 Training examples an decode the text feature ( code to text)
for example in imdb_subwords['train'].take(2):
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

# #Decode the first sequence using the Tokenizer class
# tokenizer_plaintext.sequences_to_texts(sequences_train[0:1])
#
# #print the subwords
# print(len(tokenizer_plaintext.word_index))
#
# """Subword text encoding gets around the problem of small vocabular size. With small vocabular size , we got a lot of oov in the sentenses. Subword text encoding gets around this problem by using
# parts of the word to compose whole words. This makes it more flexible when it encounters uncommon words. See how thes subwords look like for particular encoder:"""
#
# print(tokenizer_subwords.subwords)
#
# """Now we will use it on the previous plain text sentence, you will see that it won't have any OOVs even if it has a smaller vocab size (only 8k compared to 10k above)"""
#
# #Encode the first plaintext sentence using the subword text encoder
# tokenized_string = tokenizer_subwords.encode(train_sentences[0])
# print(tokenized_string)
#
# #Decode the sequence
# original_string = tokenizer_subwords.decode(tokenized_string)
# print(original_string)
