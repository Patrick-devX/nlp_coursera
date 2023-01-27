import json
import tensorflow as tf

#with open('.\data\Sarcasm_Headlines_Dataset.json', 'r') as f:
    #datastore = json.load({f})

datastore = [json.loads(line) for line in open('.\data\Sarcasm_Headlines_Dataset.json', 'r')]

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

#print('\n Sentences:', sentences)
#print('\n labels:', labels)
#print('\n urls:', urls)

# Initialize the Tokenizer class
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>') #OOV out of vocabulary

# Tokenize the inout sentences
tokenizer.fit_on_texts(sentences)

#Het the word index dictionary
word_index = tokenizer.word_index
print(f' number of words in word_index: {len(word_index)}')

#Print word_index
#print(f' word_index: {word_index}')

#Generate sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post')

#Print simple headline
index = 2
print(f'sample headlince: {sentences[index]}')
print(f'sample padded sequence headlince: {padded_sequences[index]}')

print(word_index['mom'])

# print dim of padded sequences
print(f'shape of padded sequence: {padded_sequences.shape}')
