import csv
import tensorflow as tf


def remove_stopwords(sentence):
    import re
    """
    Remove a List of stopwords
    :param sentence: (string) sentence to remove the stopwords from
    :return:
    sentence (string) lowercase sentence without the stopwords
    """
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]

    #sentence converted to lowercase-onlc
    sentence = sentence.lower()
    #sentence = [word for word in re.split("\W+", sentence) if word.lower() not in stopwords]

    for word in stopwords:
        token = " " + word + " "
        sentence = sentence.replace(token, " ")
        sentence = sentence.replace("  ", " ")
    return sentence


def parse_data_from_file(filemname):
    """
    Extracts sentences and labels from a csv file
    :param filemname: (string) path to the csv file
    :return: sentences, labels (list of strings, list of string): tuple containing lists of sentences and labels
    """
    sentence = []
    labels = []
    with open(filemname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            labels.append(line[0])
            sentence.append((line[1]))

    return sentence[1:], labels[1:]

def fit_tokenizer(sentences):
    """
    Instantiates the Tokenizer
    :param sentences: (list)  lower_cased sentences without stopwords
    :return: tokenizer (object) : an instance of the Tokenizer class containing the word-index dictionary
    """
    # Instantiate the Tokenizer class by passing in the oov_token argument
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')  # OOV out of vocabulary

    # fit on sentences
    tokenizer.fit_on_texts(sentences)

    return tokenizer

def get_padded_sequences(tokenizer, sentences):
    """
    Generates an array of token sequences and pads them to the same length
    :param tokenizer: Tokenizer instance containing the word-index dictionary
    :param sentences: (list of string) list of sentences to tokenize and pad
    :return:
    padded_sequences (array of int) tokenized sentences padded to the same length
    """

    #Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    #pad the sequences using post padding strategy
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post')

    return padded_sequences

def tokenize_labels(labels):
    """
    Tokenizes the Labels
    :param labels: (list of strings) labels to tokenize
    :return: label_sequences, label_word_index(list of string, dictionary) tokenized labels and the word_index
    """

    # Instantiate the Tokenizer class by passing in the oov_token argument
    label_tokenizer = tf.keras.preprocessing.text.Tokenizer()  # OOV out of vocabulary

    # Fit the tokenizer to the labels
    label_tokenizer.fit_on_texts(labels)

    #save the word_index
    label_word_index = label_tokenizer.word_index

    #save the sequences
    label_sequences = label_tokenizer.texts_to_sequences(labels)

    return label_sequences, label_word_index



if __name__ == '__main__':
    print(remove_stopwords('I am about to go to the store an get any snack'))
    sentences, labels = parse_data_from_file('.\data\BBC_text.csv')
    print(f'There are {len(sentences)} sentences in the dataset.\n')
    print(f'The first sentence hast  {len(sentences[0].split())} words (after removing stopwords).\n')
    print(f'There first 5 labels are {labels[:5]} sentences in the dataset.\n')

    tokenizer = fit_tokenizer(sentences)
    word_index = tokenizer.word_index
    print(f'Vocabulary contains {len(word_index)} words.\n')
    print(f'T<OOV> token included in vocabulary' if '<OOV>' in word_index  else '<OOV>' 'tocken Not included in vocabulary')

    padded_sequences = get_padded_sequences(tokenizer, sentences)
    print(f'First padded sequence looks like this \n\n {padded_sequences[0]} \n')

    label_sequences, label_word_index = tokenize_labels(labels)
    print(f'Vocabulary of labels looks like this \n\n {label_word_index} \n')
    print(f'First ten label sequences \n\n {label_sequences[:10]} \n')

