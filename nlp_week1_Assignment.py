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
    stopwords = ['a', 'about', 'above', 'after', 'again',
                 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been',
                 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do', 'doing',
                 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he',
                 "he's", 'her', 'here',"here's", 'hers', 'herself','him', 'himself', 'his', 'how', "how's", 'i',
                 "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's",
                 "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "or",
                 "ought", "our", "ours", "ourselves", "out", "over", "own","same","she", "she'd", "she'll", "she's",
                 "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
                 "themselves", "then", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
                 "those", "through", "to", "too", "unther", "until", "up", "very", "was", "we", "we'd", "we'll",
                 "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
                 "whos's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "your", "yours",
                 "yourself", "yourselves"]

    #sentence converted to lowercase-onlc
    sentence = sentence.lower()

    sentence = [word for word in re.split("\W+", sentence) if word.lower() not in stopwords]

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



if __name__ == '__main__':
    print(remove_stopwords('I am about to go to the store an get any snack'))
    sentences, labels = parse_data_from_file('.\data\BBC_text.csv')
    print(f'There are {len(sentences)} sentences in the dataset.\n')
    print(f'There first 5 labels are {labels[:5]} sentences in the dataset.\n')
