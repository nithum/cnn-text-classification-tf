import pandas as pd
import numpy as np
import re
import itertools
import cPickle
from collections import Counter
from baselines import tidy_labels, load_cf_data, plurality
from ngram import get_labeled_comments


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_rt():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def transform_y(y):
    if y[0]:
        return [0,1]
    else:
        return [1,0]

def load_data_and_labels_wiki_char(datfile):
    """
    Loads wikipedia data from pre-split files.
    Splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    data = pd.read_csv('data/%s.csv' % datfile)
    x_text = data[['x']].values.tolist()
    y = data[['y']].values.tolist()
    # Split by words
    x_text = [sent[0].strip() for sent in x_text]
    #x_text = [clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]
    x_text = [list(s) for s in x_text]
    # Generate labels
    y = map(transform_y, y)
    return [x_text, y]

def load_data_and_labels_wiki(datfile):
    """
    Loads wikipedia data from pre-split files.
    Splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    data = pd.read_csv('data/%s.csv' % datfile)
    x_text = data[['x']].values.tolist()
    y = data[['y']].values.tolist()
    # Split by words
    x_text = [sent[0].strip() for sent in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    y = map(transform_y, y)
    return [x_text, y]


def pad_sentences(sentences, max_length = None, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if max_length is None:
        max_length = max(len(x) for x in sentences)
    # TODO: have max_length intelligently take the min of itself and the longest sentence
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if (len(sentence) <= max_length):
            num_padding = max_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            padded_sentences.append(sentence[0:max_length])
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def filter_by_vocab(sentences, vocabulary):
	filtered_sentences = []
	for sentence in sentences:
		new_sentence = [word for word in sentence if word in vocabulary]
		filtered_sentences.append(new_sentence)
		# Should print count of number of words removed
	return filtered_sentences


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_training_data(datfile = 'b_train', max_length = 500):
    """
    Loads and preprocessed data for training on the wikipedia dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_wiki(datfile)
    sentences_padded = pad_sentences(sentences, max_length = max_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # TODO: Should test for directory existence and create directory if not
    cPickle.dump([vocabulary, vocabulary_inv], open('vocabulary/wiki.p', 'wb'))
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_training_data_char(datfile = 'b_train', max_length = 500):
    """
    Loads and preprocessed data for training on the wikipedia dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_wiki_char(datfile)
    sentences_padded = pad_sentences(sentences, max_length = max_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    # TODO: Should test for directory existence and create directory if not
    cPickle.dump([vocabulary, vocabulary_inv], open('vocabulary/wiki_char.p', 'wb'))
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_eval_data(datfile = 'b_test'):
    """
    Loads and preprocessed data for evaluating on the wikipedia dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_wiki(datfile)
    vocabs = cPickle.load(open('vocabulary/wiki.p', 'rb'))
    vocabulary, vocabulary_inv = vocabs[0], vocabs[1]
    # TODO: Is there a more intelligent way than just deleting words not in vocab?
    sentences = filter_by_vocab(sentences, vocabulary)
    # TODO: Should pickle and load the max_length as well!!
    sentences_padded = pad_sentences(sentences, max_length = 500)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_eval_data_char(datfile = 'b_test'):
    """
    Loads and preprocessed data for evaluating on the wikipedia dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_wiki(datfile)
    vocabs = cPickle.load(open('vocabulary/wiki_char.p', 'rb'))
    vocabulary, vocabulary_inv = vocabs[0], vocabs[1]
    # TODO: Is there a more intelligent way than just deleting words not in vocab?
    sentences = filter_by_vocab(sentences, vocabulary)
    # TODO: Should pickle and load the max_length as well!!
    sentences_padded = pad_sentences(sentences, max_length = 500)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

# TODO: combine the char and non-char functions intelligently


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    """
    Create a separate word vector for unknown words    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k), dtype='float32')            
    i = 0
    # TODO: Use enumerate instead
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_training_data_word2vec(datfile = 'b_train', max_length = 500):
    """
    Loads and preprocessed data for training on the wikipedia dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_wiki(datfile)
    sentences_padded = pad_sentences(sentences, max_length = max_length)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    w2v_file = 'word2vec/GoogleNews-vectors-negative300.bin' 
    w2v = load_bin_vec(w2v_file, vocabulary)
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocabulary)
    W, vocabulary = get_W(w2v)
    # word_idx_map is the new vocabulary. Could make a new vocabulary_inv ??

    # TODO: Should test for directory existence and create directory if not
    cPickle.dump([vocabulary, vocabulary_inv], open('vocabulary/wiki.p', 'wb'))
    print W.shape
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, W] # NOTE: In this case, vocabulary_inv is wrong

      