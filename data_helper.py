import csv
import re
import numpy as np
import torch
from torch.autograd import Variable
import torchtext.datasets as datasets
import os


max_sequence_len = 500

SEED = 1


def prepare_sequence(seq, word_to_ix, cuda=False):
    var = Variable(torch.LongTensor([word_to_ix[w] for w in seq]))
    return var


def prepare_label(label, cuda=False):
    var = Variable(torch.LongTensor([label]))
    return var


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
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


def line_to_words(line):
    clean_line = clean_str_sst(line.strip())
    words = clean_line.split(' ')
    return words


def get_vocab(file_list):
    max_sent_len = 0
    word_to_idx = {}
    idx = 0

    for filename in file_list:
        f = open(filename, "r")
        for line in f:
            words = line_to_words(line)
            max_sent_len = max(max_sent_len, len(words))
            for word in words:
                if not word in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
        f.close()
    return max_sent_len, word_to_idx


def convert_to_tsv(train_name, dev_name, test_name):
    f_names = [train_name]
    if not test_name == '': f_names.append(test_name)
    if not dev_name == '': f_names.append(dev_name)

    train, dev, test, data = [], [], [], []
    train_out, dev_out, test_out = open('./data/SST2_TSV/'+os.path.basename(train_name).split('.')[-1]+'.tsv', 'w'), \
                                   open('./data/SST2_TSV/' + os.path.basename(dev_name).split('.')[-1] + '.tsv', 'w'),\
                                   open('./data/SST2_TSV/' + os.path.basename(test_name).split('.')[-1] + '.tsv', 'w')
    files = []

    f_train = open(train_name, 'r')
    files.append(f_train)
    data.append(train)
    if not test == '':
        f_test = open(test_name, 'r')
        files.append(f_test)
        data.append(test)
    if not dev == '':
        f_dev = open(dev_name, 'r')
        files.append(f_dev)
        data.append(dev)

    for d, f in zip(data, files):
        for line in f:
            words = line_to_words(line)
            sent = " ".join(words[1:]) + '\t' + words[0]
            d.append(sent)

    f_train.close()
    if not test_name == '':
        f_test.close()
    if not dev_name == '':
        f_dev.close()

    train_out.write('\n'.join(train))
    dev_out.write('\n'.join(dev))
    test_out.write('\n'.join(test))

    return train, dev, test


def load_data(train_name, dev_name, test_name):
    print('loading SST2 data for training and evaluation')
    f_names = [train_name]
    if not test_name == '': f_names.append(test_name)
    if not dev_name == '': f_names.append(dev_name)

    max_sent_len, word_to_idx = get_vocab(f_names)
    train, dev, test, data = [], [], [], []
    files = []

    f_train = open(train_name, 'r')
    files.append(f_train)
    data.append(train)
    if not test == '':
        f_test = open(test_name, 'r')
        files.append(f_test)
        data.append(test)
    if not dev == '':
        f_dev = open(dev_name, 'r')
        files.append(f_dev)
        data.append(dev)

    for d, f in zip(data, files):
        for line in f:
            words = line_to_words(line)
            if len(words) > max_sent_len:
                sent = sent[:max_sent_len]
            d.append(words)
    print('train:', len(train), 'dev:', len(dev), 'test:', len(test))

    f_train.close()
    if not test_name == '':
        f_test.close()
    if not dev_name == '':
        f_dev.close()

    return train, dev, test, word_to_idx


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_qualtrics_gold(infile):
    outfile, content = 'qualtrics_gold_standard.test', ''
    writer = open(outfile, 'w')
    num_of_sentence, pos_sents, neg_sents = 0, 0, 0
    with open(infile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            # check if the row is empty
            if row[0] != '' and row[0] != 'Verbatim':
                num_of_sentence += 1
                if row[1] == 'pos':
                    content += '1 ' + clean_str(row[0]) + '\n'
                    pos_sents += 1
                elif row[1] == 'neg':
                    content += '0 ' + clean_str(row[0]) + '\n'
                    neg_sents += 1
    print(num_of_sentence)
    print("Pos sents: %d" % pos_sents)
    print("Neg sents: %d" % neg_sents)
    writer.write(content.strip())


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
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
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    train, dev, test = convert_to_tsv('data/SST2/stsa.binary.train', 'data/SST2/stsa.binary.dev', 'data/SST2/stsa.binary.test')
    print(train)
