import argparse
import numpy as np
import nltk
import pandas
from gensim.models import Word2Vec


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array(np.mean([self.word2vec[w] for w in X if w in self.word2vec] or [np.zeros(self.dim)], axis=0))


def pos2vec(pvdf):
    with open(args['output'] + "/pos2vec.txt", 'w') as of:
        full_pos_list = []
        for index, row in pvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            tagged_list = map(lambda w_p_tuple: w_p_tuple[1], nltk.pos_tag(word_list))
            full_pos_list.append(tagged_list)
        model = Word2Vec(full_pos_list, size=50, window=2, min_count=1, workers=4)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        mev = MeanEmbeddingVectorizer(w2v)
        p2v_dict = dict()
        for index, row in pvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            tagged_list = map(lambda w_p_tuple: w_p_tuple[1], nltk.pos_tag(word_list))
            vec = mev.transform(tagged_list)
            p2v_dict[row['example_id']] = " ".join(map(str, vec.tolist()))
        for k, v in p2v_dict.iteritems():
            of.write(k + " " + v + "\n")


def lex2vec(lvdf):
    sswe_dict = dict()
    with open("embedding/sswe-u.txt", 'r') as sf:
        for line in sf:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            sswe_dict[word] = coefs

    with open(args['output'] + "/lex2vec.txt", 'w') as of:
        mev = MeanEmbeddingVectorizer(sswe_dict)
        l2v_dict = dict()
        for index, row in lvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            vec = mev.transform(word_list)
            l2v_dict[row['example_id']] = " ".join(map(str, vec.tolist()))
        for k, v in l2v_dict.iteritems():
            of.write(k + " " + v + "\n")


if __name__ == '__main__':
    # python Embedder.py -i out_data_2/data_2.csv -o embedding/data_set_2 -pv y -lv y
    # python Embedder.py -i out_data_1/data_1.csv -o embedding/data_set_1 -pv y -lv y
    parser = argparse.ArgumentParser(description='Data Pre-processor')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', help='path to input preprocessed training data file', required=True)
    required.add_argument('-o', '--output', help='path to output embedding folder', required=True)
    optional.add_argument('-pv', '--pos2vec', help='pos2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-lv', '--lex2vec', help='lex2vec (y/n)', choices=['y', 'n'], required=False)
    parser._action_groups.append(optional)
    args = vars(parser.parse_args())

    df = pandas.read_csv(args['input'], sep='\t')
    if args['pos2vec'] == 'y':
        pos2vec(df)
    if args['lex2vec'] == 'y':
        lex2vec(df)

