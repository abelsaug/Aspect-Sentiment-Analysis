import argparse
import numpy as np
import nltk
import pandas
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import os.path

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in X if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
        ])
    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(list(word2vec.values())[0])
        else:
            self.dim=0
        
    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer='word')
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in X if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
            ])
    
def pos2vec(pvdf):
    with open(args['output'] + "/pos2vec.txt", 'w') as of:
        full_pos_list = []
        for index, row in pvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            tagged_list = list(map(lambda w_p_tuple: w_p_tuple[1], nltk.pos_tag(word_list)))
            full_pos_list.append(tagged_list)
        model = Word2Vec(full_pos_list, size=120, window=6, min_count=1, workers=4)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        mev = MeanEmbeddingVectorizer(w2v)
        p2v_dict = dict()
        for index, row in pvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            tagged_list = map(lambda w_p_tuple: w_p_tuple[1], nltk.pos_tag(word_list))
            vec = mev.transform(tagged_list)
            p2v_dict[row['example_id']] = " ".join(map(str, vec.tolist()))
        for k, v in p2v_dict.items():
            of.write(k + " " + v + "\n")
        return p2v_dict


def lex2vec(lvdf):
    sswe_dict = dict()
    with open("embedding/sswe-u.txt", 'r', encoding = 'utf8') as sf:
        for line in sf:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            sswe_dict[word] = coefs

    with open(args['output'] + "/lex2vec.txt", 'w') as of:
        tev = TfidfEmbeddingVectorizer(sswe_dict)
        l2v_dict = dict()
        for index, row in lvdf.iterrows():
            word_list = nltk.word_tokenize(row[' text'])
            tev.fit(word_list)
            vec = tev.transform(word_list)[0]
            l2v_dict[row['example_id']] = " ".join(map(str, vec.tolist()))
        for k, v in l2v_dict.items():
            of.write(k + " " + v + "\n")
        return l2v_dict

def sentence2sequence(glove_wordmap, sentence):
    """
     
    - Turns an input sentence into an (n,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
    
      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows, words

def glove2vec(gvdf, customTrained=False, mean = False):
    #TODO: Custom Train
    glove_vectors_file = "glove.840B.300d.txt"
    glove_wordmap = {}
    encoding="utf-8"
    X = []
    for index, row in gvdf.iterrows():
        X.append(row[' text'].lower())
    all_words = set(word for words in X for word in words.split(" "))
    word_list = []
    with open(glove_vectors_file, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(encoding)
            if word in all_words:
                nums=np.array(parts[1:], dtype=np.float32)
                glove_wordmap[word] = nums
        
    tev = TfidfEmbeddingVectorizer(glove_wordmap)
    with open(args['output'] + "/glove2vec.txt", 'w', encoding="utf8") as of:
        g2v_dict = dict()
        for index, row in gvdf.iterrows():
            word_list = row[' text'].split(" ")
            
            if mean:
                tev.fit(word_list)
                word_list = tev.transform(word_list)[0]
            else:
                word_list, words = sentence2sequence(glove_wordmap, row[' text'])
            g2v_dict[row['example_id']] = " ".join(map(str, word_list))
        for k, v in g2v_dict.items():
            of.write(k + " " + v + "\n")
        return g2v_dict
        

def word2vec(wdf, customTrained=False, mean = False):
    # Load Google's pre-trained Word2Vec model.
    if not customTrained:
        w2v_wordmap = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
    word_list = []
    sentences = []
    if customTrained:
        if os.path.exists('GoogleNews-vectors-negative300-custom.bin'):
            w2v_wordmap = Word2Vec.load('GoogleNews-vectors-negative300-custom.bin')
        else:
            for index, row in wdf.iterrows():
                sentences.append(row[' text'].split(" "))
            w2v_wordmap = Word2Vec(sentences, size=120, window=6, min_count=1, workers=7)
            # save model
            w2v_wordmap.save('GoogleNews-vectors-negative300-custom.bin')
    w2v_wordmap = {w: vec for w, vec in zip(w2v_wordmap.wv.index2word, w2v_wordmap.wv.syn0)}
    tev = TfidfEmbeddingVectorizer(w2v_wordmap)
    with open(args['output'] + "/word2vec.txt", 'w', encoding="utf8") as of:
        w2v_dict = dict()
        for index, row in wdf.iterrows():
            word_list = row[' text'].split(" ")
            if mean:
                tev.fit(word_list)
                word_list = tev.transform(word_list)[0]
            w2v_dict[row['example_id']] = " ".join(map(str, word_list))
        for k, v in w2v_dict.items():
            of.write(k + " " + v + "\n")
        return w2v_dict
            
def postion2vec(df):
    with open(args['output'] + "/position2vec.txt", 'w') as of:
        p2v_dict = dict()
        for index, row in df.iterrows():
            word_text = nltk.word_tokenize(row[' text'])
            aspect_terms = nltk.word_tokenize(row[' aspect_term'])
            aspect_index = []
            for i, word in enumerate(word_text):
                if word in aspect_terms:
                    aspect_index.append(i)
            aspect_index.sort()
            word_list = []
            for i, word in enumerate(word_text):
                if i < aspect_index[0]:
                    word_list.append(aspect_index[0] - i)
                elif i > aspect_index[-1]:
                    word_list.append(i - aspect_index[-1])
                else:
                    word_list.append(0)
            p2v_dict[row['example_id']] = " ".join(map(str, word_list))
        for k, v in p2v_dict.items():
            of.write(k + " " + v + "\n")
        return p2v_dict
    
def idv(df):
    with open(args['output'] + "/improvedvec.txt", 'w') as of:
        idv_dict = dict()
        pos_dict = pos2vec(df)
        lex_dict = lex2vec(df)
        mg2v_dict = glove2vec(df, customTrained=False, mean = True)
        g2v_dict = glove2vec(df, customTrained=False, mean = False)
        w2v_dict = word2vec(df, customTrained=False, mean = False)
        cmw2v_dict = word2vec(df, customTrained=True, mean = True)
        mw2v_dict = word2vec(df, customTrained=False, mean = True)
        position_dict = postion2vec(df)
        for index, row in df.iterrows():
            idv_dict[row['example_id']] = mg2v_dict[row['example_id']] + " " + position_dict[row['example_id']]  + " " + lex_dict[row['example_id']] + " " + pos_dict[row['example_id']]
        for k, v in idv_dict.items():
            of.write(k + " " + v + "\n")
        


    
def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res


            
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
    optional.add_argument('-gv', '--glove2vec', help='glove2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-wv', '--word2vec', help='word2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-cwv', '--customword2vec', help='custom word2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-mgv', '--meanglove2vec', help='mean glove2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-mwv', '--meanword2vec', help='mean word2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-cgv', '--customglove2vec', help='custom glove2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-mcwv', '--meancustomword2vec', help='mean custom word2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-mcgv', '--meancustomglove2vec', help='mean custom glove2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-p2v', '--position2vec', help='position2vec (y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-idv', '--improvedvec', help='improvedvec (y/n)', choices=['y', 'n'], required=False)

    parser._action_groups.append(optional)
    args = vars(parser.parse_args())

    df = pandas.read_csv(args['input'], sep='\t')
    if args['pos2vec'] == 'y':
        pos2vec(df)
    if args['lex2vec'] == 'y':
        lex2vec(df)
    if args['glove2vec'] == 'y':
        glove2vec(df, customTrained=False, mean = False)
    if args['word2vec'] == 'y':
        word2vec(df, customTrained=False, mean = False)
    if args['customword2vec'] == 'y':
        word2vec(df, customTrained=True, mean = False)
    if args['customglove2vec'] == 'y':
        glove2vec(df, customTrained=True, mean = False)
    if args['meanglove2vec'] == 'y':
        glove2vec(df, customTrained=False, mean = True)
    if args['meanword2vec'] == 'y':
        word2vec(df, customTrained=False, mean = True)
    if args['meancustomword2vec'] == 'y':
        word2vec(df, customTrained=True, mean = True)
    if args['meancustomglove2vec'] == 'y':
        glove2vec(df, customTrained=True, mean = True)
    if args['position2vec'] == 'y':
        postion2vec(df)
    if args['improvedvec'] == 'y':
        idv(df)
        

