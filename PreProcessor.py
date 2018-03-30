import pandas
import re
import enchant
from collections import Counter
import difflib
from autocorrect import spell
from nltk.corpus import stopwords
import nltk
import argparse
from nltk.stem import PorterStemmer


def split_text(df):
    print "Splitting text for merged rows..."
    cols = list(df)
    sdf = pandas.DataFrame(columns=cols)
    for index, row in df.iterrows():
        # print row
        if re.search(r'\d+_\d+', row[' text']) is not None:
            typo_list = row[' text'].split('\n')
            row_list = typo_list[0].split(',')
            row_list[0] = clean_string(row_list[0])
            # print [row[cols[0]]] + row_list
            sdf.loc[len(sdf.index)] = [row[cols[0]]] + row_list
            for t in typo_list[1:-1]:
                row_list = t.split(',')
                row_list[1] = clean_string(row_list[1])
                sdf.loc[len(sdf.index)] = row_list
            row_list = typo_list[-1].split(',')
            row_list[1] = clean_string(row_list[1])
            sdf.loc[len(sdf.index)] = row_list + [row[cols[2]], row[cols[3]], row[cols[4]]]
        else:
            row[cols[1]] = clean_string(row[cols[1]])
            sdf.loc[len(sdf.index)] = [row[c] for c in cols]
    return sdf


def clean_string(string):
    string = re.sub("\[comma]", ',', string)
    string = re.sub(r'[^a-zA-Z0-9-\'\.$!?, ]', '', string)
    string = re.sub(r'\s+', ' ', string)
    return string.strip()


def auto_correct(adf):
    print "Auto-correcting misspells..."
    cols = list(adf)
    acdf = pandas.DataFrame(columns=cols)
    for index, row in adf.iterrows():
        acdf_row = [row[c] for c in cols]
        print row['example_id']
        acdf_row[1] = get_correct_spelling(row[' text'])
        # acdf_row[2] = get_correct_spelling(row[' aspect_term'])
        acdf.loc[len(acdf.index)] = acdf_row
    return acdf


def get_correct_spelling(string):
    string = re.sub(r'([^a-zA-Z-\' ])', r' \1', string)
    string = re.sub(r'\s+', ' ', string)
    w_list = string.strip().split(' ')
    # print w_list
    tagged_list = nltk.pos_tag(w_list)
    # print tagged_list
    for i in range(len(w_list)):
        lower_w = w_list[i].lower()
        if w_list[i].isalpha() and my_dict.check(lower_w) is False:
            if tagged_list[i][1] not in ['NNP', 'NNPS']:
                # a = my_dict.suggest(w_list[i])
                # best_ratio = 0.0
                # best_words = []
                # for b in a:
                #     tmp = difflib.SequenceMatcher(None, w_list[i], b).ratio()
                #     if tmp > best_ratio:
                #         best_words = [b]
                #         best_ratio = tmp
                #     elif tmp == best_ratio:
                #         best_words.append(b)
                # print w_list[i], tagged_list[i][1], best_words
                # w_list[i] = best_words[0]  # TODO -- choose most relevant word
                # print w_list[i], spell(w_list[i])
                w_list[i] = spell(w_list[i])
    string = ' '.join(w_list)
    string = re.sub(r' ([^a-zA-Z-\' ])', r'\1', string)
    string = re.sub(r'\s+', ' ', string)

    return string


def remove_stop_words(stdf):
    print "Removing stop words..."
    cols = list(stdf)
    rsw_df = pandas.DataFrame(columns=cols)
    for index, row in stdf.iterrows():
        rsw_row = [row[c] for c in cols]
        rsw_text = re.sub(r'([^a-zA-Z-\' ])', r' \1', rsw_row[1])
        rsw_word_list = rsw_text.split(' ')
        filtered_text = []
        for w in rsw_word_list:
            if w.isalpha() is True:
                if w not in stopwords.words('english'):
                    filtered_text.append(w)
            else:
                filtered_text.append(w)
        join_text = ' '.join(filtered_text)
        # print row[' text']
        rsw_row[1] = re.sub(r' ([^a-zA-Z-\' ])', r'\1', join_text)
        # print rsw_row[1]
        rsw_df.loc[len(rsw_df.index)] = rsw_row
    return rsw_df


def stem_words(psdf):
    print "Stemming words..."
    cols = list(psdf)
    psw_df = pandas.DataFrame(columns=cols)
    for index, row in psdf.iterrows():
        psw_row = [row[c] for c in cols]
        psw_text = re.sub(r'([^a-zA-Z-\' ])', r' \1', psw_row[1])
        psw_word_list = psw_text.split(' ')
        stemmed_list = []
        for w in psw_word_list:
            if w.isalpha() is True:
                stemmed_list.append(ps.stem(w))
            else:
                stemmed_list.append(w)
        stemmed_text = ' '.join(stemmed_list)
        psw_row[1] = re.sub(r' ([^a-zA-Z-\' ])', r'\1', stemmed_text)
        psw_df.loc[len(psw_df.index)] = psw_row
    return psw_df


def remove_proper_nouns(pndf):
    print "Removing proper nouns..."
    cols = list(pndf)
    pn_df = pandas.DataFrame(columns=cols)

    for index, row in pndf.iterrows():
        pnw_row = [row[c] for c in cols]
        pnw_text = re.sub(r'([^a-zA-Z-\' ])', r' \1', pnw_row[1])
        pnw_text = re.sub(r'\s+', ' ', pnw_text).strip()
        pnw_word_list = pnw_text.split(' ')
        # print pnw_word_list
        tagged_list = nltk.pos_tag(pnw_word_list)
        rm_pn_list = []
        for i in range(len(pnw_word_list)):
            if pnw_word_list[i].isalpha() is True:
                if tagged_list[i][1] not in ['NNP', 'NNPS']:
                    rm_pn_list.append(pnw_word_list[i])
            else:
                rm_pn_list.append(pnw_word_list[i])
        rm_pn_text = ' '.join(rm_pn_list)
        pnw_row[1] = re.sub(r' ([^a-zA-Z-\' ])', r'\1', rm_pn_text)
        pn_df.loc[len(pn_df.index)] = pnw_row
    return pn_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Pre-processor')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', help='path to input training data file', required=True)
    required.add_argument('-o', '--output', help='path to output processed data file', required=True)
    optional.add_argument('-m', '--model', help='training model(Naive Baye\'s)', required=False)  #TODO reqd. model-feature engg.
    optional.add_argument('-s', '--stopwords', help='remove stopwords(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-p', '--stemwords', help='porter-stem words(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-n', '--propernouns', help='remove propernouns(y/n)', choices=['y', 'n'], required=False)
    parser._action_groups.append(optional)
    args = vars(parser.parse_args())

    en_dict = enchant.DictWithPWL("en_US")
    ps = PorterStemmer()
    df = pandas.read_csv(args['input'])
    df = split_text(df)

    text_set = set(df[' text'].tolist())
    text_str = ' '.join(text_set).lower()
    text_str = re.sub(r'[^a-zA-Z-\' ]', '', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)
    word_list = text_str.strip().split(' ')
    word_freqs = Counter(word_list)
    # print word_freqs
    f = open('pwl_dict.txt', 'wr+')
    f.truncate()
    f.close()
    my_file = open('pwl_dict.txt', 'a')
    for word in word_freqs:
        if word_freqs[word] > 3 and en_dict.check(word) is False:
            my_file.write(word + "\n")
    my_file.close()
    my_dict = enchant.DictWithPWL("en_US", "pwl_dict.txt")

    df = auto_correct(df)

    # OPTIONALS
    if args['stopwords'] == 'y':
        df = remove_stop_words(df)
    if args['stemwords'] == 'y':
        df = stem_words(df)
    if args['propernouns'] == 'y':
        df = remove_proper_nouns(df)

    df.to_csv(args['output'], sep='\t')






