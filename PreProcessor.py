import argparse
import re
import autocorrect
import nltk
import pandas
import enchant
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus.reader import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from nltk.parse.stanford import StanfordDependencyParser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def split_text(df):
    print "Splitting text for merged rows..."
    cols = list(df)
    sdf = pandas.DataFrame(columns=cols)
    for index, row in df.iterrows():
        if re.search(r'\d+_\d+', row[' text']) is not None:
            typo_list = row[' text'].split('\n')
            row_list = typo_list[0].split(',')
            row_list[0] = clean_string(row_list[0])
            sdf.loc[len(sdf.index)] = [row[cols[0]]] + row_list
            for t in typo_list[1:-1]:
                row_list = t.split(',')
                row_list[1] = clean_string(row_list[1])
                sdf.loc[len(sdf.index)] = row_list
            row_list = typo_list[-1].split(',')
            row_list[1] = clean_string(row_list[1])
            row[cols[2]] = clean_string(row[cols[2]])
            sdf.loc[len(sdf.index)] = row_list + [row[cols[2]], row[cols[3]], row[cols[4]]]
        else:
            row[cols[1]] = clean_string(row[cols[1]])
            row[cols[2]] = clean_string(row[cols[2]])
            sdf.loc[len(sdf.index)] = [row[c] for c in cols]

    return sdf


def replace_emoticons(string):
    repl_str = []
    for word in string.split():
        repl_str.append(emoticons.get(word, word))
    return ' '.join(repl_str)


def clean_string(string):
    string = re.sub("\[comma]", ',', string)
    string = replace_emoticons(string)
    string = re.sub(r'[^a-zA-Z0-9\.$!?%, ]', ' ', string)
    # string = re.sub(r'[()\"\[\]_]', ' ', string)
    string = re.sub(r'\.+', '.', string)
    string = re.sub(r"\.(\w)", r' \1', string)
    string = re.sub(r'\s+', ' ', string)
    string = re.sub(r'\.\s\.', '.', string)
    return string.strip()


def remove_punc(pudf):
    print "Removing all punctuations..."
    cols = list(pudf)
    punc_df = pandas.DataFrame(columns=cols)
    for index, row in pudf.iterrows():
        punc_row = [row[c] for c in cols]
        punc_row[1] = re.sub(r'([^a-zA-Z0-9$%\'_ ])', ' ', punc_row[1])
        punc_row[1] = re.sub(r'\s+', ' ', punc_row[1]).strip()
        punc_row[2] = re.sub(r'([^a-zA-Z0-9$%\'_ ])', ' ', punc_row[2])
        punc_row[2] = re.sub(r'\s+', ' ', punc_row[2]).strip()
        if args['aspdep'] == 'y':
            punc_row[5] = re.sub(r'([^a-zA-Z0-9$%\'_ ])', ' ', punc_row[5])
            punc_row[5] = re.sub(r'\s+', ' ', punc_row[5]).strip()
        punc_df.loc[len(punc_df.index)] = punc_row
    return punc_df


def lower_case(lcdf):
    print "Converting to lowercase..."
    cols = list(lcdf)
    lc_df = pandas.DataFrame(columns=cols)
    for index, row in lcdf.iterrows():
        lc_row = [row[c] for c in cols]
        lc_row[1] = lc_row[1].lower()
        lc_row[2] = lc_row[2].lower()
        if args['aspdep'] == 'y':
            lc_row[5] = lc_row[5].lower()
        lc_df.loc[len(lc_df.index)] = lc_row
    return lc_df


def auto_correct(adf):
    print("Auto-correcting misspells...")
    cols = list(adf)
    acdf = pandas.DataFrame(columns=cols)
    for index, row in adf.iterrows():
        acdf_row = [row[c] for c in cols]
        acdf_row[1] = get_correct_spelling(row[' text'])
        acdf_row[2] = get_correct_spelling(row[' aspect_term'])
        acdf.loc[len(acdf.index)] = acdf_row
    return acdf


def segment_str(chars, exclude=None):
    words = []
    if not chars.isalpha():
        return [chars]
    if not exclude:
        exclude = set()

    working_chars = chars
    while working_chars:
        for i in range(len(working_chars), 1, -1):
            segment = working_chars[:i]
            if eng_dict.check(segment) and segment not in exclude:
                words.append(segment)
                working_chars = working_chars[i:]
                break
        else:
            if words:
                exclude.add(words[-1])
                return segment_str(chars, exclude=exclude)
            return [chars]
    return words


def get_correct_spelling(string):
    string = re.sub(r'([^a-zA-Z0-9-\'$%/ ])', r' \1', string)
    string = re.sub(r'\s+', ' ', string)
    w_list = string.strip().split(' ')
    correct_w_list = []
    tagged_list = nltk.pos_tag(w_list)
    for i in range(len(w_list)):
        if w_list[i].isalpha():
            if check_in_dict(w_list[i]) is False:
                if tagged_list[i][1] not in ['NNP', 'NNPS']:
                    if autocorrect.spell(w_list[i]) == w_list[i]:
                        seg_list = segment_str(w_list[i])
                        correct_w_list += map(lambda w: autocorrect.spell(w), seg_list)
                    else:
                        correct_w_list.append(autocorrect.spell(w_list[i]))
                else:
                    correct_w_list.append(autocorrect.spell(w_list[i]))
            else:
                correct_w_list.append(w_list[i])
        else:
            correct_w_list.append(w_list[i])

    string = ' '.join(correct_w_list)
    string = re.sub(r' ([^a-zA-Z0-9-\'$/% ])', r'\1', string)
    string = re.sub(r'\s+', ' ', string)

    return string


def check_in_dict(wrd):
    match_set = (autocorrect.common([wrd]) or autocorrect.exact([wrd]) or autocorrect.known([wrd]))
    return bool(match_set)


def remove_stop_words(stdf):
    print "Removing stop words..."
    cols = list(stdf)
    rsw_df = pandas.DataFrame(columns=cols)
    for index, row in stdf.iterrows():
        rsw_row = [row[c] for c in cols]
        rsw_text = re.sub(r'([^a-zA-Z0-9$%-\' ])', r' \1', rsw_row[1])
        rsw_word_list = rsw_text.split(' ')
        rsw_aspect_list = rsw_row[2].split(' ')
        filtered_text = []
        filtered_aspect = []
        for w in rsw_word_list:
            if w not in stopwords.words('english') or w in neg_list:
                filtered_text.append(w)
        for a in rsw_aspect_list:
            if a not in stopwords.words('english') or a in neg_list:
                filtered_aspect.append(a)
        join_text = ' '.join(filtered_text)
        rsw_row[2] = ' '.join(filtered_aspect)
        rsw_row[1] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', join_text)

        rsw_df.loc[len(rsw_df.index)] = rsw_row
    return rsw_df


def lemmatize_str(lmdf):
    print "Lemmatizing words..."
    cols = list(lmdf)
    lmw_df = pandas.DataFrame(columns=cols)
    for index, row in lmdf.iterrows():
        lmw_row = [row[c] for c in cols]
        lmw_word_list = word_tokenize(lmw_row[1])
        lmw_aspect_list = word_tokenize(lmw_row[2])
        lemmatized_w_list = []
        lemmatized_a_list = []
        word_lem_dict = dict()
        for w, tag in pos_tag(lmw_word_list):

            if w.isalpha() is True:
                lemmatized_w_list.append(lmtzr.lemmatize(w, tag_map[tag[0]]))
                word_lem_dict[w] = lmtzr.lemmatize(w, tag_map[tag[0]])
            else:
                lemmatized_w_list.append(w)
                word_lem_dict[w] = w

        for a in lmw_aspect_list:
            lemmatized_a_list.append(word_lem_dict[a])

        lemmatized_text = ' '.join(lemmatized_w_list)
        lemmatized_aspect = ' '.join(lemmatized_a_list)

        lmw_row[1] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', lemmatized_text)
        lmw_row[2] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', lemmatized_aspect)
        lmw_df.loc[len(lmw_df.index)] = lmw_row
    return lmw_df


def stem_words(psdf):
    print "Stemming words..."
    cols = list(psdf)
    psw_df = pandas.DataFrame(columns=cols)
    for index, row in psdf.iterrows():
        psw_row = [row[c] for c in cols]
        psw_word_list = word_tokenize(psw_row[1])
        psw_aspect_list = word_tokenize(psw_row[2])
        stemmed_w_list = []
        stemmed_a_list = []

        for w in psw_word_list:
            if w.isalpha() is True:
                stemmed_w_list.append(ps.stem(w))
            else:
                stemmed_w_list.append(w)
        for a in psw_aspect_list:
            if a.isalpha() is True:
                stemmed_a_list.append(ps.stem(a))
            else:
                stemmed_a_list.append(a)
        if args['aspdep'] == 'y':
            stemmed_ad_list = []
            psw_aspdep_list = word_tokenize(psw_row[5])
            for ad in psw_aspdep_list:
                if ad.isalpha() is True:
                    stemmed_ad_list.append(ps.stem(ad))
                else:
                    stemmed_ad_list.append(ad)
            stemmed_aspdep = ' '.join(stemmed_ad_list)
            psw_row[5] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', stemmed_aspdep)

        stemmed_text = ' '.join(stemmed_w_list)
        stemmed_aspect = ' '.join(stemmed_a_list)
        psw_row[1] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', stemmed_text)
        psw_row[2] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', stemmed_aspect)
        psw_df.loc[len(psw_df.index)] = psw_row
    return psw_df


def remove_proper_nouns(pndf):
    print "Removing proper nouns..."
    cols = list(pndf)
    pn_df = pandas.DataFrame(columns=cols)

    for index, row in pndf.iterrows():
        pnw_row = [row[c] for c in cols]
        pnw_text = re.sub(r'([^a-zA-Z0-9$%-\' ])', r' \1', pnw_row[1])
        pnw_text = re.sub(r'\s+', ' ', pnw_text).strip()
        pnw_word_list = pnw_text.split(' ')
        tagged_list = nltk.pos_tag(pnw_word_list)
        rm_pn_list = []
        rm_pnad_list = []
        for i in range(len(pnw_word_list)):
            if pnw_word_list[i].isalpha() is True:
                if tagged_list[i][1] not in ['NNP', 'NNPS'] or pnw_word_list[i] in row[' aspect_term']:
                    rm_pn_list.append(pnw_word_list[i])
                    if args['aspdep'] == 'y':
                        if pnw_word_list[i] in row['asp_dep_words']:
                            rm_pnad_list.append(pnw_word_list[i])

            else:
                rm_pn_list.append(pnw_word_list[i])
                if args['aspdep'] == 'y':
                    if pnw_word_list[i] in row['asp_dep_words']:
                        rm_pnad_list.append(pnw_word_list[i])

        rm_pn_text = ' '.join(rm_pn_list)
        pnw_row[1] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', rm_pn_text)
        if args['aspdep'] == 'y':
            rm_pnad_text = ' '.join(rm_pnad_list)
            pnw_row[5] = re.sub(r' ([^a-zA-Z0-9$%-\' ])', r'\1', rm_pnad_text)
        pn_df.loc[len(pn_df.index)] = pnw_row
    return pn_df


def asp_dep_set_to_list(text, dp_set):
    ad_text = re.sub(r'([^a-zA-Z0-9$%-\' ])', r' \1', text)
    ad_text = re.sub(r'\s+', ' ', ad_text).strip()
    ad_word_list = ad_text.split(' ')
    ad_list = []
    for i in range(len(ad_word_list)):
        if ad_word_list[i] in dp_set:
            ad_list.append(ad_word_list[i])

    return ad_list


def extract_aspect_related_words(ardf):
    print "Extracting aspect related words from text..."
    cols = list(ardf)
    cols.append('asp_dep_words')
    ar_df = pandas.DataFrame(columns=cols)
    count = 0
    for index, row in ardf.iterrows():
        count += 1
        print count, row['example_id']
        dep_set = set()
        result = list(sdp.raw_parse(row[' text']))
        parse_triples_list = [item for item in result[0].triples()]
        if not parse_triples_list:
            dep_set.add(row[' aspect_term'])
        else:
            for governor, dep, dependent in parse_triples_list:
                if governor[0] in row[' aspect_term'].split(' ') or dependent[0] in row[' aspect_term'].split(' '):
                    dep_set.add(governor[0])
                    dep_set.add(dependent[0])
            for governor, dep, dependent in parse_triples_list:
                if (governor[0] in dep_set and governor[0] not in row[' aspect_term']) or (
                        dependent[0] in dep_set and dependent[0] not in row[' aspect_term']):
                    if governor[1] == 'JJ' or dependent[1] == 'JJ':
                        dep_set.add(governor[0])
                        dep_set.add(dependent[0])

        ad_list = asp_dep_set_to_list(row[' text'], dep_set)
        ar_row = [row[c] for c in cols[:-1]]
        ar_row.append(' '.join(ad_list))
        ar_df.loc[len(ar_df.index)] = ar_row
    return ar_df


def generate_opinion_polarity_feature(opdf):
    print "Generating opinion polarity feature..."
    cols = list(opdf)
    cols.append('opin_polarity')
    op_df = pandas.DataFrame(columns=cols)
    for index, row in opdf.iterrows():
        snt = analyser.polarity_scores(row['asp_dep_words'])
        # del snt['compound']
        # snt_polarity = max(snt, key=snt.get)
        # f_pol = filter(lambda (i, pol): pol == snt_polarity, enumerate(polarity))[0]
        # opin_polarity = f_pol[0] - 1
        opin_polarity = 1.0
        if snt['compound'] == 0.0:
            opin_polarity = 0.0
        elif snt['compound'] < 0.0:
            opin_polarity = -1.0
        op_row = [row[c] for c in cols[:-1]]
        op_row.append(opin_polarity)
        op_df.loc[len(op_df.index)] = op_row
    return op_df


def validate_data(df):
    for index, row in df.iterrows():
        for c in list(df):
            if row[c] is None or row[c] == "":
                raise ValueError("Validation error for index=%s and example_id=%s at col=%s record=%s" % (
                    index, row['example_id'], c, df.loc[index]))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Pre-processor')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', help='path to input training data file', required=True)
    required.add_argument('-o', '--output', help='path to output processed data file', required=True)
    optional.add_argument('-sw', '--stopwords', help='remove stopwords(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-ps', '--stemwords', help='porter-stem words(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-lm', '--lemmatize', help='lemmatize(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-pn', '--propernouns', help='remove propernouns(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-pu', '--punc', help='remove punctuations(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-lo', '--lowercase', help='to lowercase(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-ad', '--aspdep', help='extract aspect dependencies(y/n)', choices=['y', 'n'],
                          required=False)
    optional.add_argument('-op', '--opinpol', help='generate opinion polarity feature(y/n)', choices=['y', 'n'],
                          required=False)

    parser._action_groups.append(optional)
    args = vars(parser.parse_args())

    df = pandas.read_csv(args['input'])
    eng_dict = enchant.Dict("en_US")

    emoticons = {
        ":-(": "sad", ":(": "sad", ":-|": "sad",
        ";-(": "sad", ";-<": "sad", "|-{": "sad",
        ":-)": "happy", ":)": "happy", ":o)": "happy",
        ":-}": "happy", ";-}": "happy", ":->": "happy",
        ";-)": "happy"
    }
    df = split_text(df)

    with open("include_dict.txt") as inc_file:
        for line in inc_file:
            autocorrect.word.KNOWN_WORDS.add(line.rstrip('\n'))
    with open("exclude_dict.txt") as exd_file:
        for line in exd_file:
            autocorrect.word.KNOWN_WORDS.remove(line.rstrip('\n'))
    df = auto_correct(df)

    # OPTIONALS

    if args['lemmatize'] == 'y':
        lmtzr = WordNetLemmatizer()
        tag_map = defaultdict(lambda: wordnet.NOUN)
        tag_map['J'] = wordnet.ADJ
        tag_map['V'] = wordnet.VERB
        tag_map['R'] = wordnet.ADV
        df = lemmatize_str(df)
    if args['stopwords'] == 'y':
        neg_list = ["no", "not", "never", "n't"]
        df = remove_stop_words(df)
    if args['aspdep'] == 'y':
        sdp = StanfordDependencyParser(
            path_to_jar="stanford-nlp-jars/stanford-corenlp-3.9.0.jar",
            path_to_models_jar="stanford-nlp-jars/stanford-corenlp-3.9.0-models.jar")
        df = extract_aspect_related_words(df)
    if args['opinpol'] == 'y':
        if args['aspdep'] == 'y':
            polarity = ['neg', 'neu', 'pos']
            analyser = SentimentIntensityAnalyzer()
            df = generate_opinion_polarity_feature(df)
        else:
            raise ValueError("set aspdep=y to generate lexicon feature!")
    if args['stemwords'] == 'y':
        ps = PorterStemmer()
        df = stem_words(df)
    if args['propernouns'] == 'y':
        df = remove_proper_nouns(df)
    if args['punc'] == 'y':
        df = remove_punc(df)
    if args['lowercase'] == 'y':
        df = lower_case(df)

    if validate_data(df):
        df.to_csv(args['output'], sep='\t', encoding="utf-8")



