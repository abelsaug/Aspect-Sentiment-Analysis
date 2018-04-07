import argparse
import re
import autocorrect
import nltk
import pandas
import enchant
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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
            sdf.loc[len(sdf.index)] = row_list + [row[cols[2]], row[cols[3]], row[cols[4]]]
        else:
            row[cols[1]] = clean_string(row[cols[1]])
            sdf.loc[len(sdf.index)] = [row[c] for c in cols]
    return sdf


def clean_string(string):
    string = re.sub("\[comma]", ',', string)
    string = re.sub(r'[^a-zA-Z0-9-\'\.$!?/%, ]', ' ', string)
    # string = re.sub(r'[()\"\[\]_]', ' ', string)
    string = re.sub(r'\s+', ' ', string)
    return string.strip()


def remove_punc(pudf):
    print "Removing all punctuations..."
    cols = list(pudf)
    punc_df = pandas.DataFrame(columns=cols)
    for index, row in pudf.iterrows():
        punc_row = [row[c] for c in cols]
        punc_row[1] = re.sub(r'([^a-zA-Z0-9$%\' ])', ' ', punc_row[1])
        punc_row[1] = re.sub(r'\s+', ' ', punc_row[1]).strip()
        punc_df.loc[len(punc_df.index)] = punc_row
    return punc_df


def lower_case(pudf):
    print "Converting to lowercase..."
    cols = list(pudf)
    punc_df = pandas.DataFrame(columns=cols)
    for index, row in pudf.iterrows():
        punc_row = [row[c] for c in cols]
        punc_row[1] = punc_row[1].lower()
        punc_df.loc[len(punc_df.index)] = punc_row
    return punc_df


def auto_correct(adf):
    print "Auto-correcting misspells..."
    cols = list(adf)
    acdf = pandas.DataFrame(columns=cols)
    for index, row in adf.iterrows():
        acdf_row = [row[c] for c in cols]
        acdf_row[1] = get_correct_spelling(row[' text'])
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
                if len(w_list[i]) >= 2 and w_list[i][0].isupper() and w_list[i][1].islower():
                    correct_w_list.append(w_list[i])
                elif tagged_list[i][1] not in ['NNP', 'NNPS']:
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
        rsw_row[1] = re.sub(r' ([^a-zA-Z-\' ])', r'\1', join_text)

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
    optional.add_argument('-s', '--stopwords', help='remove stopwords(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-r', '--stemwords', help='porter-stem words(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-n', '--propernouns', help='remove propernouns(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-p', '--punc', help='remove punctuations(y/n)', choices=['y', 'n'], required=False)
    optional.add_argument('-l', '--lowercase', help='to lowercase(y/n)', choices=['y', 'n'], required=False)

    parser._action_groups.append(optional)
    args = vars(parser.parse_args())

    ps = PorterStemmer()
    df = pandas.read_csv(args['input'])
    eng_dict = enchant.Dict("en_US")

    df = split_text(df)

    with open("include_dict.txt") as inc_file:
        for line in inc_file:
            autocorrect.word.KNOWN_WORDS.add(line.rstrip('\n'))
    with open("exclude_dict.txt") as exd_file:
        for line in exd_file:
            autocorrect.word.KNOWN_WORDS.remove(line.rstrip('\n'))
    df = auto_correct(df)

    # OPTIONALS
    if args['stopwords'] == 'y':
        df = remove_stop_words(df)
    if args['stemwords'] == 'y':
        df = stem_words(df)
    if args['propernouns'] == 'y':
        df = remove_proper_nouns(df)
    if args['punc'] == 'y':
        df = remove_punc(df)
    if args['lowercase'] == 'y':
        df = lower_case(df)

    df.to_csv(args['output'], sep='\t')






