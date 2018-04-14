import pandas
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


def apply_aspdep_weight(train_df, weight):
    train_text = train_df[' text'].values.astype('U')
    train_aspdep = train_df['asp_dep_words'].values.astype('U')
    text_count_vect = CountVectorizer()
    x_text_counts = text_count_vect.fit_transform(train_text)
    text_voc = text_count_vect.vocabulary_
    asp_dep_vect = CountVectorizer(vocabulary=text_voc)
    x_aspdep_counts = asp_dep_vect.fit_transform(train_aspdep)
    x_count_vec = x_text_counts + weight * x_aspdep_counts
    x_tfidf_vec = TfidfTransformer(use_idf=True).fit_transform(x_count_vec)
    return x_tfidf_vec


def extract_aspect_related_words(sdp, ardf):
    print("Extracting aspect related words from text...")
    cols = list(ardf)
    cols.append('asp_dep_words')
    ar_df = pandas.DataFrame(columns=cols)
    count = 0
    for index, row in ardf.iterrows():
        count += 1
        print(count)
        dep_set = set()
        result = list(sdp.raw_parse(row[' text']))
        parse_triples_list = [item for item in result[0].triples()]
        for governor, dep, dependent in parse_triples_list:
            if governor[0] in row[' aspect_term'] or dependent[0] in row[' aspect_term']:
                dep_set.add(governor[0])
                dep_set.add(dependent[0])
        ar_row = [row[c] for c in cols[:-1]]
        ar_row.append(' '.join(list(dep_set)))
        ar_df.loc[len(ar_df.index)] = ar_row
        # print
    return ar_df

def get_cv_metrics(text_clf, train_data, train_class, k_split):
    accuracy_scores = cross_val_score(text_clf,  # steps to convert raw messages      into models
                             train_data,  # training data
                             train_class,  # training labels
                             cv=k_split,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring='accuracy',  # which scoring metric?
                             n_jobs=-1,  # -1 = use all cores = faster
                             )
    cv_predicted = cross_val_predict(text_clf,
                                     train_data,
                                     train_class,
                                     cv=k_split)

    return numpy.mean(accuracy_scores), classification_report(train_class, cv_predicted)