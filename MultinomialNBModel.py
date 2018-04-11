import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import numpy
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def extract_aspect_related_words(ardf):
    print "Extracting aspect related words from text..."
    cols = list(ardf)
    cols.append('asp_dep_words')
    ar_df = pandas.DataFrame(columns=cols)
    count = 0
    for index, row in ardf.iterrows():
        count += 1
        print count
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


if __name__ == '__main__':
    '''TRAINING'''
    train_flag = True
    if train_flag:
        train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')

        # train_data = train_df['asp_dep_words'].values.astype('U')
        train_data = train_df['asp_dep_words'].values.astype('U')
        train_class = train_df[' class'].as_matrix()
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB(alpha=0.4, fit_prior=True, class_prior=None))])
        text_clf.fit(train_data, train_class)
        joblib.dump(text_clf, 'Multinomial_nb_model.pkl')

        '''PERFORMANCE EVALUATION'''
        scores = cross_val_score(text_clf,  # steps to convert raw messages      into models
                                 train_data,  # training data
                                 train_class,  # training labels
                                 cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                                 scoring='accuracy',  # which scoring metric?
                                 n_jobs=-1,  # -1 = use all cores = faster
                                 )
        print "Accuracy: %s" % (numpy.mean(scores))
        cv_predicted = cross_val_predict(text_clf,
                                    train_df['asp_dep_words'].values.astype('U'),
                                    train_df[' class'].as_matrix(),
                                    cv=10)
        print classification_report(train_class, cv_predicted)

    '''HYPER-PARAMETER TUNING'''
    tuning_flag = False
    if tuning_flag:
        clf = joblib.load('Multinomial_nb_model.pkl')
        train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
        train_data = train_df['asp_dep_words'].values.astype('U')
        train_class = train_df[' class'].as_matrix()
        # numpy.arange(0.0, 1.1, 0.1)
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__fit_prior': (True, False),
                      'clf__alpha': numpy.arange(0.0, 1.1, 0.1)}

        gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, train_class)
        print gs_clf.best_score_
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    '''TESTING'''
    test_flag = False
    if test_flag:
        sdp = StanfordDependencyParser(
            path_to_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0.jar",
            path_to_models_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0-models.jar")
        test_df = pandas.read_csv('test_1.csv', sep='\t')
        test_df = extract_aspect_related_words(test_df[:5])
        docs_test = test_df['asp_dep_words'].as_matrix()
        clf = joblib.load('Multinomial_nb_model.pkl')
        predicted = clf.predict(docs_test)
        print predicted
        with open('result_1.txt', 'w') as res_file:
            for doc, category in zip(docs_test, predicted):
                print "%r => %s" % (str(doc), category)
                res_file.write("%r => %s\n" % (str(doc), category))


