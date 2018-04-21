import numpy
import pandas
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

import model_utils

if __name__ == '__main__':
    '''TRAINING'''
    train_flag = True
    if train_flag:
        train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
        train_class = train_df[' class'].as_matrix()
        train_data = model_utils.apply_aspdep_weight(train_df, 0.4)
        text_clf = MultinomialNB(alpha=0.4, fit_prior=True, class_prior=None).fit(train_data, train_class)
        joblib.dump(text_clf, 'Multinomial_nb_model.pkl')

        '''PERFORMANCE EVALUATION'''
        accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
        print("Accuracy: %s" % accuracy)
        print(clf_report)

    '''HYPER-PARAMETER TUNING'''
    tuning_flag = False
    if tuning_flag:
        clf = joblib.load('Multinomial_nb_model.pkl')
        train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
        train_data = model_utils.apply_aspdep_weight(train_df, 0.4)
        train_class = train_df[' class'].as_matrix()
        # numpy.arange(0.0, 1.1, 0.1)
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__fit_prior': (True, False),
                      'clf__alpha': numpy.arange(0.0, 1.1, 0.1)}

        gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, train_class)
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    '''TESTING'''
    test_flag = False
    if test_flag:
        sdp = StanfordDependencyParser(
            path_to_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0.jar",
            path_to_models_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0-models.jar")
        test_df = pandas.read_csv('test_1.csv', sep='\t')
        test_df = model_utils.extract_aspect_related_words(sdp, test_df[:5])
        docs_test = test_df['asp_dep_words'].as_matrix()
        clf = joblib.load('Multinomial_nb_model.pkl')
        predicted = clf.predict(docs_test)
        print(predicted)
        with open('result_1.txt', 'w') as res_file:
            for doc, category in zip(docs_test, predicted):
                print("%r => %s" % (str(doc), category))
                res_file.write("%r => %s\n" % (str(doc), category))


