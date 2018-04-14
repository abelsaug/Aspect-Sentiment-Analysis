import numpy as np
import pandas
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
import stacked_generalization
from sklearn.linear_model import LogisticRegression


import model_utils


def train_MultinomialNB():
    '''TRAINING'''
    train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
    train_class = train_df[' class'].as_matrix()
    train_class_1 = []

    for classes in train_class:
        if classes !=0:
            train_class_1.append(1)
        else:
            train_class_1.append(0)
    train_class = np.asarray(train_class_1)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
    text_clf = MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None).fit(train_data, train_class) # 0.3, 0.6   Accuracy:  0.6977029991497758
    joblib.dump(text_clf, 'Multinomial_nb_neutral_model.pkl')



#         '''PERFORMANCE EVALUATION'''
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)

        
def train_BernoulliNB():
    '''TRAINING'''
    train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
    train_class = train_df[' class'].as_matrix()
    train_class_1 = []

    for classes in train_class:
        if classes !=0:
            train_class_1.append(1)
        else:
            train_class_1.append(0)
    train_class = np.asarray(train_class_1)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
    text_clf = BernoulliNB(alpha=1.2, fit_prior=True, class_prior=None).fit(train_data, train_class) # 1.2 Accuracy:  0.6808777267210551
    joblib.dump(text_clf, 'Bernoulli_nb_neutral_model.pkl')

#         '''PERFORMANCE EVALUATION'''
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)

        
def train_SGD():
    '''TRAINING'''
    train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
    train_class = train_df[' class'].as_matrix()
    train_class_1 = []

    for classes in train_class:
        if classes !=0:
            train_class_1.append(1)
        else:
            train_class_1.append(0)
    train_class = np.asarray(train_class_1)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.3)


    text_clf = linear_model.SGDClassifier(loss='squared_loss', penalty='l2',alpha=1e-3, random_state=607,max_iter=1000000, tol=1e-2).fit(train_data, train_class)        #Accuracy:  0.7106302368296371 
    joblib.dump(text_clf, 'SGD_nb_neutral_model.pkl')



#         '''PERFORMANCE EVALUATION'''
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)

        
def train_StackedGeneralizer():
    '''TRAINING'''
    
    train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
    train_class = train_df[' class'].as_matrix()
    train_class_1 = []

    for classes in train_class:
        if classes !=0:
            train_class_1.append(1)
        else:
            train_class_1.append(0)
    train_class = np.asarray(train_class_1)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
    base_models = [BernoulliNB(alpha=1.2, fit_prior=True, class_prior=None), MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None), linear_model.SGDClassifier(loss='squared_loss', penalty='l2',alpha=1e-3, random_state=607,max_iter=1000000, tol=1e-2)]

    # define blending model
    blending_model = LogisticRegression()

    # initialize multi-stage model
    sg = StackedGeneralizer(base_models, blending_model, n_folds=10, verbose=True)

    sg.fit(train_data, train_class)
#         '''PERFORMANCE EVALUATION'''
    accuracy, clf_report = model_utils.get_cv_metrics(sg, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)

def hyperparam_tuning_MultinomialNB():    
    '''HYPER-PARAMETER TUNING'''
    clf = joblib.load('Multinomial_nb_model.pkl')
    train_df = pandas.read_csv('preproc_train_1_1.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
    train_class = train_df[' class'].as_matrix()
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__fit_prior': (True, False),
                  'clf__alpha': numpy.arange(0.0, 1.1, 0.1)}
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    
def final_testing():
    '''TESTING'''
    sdp = StanfordDependencyParser(
        path_to_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0.jar",
        path_to_models_jar="/home/philip/Documents/Sem 2/NLP/stanford-corenlp-full-2018-01-31/stanford-corenlp-3.9.0-models.jar")
    test_df = pandas.read_csv('test_1_1.csv', sep='\t')
    test_df = model_utils.extract_aspect_related_words(sdp, test_df[:5])
    docs_test = test_df['asp_dep_words'].as_matrix()
    clf = joblib.load('Multinomial_nb_model.pkl')
    predicted = clf.predict(docs_test)
    print(predicted)
    with open('result_1.txt', 'w') as res_file:
        for doc, category in zip(docs_test, predicted):
#                 print "%r => %s" % (str(doc), category) //TODO: Convert this line to Python 3
            res_file.write("%r => %s\n" % (str(doc), category))

if __name__ == '__main__':
    train_StackedGeneralizer()
