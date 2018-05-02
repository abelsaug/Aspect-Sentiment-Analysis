import numpy as np
import pandas
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from stacked_generalization import StackedGeneralizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import model_utils
# import Embedder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


def train_MultinomialNB(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()

    train_data_1 = model_utils.apply_aspdep_weight(train_df, 0.9)
    text_clf = MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None).fit(train_data_1, train_class) # 1.2, 0.01   Accuracy:  0.7407742366772097 Weights:  1.2


    joblib.dump(text_clf, 'Multinomial_nb_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_BernoulliNB(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()

    train_data_1 = model_utils.apply_aspdep_weight(train_df, 0.0)
    text_clf = BernoulliNB(alpha=0.6, fit_prior=True, class_prior=None).fit(train_data_1, train_class) # 1.2 Accuracy:  0.7047700662655685 Weights:  0.0 Alpha 0.6

    joblib.dump(text_clf, 'Bernoulli_nb_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_SGD(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()

    train_data_1 = model_utils.apply_aspdep_weight(train_df, 0.5)


    text_clf = linear_model.SGDClassifier(loss='squared_loss', penalty='l2',alpha=1e-3, random_state=607,max_iter=1000000, tol=1e-2).fit(train_data_1, train_class)        #Accuracy:  0.7797260574839455 @ 2.1 weight
    # Accuracy:  0.7428003692958715 Weights:  0.5

    joblib.dump(text_clf, 'SGD_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_SVC(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    # train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()
    train_data = model_utils.apply_aspdep_weight(train_df, 0.7)
    print train_data[0]
    text_clf = SVC(C=0.2, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma=0.5, kernel='poly',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False).fit(train_data, train_class)
    print(set(text_clf.predict(train_data)))
    print(set(train_class))
    joblib.dump(text_clf, 'model_dumps/SVC_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_RF(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()

    train_data_1 = model_utils.apply_aspdep_weight(train_df, 1.1)
    text_clf = RandomForestClassifier(n_estimators = 300, max_depth=180, random_state=607, n_jobs = -1).fit(train_data_1, train_class)
    joblib.dump(text_clf, 'RF_model.pkl')
    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy)

    print(clf_report)


def train_polarity_clf(filePath):
    train_df = pandas.read_csv(filePath, sep='\t')
    train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()
    train_data = train_df['opin_polarity'].as_matrix()
    print train_data
    # text_clf = BernoulliNB(alpha=1.0, fit_prior=True, class_prior=None).fit(train_data, train_class)
    text_clf = LogisticRegression(random_state=0).fit(train_data)
    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data, train_class, k_split=10)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_ET(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()
    train_data_1 = model_utils.apply_aspdep_weight(train_df, 0.3)
    text_clf = ExtraTreesClassifier(n_estimators = 10, max_depth=2, random_state=0, n_jobs = -1).fit(train_data_1, train_class)


    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(text_clf, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy)


#     print(clf_report)


# def train_gcForest(filePath):

# def train_xgBoost(filePath):


def train_StackedGeneralizer(filePath):
    """TRAINING"""
    train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
    train_class = train_df[' class'].as_matrix()        
    train_data_1 = model_utils.apply_aspdep_weight(train_df, 1.7)
#     base_models = [MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None), BernoulliNB(alpha=1.2, fit_prior=True, class_prior=None), 
#                    linear_model.SGDClassifier(loss='squared_loss', penalty='l2', alpha=1e-3, random_state=607,
#                                               max_iter=1000000, tol=1e-2)]

    base_models = [joblib.load('Multinomial_nb_model.pkl'),joblib.load('Bernoulli_nb_model.pkl'), joblib.load('SGD_model.pkl'), joblib.load('RF_model.pkl')]

    # define blending model
    blending_model = LogisticRegression(random_state=607)

    # initialize multi-stage model
    sg = StackedGeneralizer(base_models, blending_model, n_folds=10, verbose=False)
    sg.fit(train_data_1, train_class)
#     joblib.dump(sg, 'Stacked_model.pkl')
    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.get_cv_metrics(sg, train_data_1, train_class, k_split=10)
    print("Accuracy: ", accuracy) #Accuracy:  0.7497418660799471 Weights:  1.7
# Accuracy:  0.870205361374062 Weights:  2.4
#     print(clf_report)





# def train_VotingClassifier(filePath):
#     """TRAINING"""
#     train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
#     train_class = train_df[' class'].as_matrix()

#     train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
#     clf1 = MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None)
#     clf2 = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=607,
#                                               max_iter=1000000, tol=1e-2)
#     clf3 = BernoulliNB(alpha=1.2, fit_prior=True, class_prior=None)
#     eclf1 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='hard')
#     eclf2 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='soft')
#     eclf3 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='soft', weights=[2,2,1], flatten_transform=True)
#     eclf1.fit(train_data, train_class)
#     eclf2.fit(train_data, train_class)
#     eclf3.fit(train_data, train_class)
# #     joblib.dump(sg, 'Stacked_model.pkl')
#     """PERFORMANCE EVALUATION"""
#     accuracy, clf_report = model_utils.get_cv_metrics(eclf1, train_data, train_class, k_split=10)
#     print("Accuracy: ", accuracy) #Accuracy: 73.06
#     print(clf_report)
#     accuracy, clf_report = model_utils.get_cv_metrics(eclf2, train_data, train_class, k_split=10)
#     print("Accuracy: ", accuracy) #Accuracy: 71.58
#     print(clf_report)
#     accuracy, clf_report = model_utils.get_cv_metrics(eclf3, train_data, train_class, k_split=10)
#     print("Accuracy: ", accuracy) #Accuracy: 72.16
#     print(clf_report)

def hyperparam_tuning_MultinomialNB():
    """HYPER-PARAMETER TUNING"""
    clf = joblib.load('Multinomial_nb_model.pkl')

    train_df = pandas.read_csv('out_data_2/data_2_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.9)
    train_class = train_df[' class'].as_matrix()
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__fit_prior': (True, False),
                  'clf__alpha': np.arange(0.0, 1.1, 0.1)}
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_SVC():
    """HYPER-PARAMETER TUNING"""
    clf = SVC()

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()
    parameters = {
        'C': np.arange(1, 5, 1).tolist(),
        'kernel': ['rbf', 'poly'],  # precomputed,'poly', 'sigmoid'
        'degree': np.arange(0, 3, 1).tolist(),
        'gamma': np.arange(0.0, 1.0, 0.1).tolist(),
        'coef0': np.arange(0.0, 1.0, 0.1).tolist(),
        'shrinking': [True],
        'probability': [False],
        'tol': np.arange(0.001, 0.01, 0.001).tolist(),
        'cache_size': [2000],
        'class_weight': [None],
        'verbose': [False],
        'max_iter': [-1],
        'random_state': [None],
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_SGD():
    """HYPER-PARAMETER TUNING"""
    clf = linear_model.SGDClassifier()

    train_df = pandas.read_csv('out_data_2/data_2_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()
    parameters = {'loss': ['hinge', 'huber', 'squared_loss'],
                  'max_iter': np.arange(20, 100, 50),
                  'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                  'alpha': np.arange(0.001, 1, 0.5)}
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def final_testing():  # TODO fix test_data input transform to clf
    """TESTING"""
    sdp = StanfordDependencyParser(
        path_to_jar="stanford-nlp-jars/stanford-corenlp-3.9.0.jar",
        path_to_models_jar="stanford-nlp-jars/stanford-corenlp-3.9.0-models.jar")
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


if __name__ == '__main__':
    # fileLists = ['out_data_1/data_1_lm.csv','out_data_1/data_1_lm_pn.csv','out_data_1/data_1_lm_ps.csv','out_data_1/data_1_lm_sw.csv','out_data_1/data_1_lm_sw_ps.csv','out_data_1/data_1_lm_sw_ps_pn.csv','out_data_1/data_1_pn.csv','out_data_1/data_1_ps.csv','out_data_1/data_1_sw.csv','out_data_1/data_1_sw_ps_pn.csv']

    # fileLists = ['out_data_2/data_2.csv',
    #              'out_data_2/data_2_lm.csv',
    #              'out_data_2/data_2_lm_pn.csv',
    #              'out_data_2/data_2_lm_ps.csv',
    #              'out_data_2/data_2_lm_sw.csv',
    #              'out_data_2/data_2_lm_sw_ps.csv',
    #              'out_data_2/data_2_lm_sw_ps_pn.csv',
    #              'out_data_2/data_2_pn.csv',
    #              'out_data_2/data_2_ps.csv',
    #              'out_data_2/data_2_sw.csv',
    #              'out_data_2/data_2_sw_ps.csv',
    #              'out_data_2/data_2_sw_ps_pn.csv']
    #     fileLists = ['out_data_1/data_1_lm.csv','out_data_1/data_1_lm_ps.csv','out_data_1/data_1_ps.csv','out_data_1/data_1_sw.csv','out_data_1/data_1_sw_ps.csv','out_data_1/data_1_sw_ps_pn.csv']
    #     fileLists = ['out_data_1/data_1_pn.csv', 'out_data_1/data_1_ps.csv', 'out_data_1/data_1_sw.csv']
    #     fileLists = ['out_data_1/data_1_pn.csv']
    fileLists = ['out_data_1/data_1_sw.csv']
    for fileno, filePath in enumerate(fileLists):
        # print("Bernoulli NB for file No: ", fileno)
        # train_BernoulliNB(filePath)

        # print("SGD for file No: ", fileno)
        # train_SGD(filePath)
        # print("SVC for file No: ", fileno)
        # train_SVC(filePath)
        # print("Stacked Generalizer for file No: ", fileno)
        # train_StackedGeneralizer(filePath)
        #         print("Voting Classifier for file No: ", fileno)
        #         train_VotingClassifier(filePath)
        # print("Opinion polarity classifier for file No: ", fileno)
        # train_polarity_clf(filePath)
        # print("Multinomial NB for file No: ", fileno)
        # train_MultinomialNB(filePath)
        #         print("Bernoulli NB for file No: ", fileno)
        #         train_BernoulliNB(filePath)
        # print("Random Forest for file No: ", fileno)
        # train_RF(filePath)
#           train_ET(filePath)
#         print("SGD for file No: ", fileno)
#         train_SGD(filePath)
#         print("Stacked Generalizer for file No: ", fileno)
#         train_StackedGeneralizer(filePath)
#         print("Voting Classifier for file No: ", fileno)
#         train_VotingClassifier(filePath)


        hyperparam_tuning_SVC()
