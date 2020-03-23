import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import RandomizedSearchCV

import df_handle
import model_evaluation

"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html   #### OTTIMO
https://scikit-learn.org/stable/modules/grid_search.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
"""




""" CLASSIFICATION """


def fit_best_clf(X_train, y_train, model=DecisionTreeClassifier(), params_domain_dic={'min_samples_leaf': range(1,150)}, iterN=100):

    search = RandomizedSearchCV(model, param_distributions=params_domain_dic, n_iter=iterN, scoring='f1_macro', random_state=100)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print('Best values for searched parameters: ', best_params, sep='\n')

    clf = search.best_estimator_

    return clf




def adjust_predict(clf, X_test, thr=0.5):
    y_score = clf.predict_proba(X_test)[:, 1]
    return np.array([1 if y > thr else 0 for y in y_score])







""" REGRESSION """






def main():

    # import e setup Dataset

    X_train, X_test, y_train, y_test = df_handle.setup_df('train.csv', 'outClass', 'test.csv')


    # selezione del modello, del dominio dei parametri in cui cercare e del numero di ricerche

    model = DecisionTreeClassifier()
    
    model.get_params(deep=False)
    params_domain_dic = {'min_samples_leaf': range(1,100)}
    iterN = 100

    # fitting del miglior modello

    clf = fit_best_clf(X_train, y_train,
    model=model, iterN=iterN, params_domain_dic=params_domain_dic)
    
    
    
    # valutazione del miglior modello

    model_evaluation.validate_clf(clf, X_train, y_train)

    model_evaluation.test_clf(clf, X_test, y_test)



if __name__ == "__main__":
    main()