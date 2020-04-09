import os

import numpy as np

import matplotlib.pyplot as plt
import pydotplus
from sklearn import tree
from IPython.display import Image
from scipy.special import expit

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV

import model_evaluation


def main():
    import df_handle
    
    X_train, X_test, y_train, y_test = df_handle.setup_df('train.csv', 'outClass', 'test.csv')  # import e setup Dataset


    model = DecisionTreeClassifier()  #seleziona il modello
    model.get_params(deep=False)  # per conoscere tutti i possibili iper-parametri del modello

    params_domain_dic = {'min_samples_split': range(1,200), 'min_samples_leaf': range(1,100)}  #setta il range in cui devono essere cercati gli iper-parametri migliori per il modello
    nbr_iter = 100   #setta il numero di terazioni che l'algoritmo Random Search deve eseguire per trovare i parametri migliori
    
    clf = fit_best_clf(X_train=X_train, y_train=y_train,
    model=model, nbr_iter=nbr_iter, params_domain_dic=params_domain_dic)  # fitting del miglior modello
    

    # validazione del miglior modello
    model_evaluation.validate_clf(clf=clf, X_train=X_train, y_train=y_train)   #Cross-Validation dul Training Set
    model_evaluation.feature_importance(estimator=clf, X_train=X_train, y_train=y_train)  #plot dell'importanza degli attributi

    # test del miglior modello
    model_evaluation.test_clf(clf=clf, X_test=X_test, y_test=y_test)

    #plot dei decision boundary su ogni coppia di attributi
    model_evaluation.decision_boundary_scatterplots(clf=clf, X=X_train, y=y_train)



"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html   #### OTTIMO
https://scikit-learn.org/stable/modules/grid_search.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
"""



##########################################  CLASSIFICATION  ##############################################


def fit_best_clf(X_train, y_train, model=DecisionTreeClassifier(),
params_domain_dic={'min_samples_split': range(1,1000), 'min_samples_leaf': range(1,1000)},
nbr_iter=100, score_metric='f1_macro'):

    #Random Search che cerca e fitta il miglior modello facendo 100 diversi tentativi con diversi valori degli iper-parametri
    #alla fine viene scelto il modello che porta all' f1 score maggiore.

    search = RandomizedSearchCV(model, param_distributions=params_domain_dic, n_iter=nbr_iter, scoring=score_metric, random_state=100)
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print('Best values for searched parameters: ', best_params, sep='\n')

    clf = search.best_estimator_

    del search, best_params

    return clf




def adjust_predict(clf, X_test, thr=0.5):
    y_score = clf.predict_proba(X_test)[:, 1]
    return np.array([1 if y > thr else 0 for y in y_score])



def tree_plot(tree_clf, X_train, y_names=['0', '1'], tree_depth=8):

    os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CONDA_PREFIX'] + r"\Library\bin\graphviz"

    dot_data = tree.export_graphviz(tree_clf, feature_names=X_train.columns, class_names=y_names, filled=True, rounded=True,
    special_characters=True, max_depth=tree_depth, out_file=None, )  
    
    graph = pydotplus.graph_from_dot_data(dot_data)  
    return Image(graph.create_png())




def oneDim_logisticReg(X_train, X_test, y_train, y_test, x, y='Output Class', x_max=None):

    x_train_oneDim = X_train[[x]]
    x_test_oneDim = X_test[[x]]

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train_oneDim, y_train)

    print('Coefficient:', clf.coef_, end='\n\n')
    print('Intercept:', clf.intercept_)

    print('\n')
    plt.scatter(x_test_oneDim, y_test, s=15)
    logisticReg = expit(sorted(x_test_oneDim.values) * clf.coef_ + clf.intercept_).ravel()
    plt.plot(sorted(x_test_oneDim.values), logisticReg, color='red', linewidth=3)

    plt.axis(xmin=x_test_oneDim.min()[0] - 1, xmax=x_max)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.title('Logistic Regression with ' + x, fontsize=20, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

    print('\n')
    print('Model Validation on Training Set', end='\n\n\n')
    model_evaluation.validate_clf(clf, x_train_oneDim, y_train)
    
    print('\n\n')
    print('Model Test on Test Set', end='\n\n\n')
    model_evaluation.test_clf(clf, x_test_oneDim, y_test)

    del x_train_oneDim, x_test_oneDim, clf, logisticReg





###########################################  REGRESSION  ##############################################



def oneDim_linearReg(X_train, X_test, y_train, y_test, x, y='Output Variable'):
    
    x_train_oneDim = X_train[[x]]
    x_test_oneDim = X_test[[x]]

    reg = LinearRegression()
    reg.fit(x_train_oneDim, y_train)

    print('Coefficients: \n', reg.coef_, end='\n\n')
    print('Intercept: \n', reg.intercept_)

    y_pred = reg.predict(x_test_oneDim)

    plt.scatter(x_test_oneDim, y_test, s=15)
    plt.plot(x_test_oneDim, y_pred, color='red', linewidth=3)
    
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.title('Linear Regression with ' + x, fontsize=20, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

    print('Model Test on Test Set', end='\n\n')
    model_evaluation.test_reg(reg, x_test_oneDim, y_test)

    del x_train_oneDim, x_test_oneDim, reg, y_pred








if __name__ == "__main__":
    main()